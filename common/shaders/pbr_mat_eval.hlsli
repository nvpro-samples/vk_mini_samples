/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


//-------------------------------------------------------------------------------------------------
// This file takes the incoming GltfShadeMaterial (material uploaded in a buffer) and
// evaluates it, basically sample the textures and return the struct PbrMaterial
// which is used by the Bsdf functions to evaluate and sample the material
//

#ifndef MAT_EVAL_H
#define MAT_EVAL_H 1

#include "pbr_mat_struct.hlsli"
#include "dh_scn_desc.hlsli"

//  https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB

////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// MATERIAL FOR EVALUATION
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------
static const float g_min_reflectance = 0.04F;
//-----------------------------------------------------------------------


// sRGB to linear approximation, see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
float4 srgbToLinear(in float4 sRgb)
{
  //return float4(pow(sRgb.xyz, float3(2.2f)), sRgb.w);
  float3 rgb = sRgb.xyz * (sRgb.xyz * (sRgb.xyz * 0.305306011F + 0.682171111F) + 0.012522878F);
  return float4(rgb, sRgb.a);
}


//-----------------------------------------------------------------------
// From the incoming material return the material for evaluating PBR
//-----------------------------------------------------------------------
PbrMaterial evaluateMaterial(in GltfShadeMaterial material,
#ifdef __SLANG__
  in Sampler2D texturesMap[], 
#else
  in Texture2D texturesMap[],
  in SamplerState samplers,
#endif
  in float3 normal,
  in float3 tangent,
  in float3 bitangent,
  in float2 texCoord,
  in bool isInside
)
{
  float perceptual_roughness = 0.0F;
  float metallic = 0.0F;
  float3 f0 = float3(0.0F, 0.0F, 0.0F);
  float3 f90 = float3(1.0F, 1.0F, 1.0F);
  float4 baseColor = float4(0.0F, 0.0F, 0.0F, 1.0F);

  // Normal Map
  if(material.normalTexture > -1)
  {
    float3x3 tbn = float3x3(tangent, bitangent, normal);
#ifdef __SLANG__
    float3 normal_vector = texturesMap[material.normalTexture].Sample(texCoord).xyz;
#else
    float3 normal_vector = texturesMap[material.normalTexture].SampleLevel(samplers, texCoord.xy, 0).xyz;
#endif
    normal_vector = normal_vector * 2.0F - 1.0F;
    normal_vector *= float3(material.normalTextureScale, material.normalTextureScale, 1.0F);
    normal = normalize(mul(normal_vector, tbn));
  }


  // Metallic-Roughness
  {
    perceptual_roughness = material.pbrRoughnessFactor;
    metallic = material.pbrMetallicFactor;
    if(material.pbrMetallicRoughnessTexture > -1)
    {
      // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
#ifdef __SLANG__
      float4 mr_sample = texturesMap[material.pbrMetallicRoughnessTexture].Sample(texCoord);
#else
      float4 mr_sample = texturesMap[material.pbrMetallicRoughnessTexture].SampleLevel(samplers, texCoord.xy, 0);
#endif
      perceptual_roughness *= mr_sample.g;
      metallic *= mr_sample.b;
    }

    // The albedo may be defined from a base texture or a flat color
    baseColor = material.pbrBaseColorFactor;
    if(material.pbrBaseColorTexture > -1)
    {
#ifdef __SLANG__
      baseColor *= texturesMap[material.pbrBaseColorTexture].Sample(texCoord);
#else
      baseColor *= texturesMap[material.pbrBaseColorTexture].SampleLevel(samplers, texCoord.xy, 0);
#endif
    }
    float3 specular_color = lerp(float3(g_min_reflectance, g_min_reflectance, g_min_reflectance), baseColor.xyz, metallic);
    f0 = specular_color;
  }

  // Protection
  metallic = clamp(metallic, 0.0F, 1.0F);


  // Emissive term
  float3 emissive = material.emissiveFactor;
  if(material.emissiveTexture > -1)
  {
#ifdef __SLANG__
    emissive *= float3(texturesMap[material.emissiveTexture].Sample(texCoord).xyz);
#else
    emissive *= float3(texturesMap[material.emissiveTexture].SampleLevel(samplers, texCoord.xy, 0).xyz);
#endif
  }
  
  // KHR_materials_specular
  // https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_specular
  float4 specularColorTexture = float4(1.0F, 1.0F, 1.0F, 1.0F);
  if(material.specularColorTexture > -1)
  {
#ifdef __SLANG__
    specularColorTexture = texturesMap[material.specularColorTexture].SampleLevel(texCoord, 0);
#else
    specularColorTexture = texturesMap[material.specularColorTexture].SampleLevel(samplers, texCoord.xy, 0);
#endif
  }
  float specularTexture = 1.0F;
  if(material.specularTexture > -1)
  {
#ifdef __SLANG__
    specularTexture = texturesMap[material.specularTexture].SampleLevel(texCoord, 0).a;
#else
    specularTexture = texturesMap[material.specularTexture].SampleLevel(samplers, texCoord.xy, 0).a;
#endif
  }
  

  // Dielectric Specular
  float ior1 = 1.0F;
  float ior2 = material.ior;
  if(isInside)
  {
    ior1 = material.ior;
    ior2 = 1.0F;
  }
  float iorRatio = ((ior1 - ior2) / (ior1 + ior2));
  float iorRatioSqr = iorRatio * iorRatio;

  float3 dielectricSpecularF0 = material.specularColorFactor * specularColorTexture.rgb;
  float dielectricSpecularF90 = material.specularFactor * specularTexture;

  f0 = lerp(min(iorRatioSqr * dielectricSpecularF0, float3(1.0F, 1.0F, 1.0F)) * dielectricSpecularF0, baseColor.rgb, metallic);
  float tempf90 = lerp(dielectricSpecularF90, 1.0F, metallic);
  f90 = float3(tempf90, tempf90, tempf90);
  

  // Material Evaluated
  PbrMaterial pbrMat;
  pbrMat.albedo = baseColor;
  pbrMat.f0 = f0;
  pbrMat.f90 = f90;
  pbrMat.roughness = perceptual_roughness;
  pbrMat.metallic = metallic;
  pbrMat.emissive = max(float3(0.0F, 0.0F, 0.0F), emissive);
  pbrMat.normal = normal;
  pbrMat.eta = (material.thicknessFactor == 0.0F) ? 1.0F : ior1 / ior2;


  // KHR_materials_transmission
  pbrMat.transmissionFactor = material.transmissionFactor;
  if(material.transmissionTexture > -1)
  {
#ifdef __SLANG__
    pbrMat.transmissionFactor *= texturesMap[material.transmissionTexture].SampleLevel(texCoord, 0).r;
#else
    pbrMat.transmissionFactor *= texturesMap[material.transmissionTexture].SampleLevel(samplers, texCoord.xy, 0).r;
#endif
  }

  // KHR_materials_ior
  pbrMat.ior = material.ior;

  // KHR_materials_volume
  pbrMat.attenuationColor = material.attenuationColor;
  pbrMat.attenuationDistance = material.attenuationDistance;
  pbrMat.thicknessFactor = material.thicknessFactor;

  // KHR_materials_clearcoat
  pbrMat.clearcoatFactor = material.clearcoatFactor;
  pbrMat.clearcoatRoughness = material.clearcoatRoughness;
  if(material.clearcoatTexture > -1)
  {
#ifdef __SLANG__
    pbrMat.clearcoatFactor *= texturesMap[material.clearcoatTexture].SampleLevel(texCoord, 0).r;
#else
    pbrMat.clearcoatFactor *= texturesMap[material.clearcoatTexture].SampleLevel(samplers, texCoord.xy, 0).r;
#endif
  }
  if(material.clearcoatRoughnessTexture > -1)
  {
#ifdef __SLANG__
    pbrMat.clearcoatRoughness *= texturesMap[material.clearcoatRoughnessTexture].SampleLevel(texCoord, 0).g;
#else
    pbrMat.clearcoatRoughness *= texturesMap[material.clearcoatRoughnessTexture].SampleLevel(samplers, texCoord.xy, 0).g;
#endif
  }
  pbrMat.clearcoatRoughness = max(pbrMat.clearcoatRoughness, 0.001);

  return pbrMat;
}

PbrMaterial evaluateMaterial(in GltfShadeMaterial material,
#ifdef __SLANG__
  in Sampler2D texturesMap[], 
#else
  in Texture2D texturesMap[],
  in SamplerState samplers,
#endif
  in float3 normal,
  in float3 tangent,
  in float3 bitangent,
  in float2 texCoord)
{
  return evaluateMaterial(material,
#ifdef __SLANG__
  texturesMap, 
#else
  texturesMap,
  samplers,
#endif 
  normal, tangent, bitangent, texCoord, false);
}

#endif  // MAT_EVAL_H
