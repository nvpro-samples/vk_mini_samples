/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#ifndef NO_TEXTURES
#define USE_TEXTURES
#endif


struct MeshState
{
  float3 N; // Normal
  float3 T; // Tangent
  float3 B; // Bitangent
  float3 Ng; // Geometric normal
  float2 tc[2]; // Texture coordinates
  bool isInside;
};

static const float g_min_reflectance = 0.04F;

// If both anisotropic roughness values fall below this threshold, the BSDF switches to specular.
#define MICROFACET_MIN_ROUGHNESS 0.0014142f


#ifdef __SLANG__
#define TextureType Sampler2D
#else
#define TextureType Texture2D
#endif


float4 getTexture(in GltfTextureInfo tinfo, in MeshState mstate, in TextureType texturesMap[], in SamplerState samplers)
{
  // KHR_texture_transform
  float3 t = float3(mstate.tc[tinfo.texCoord].x, mstate.tc[tinfo.texCoord].y, 1.0);
  float3 tt = mul(tinfo.uvTransform, t);
  float2 texCoord = tt.xy;
#ifdef __SLANG__
  return texturesMap[tinfo.index].Sample(texCoord);
#else
  return texturesMap[tinfo.index].SampleLevel(samplers, texCoord, 0);
#endif
}

bool isTexturePresent(in GltfTextureInfo tinfo)
{
  return tinfo.index > -1;
}


//-----------------------------------------------------------------------
// From the incoming material return the material for evaluating PBR
//-----------------------------------------------------------------------
PbrMaterial evaluateMaterial(GltfShadeMaterial material, MeshState mesh
#ifdef USE_TEXTURES
  , TextureType texturesMap[]
  , SamplerState samplers
#endif
)
{
   // Material Evaluated
  PbrMaterial pbrMat;

  // Base Color/Albedo may be defined from a base texture or a flat color
  float4 baseColor = material.pbrBaseColorFactor;
  if (isTexturePresent(material.pbrBaseColorTexture))
  {
    baseColor *= getTexture(material.pbrBaseColorTexture, mesh, texturesMap, samplers);
  }
  pbrMat.baseColor = baseColor.rgb;
  pbrMat.opacity = baseColor.a;


  // Metallic-Roughness
  float roughness = material.pbrRoughnessFactor;
  float metallic = material.pbrMetallicFactor;
  if (isTexturePresent(material.pbrMetallicRoughnessTexture))
  {
    // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
    float4 mr_sample = getTexture(material.pbrMetallicRoughnessTexture, mesh, texturesMap, samplers);
    roughness *= mr_sample.g;
    metallic *= mr_sample.b;
  }
  float r2 = roughness * roughness;
  pbrMat.roughness = float2(r2, r2); // Square roughness for the microfacet model
  pbrMat.metallic = clamp(metallic, 0.0F, 1.0F);

  // Normal Map
  float3 normal = mesh.N;
  float3 tangent = mesh.T;
  float3 bitangent = mesh.B;
  if (isTexturePresent(material.normalTexture))
  {
    float3x3 tbn = float3x3(mesh.T, mesh.B, mesh.N);
    float3 normal_vector = getTexture(material.normalTexture, mesh, texturesMap, samplers).xyz;
    normal_vector = normal_vector * 2.0F - 1.0F;
    normal_vector *= float3(material.normalTextureScale, material.normalTextureScale, 1.0F);
    normal = normalize(mul(normal_vector, tbn));
  }
  // We should always have B = cross(N, T) * bitangentSign. Orthonormalize,
  // in case the normal was changed or the input wasn't orthonormal:
  pbrMat.N = normal;
  pbrMat.B = cross(pbrMat.N, tangent);
  float bitangentSign = sign(dot(bitangent, pbrMat.B));
  pbrMat.B = normalize(pbrMat.B) * bitangentSign;
  pbrMat.T = normalize(cross(pbrMat.B, pbrMat.N)) * bitangentSign;
  pbrMat.Ng = mesh.Ng;

  // Emissive term
  float3 emissive = material.emissiveFactor;
  if (isTexturePresent(material.emissiveTexture))
  {
    emissive *= float3(getTexture(material.emissiveTexture, mesh, texturesMap, samplers).xyz);
  }
  pbrMat.emissive = max(float3(0.0F, 0.0F, 0.0F), emissive);

  // KHR_materials_specular
  // https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_specular
  pbrMat.specularColor = material.specularColorFactor;
  if (isTexturePresent(material.specularColorTexture))
  {
    pbrMat.specularColor *= getTexture(material.specularColorTexture, mesh, texturesMap, samplers).rgb;
  }

  pbrMat.specular = material.specularFactor;
  if (isTexturePresent(material.specularTexture))
  {
    pbrMat.specular *= getTexture(material.specularTexture, mesh, texturesMap, samplers).a;
  }


  // Dielectric Specular
  float ior1 = 1.0F; // IOR of the current medium (e.g., air)
  float ior2 = material.ior; // IOR of the material
  if (mesh.isInside && (material.thicknessFactor > 0))  // If the material is thin-walled, we don't need to consider the inside IOR.
  {
    ior1 = material.ior;
    ior2 = 1.0F;
  }
  pbrMat.ior1 = ior1;
  pbrMat.ior2 = ior2;


  // KHR_materials_transmission
  pbrMat.transmission = material.transmissionFactor;
  if (isTexturePresent(material.transmissionTexture))
  {
    pbrMat.transmission *= getTexture(material.transmissionTexture, mesh, texturesMap, samplers).r;
  }

  // KHR_materials_volume
  pbrMat.attenuationColor = material.attenuationColor;
  pbrMat.attenuationDistance = material.attenuationDistance;
  pbrMat.thickness = material.thicknessFactor;

  // KHR_materials_clearcoat
  pbrMat.clearcoat = material.clearcoatFactor;
  pbrMat.clearcoatRoughness = material.clearcoatRoughness;
  pbrMat.Nc = pbrMat.N;
  
  if (isTexturePresent(material.clearcoatTexture))
  {
    pbrMat.clearcoat *= getTexture(material.clearcoatTexture, mesh, texturesMap, samplers).r;
  }

  if (isTexturePresent(material.clearcoatRoughnessTexture))
  {
    pbrMat.clearcoatRoughness *= getTexture(material.clearcoatRoughnessTexture, mesh, texturesMap, samplers).g;
  }
  pbrMat.clearcoatRoughness = max(pbrMat.clearcoatRoughness, 0.001F);

  if (isTexturePresent(material.clearcoatNormalTexture))
  {
    float3x3 tbn = float3x3(pbrMat.T, pbrMat.B, pbrMat.Nc);
    float3 normal_vector = getTexture(material.clearcoatNormalTexture, mesh, texturesMap, samplers).xyz;
    normal_vector = normal_vector * 2.0F - 1.0F;
    pbrMat.Nc = normalize(mul(normal_vector, tbn));
  }

  // Iridescence
  float iridescence = material.iridescenceFactor;
  if (isTexturePresent(material.iridescenceTexture))
  {
    iridescence *= getTexture(material.iridescenceTexture, mesh, texturesMap, samplers).x;
  }
  float iridescenceThickness = material.iridescenceThicknessMaximum;
  if (isTexturePresent(material.iridescenceThicknessTexture))
  {
    const float t = getTexture(material.iridescenceThicknessTexture, mesh, texturesMap, samplers).y;
    iridescenceThickness = lerp(material.iridescenceThicknessMinimum, material.iridescenceThicknessMaximum, t);
  }
  pbrMat.iridescence = (iridescenceThickness > 0.0f) ? iridescence : 0.0f; // No iridescence when the thickness is zero.
  pbrMat.iridescenceIor = material.iridescenceIor;
  pbrMat.iridescenceThickness = iridescenceThickness;

  // KHR_materials_anisotropy
  float anisotropyStrength = material.anisotropyStrength;
  float2 anisotropyDirection = float2(1.0f, 0.0f); // By default the anisotropy strength is along the tangent.
  if (isTexturePresent(material.anisotropyTexture))
  {
    const float4 anisotropyTex = getTexture(material.anisotropyTexture, mesh, texturesMap, samplers);

    // .xy encodes the direction in (tangent, bitangent) space. Remap from [0, 1] to [-1, 1].
    anisotropyDirection = normalize(float2(anisotropyTex.xy) * 2.0f - 1.0f);
    // .z encodes the strength in range [0, 1].
    anisotropyStrength *= anisotropyTex.z;
  }

  // If the anisotropyStrength == 0.0f (default), the roughness is isotropic.
  // No need to rotate the anisotropyDirection or tangent space.
  if (anisotropyStrength > 0.0F)
  {
    pbrMat.roughness.x = lerp(pbrMat.roughness.y, 1.0f, anisotropyStrength * anisotropyStrength);

    const float s = material.anisotropyRotation.x;
    const float c = material.anisotropyRotation.y;

    anisotropyDirection =
        float2(c * anisotropyDirection.x + s * anisotropyDirection.y, c * anisotropyDirection.y - s * anisotropyDirection.x);

    const float3 T_aniso = pbrMat.T * anisotropyDirection.x + pbrMat.B * anisotropyDirection.y;

    pbrMat.B = normalize(cross(pbrMat.N, T_aniso));
    pbrMat.T = cross(pbrMat.B, pbrMat.N);
  }


  // KHR_materials_sheen
  float3 sheenColor = material.sheenColorFactor;
  if (isTexturePresent(material.sheenColorTexture))
  {
    sheenColor *= float3(getTexture(material.sheenColorTexture, mesh, texturesMap, samplers).xyz); // sRGB
  }
  pbrMat.sheenColor = sheenColor; // No sheen if this is black.

  float sheenRoughness = material.sheenRoughnessFactor;
  if (isTexturePresent(material.sheenRoughnessTexture))
  {
    sheenRoughness *= getTexture(material.sheenRoughnessTexture, mesh, texturesMap, samplers).w;
  }
  sheenRoughness = max(MICROFACET_MIN_ROUGHNESS, sheenRoughness);
  pbrMat.sheenRoughness = sheenRoughness;

  return pbrMat;
}


#endif  // MAT_EVAL_H
