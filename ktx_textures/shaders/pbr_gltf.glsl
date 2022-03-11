/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

//-------------------------------------------------------------------------------------------------
// This fils has all the Gltf sampling and evaluation methods


#ifndef PBR_GLTF_GLSL
#define PBR_GLTF_GLSL 1

//  https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

// GLTF material
#define MATERIAL_METALLICROUGHNESS 0
#define MATERIAL_SPECULARGLOSSINESS 1
#define ALPHA_OPAQUE 0
#define ALPHA_MASK 1
#define ALPHA_BLEND 2

#ifndef M_1_PI
#define M_1_PI 0.318309886183790671538f  // 1/pi
#endif

#ifndef M_PI
#define M_PI 3.14159862f
#endif

#ifndef M_TWO_PI
#define M_TWO_PI 6.28318530717958648
#endif

//-----------------------------------------------------------------------
const float c_MinReflectance = 0.04;
//-----------------------------------------------------------------------


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// MATERIAL FOR EVALUATION
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

struct MaterialEval
{
  vec4  albedo;
  float roughness;
  float metallic;
  vec3  normal;
  vec3  tangent;
  vec3  bitangent;
  vec3  emissive;
  vec3  f0;
};

// sRGB to linear approximation, see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec4 srgbToLinear(in vec4 sRGB)
{
  if(frameInfo.isSrgb > 0)
    return sRGB;
  //return vec4(pow(sRGB.xyz, vec3(2.2f)), sRGB.w);
  vec3 RGB = sRGB.xyz * (sRGB.xyz * (sRGB.xyz * 0.305306011 + 0.682171111) + 0.012522878);
  return vec4(RGB, sRGB.a);
}

float clampedDot(vec3 x, vec3 y)
{
  return clamp(dot(x, y), 0.0, 1.0);
}

float getPerceivedBrightness(vec3 vector)
{
  return sqrt(0.299f * vector.x * vector.x + 0.587f * vector.y * vector.y + 0.114f * vector.z * vector.z);
}


//-------------------------------------------------------------------------------------------------
// Specular-Glossiness converter
// See: // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows/js/three.pbrUtilities.js#L34
//-------------------------------------------------------------------------------------------------
float solveMetallic(vec3 diffuse, vec3 specular, float oneMinusSpecularStrength)
{
  float specularBrightness = getPerceivedBrightness(specular);

  if(specularBrightness < c_MinReflectance)
  {
    return 0.f;
  }

  float diffuseBrightness = getPerceivedBrightness(diffuse);

  float a = c_MinReflectance;
  float b = diffuseBrightness * oneMinusSpecularStrength / (1.f - c_MinReflectance) + specularBrightness - 2.f * c_MinReflectance;
  float c = c_MinReflectance - specularBrightness;
  float D = max(b * b - 4.0 * a * c, 0);

  return clamp((-b + sqrt(D)) / (2.f * a), 0.f, 1.f);
}


//-----------------------------------------------------------------------
// From the incoming material return the material for evaluating PBR
//-----------------------------------------------------------------------
MaterialEval evaluateMaterial(in GltfShadeMaterial material, in vec3 normal, in vec3 tangent, in vec3 bitangent, in vec2 uv)
{
  float perceptualRoughness = 0.0;
  float metallic            = 0.0;
  vec3  f0                  = vec3(0.0);
  vec4  baseColor           = vec4(0.0, 0.0, 0.0, 1.0);

  // Normal Map
  if(material.normalTexture > -1)
  {
    mat3 TBN          = mat3(tangent, bitangent, normal);
    vec3 normalVector = texture(texturesMap[nonuniformEXT(material.normalTexture)], uv).xyz;
    normalVector      = normalize(normalVector * 2.f - 1.f);
    normalVector *= vec3(material.normalTextureScale, material.normalTextureScale, 1.0);
    normal    = normalize(TBN * normalVector);
    tangent   = cross(bitangent, normal);
    bitangent = cross(normal, tangent);
  }


  if(material.shadingModel == MATERIAL_METALLICROUGHNESS)
  {
    perceptualRoughness = material.pbrRoughnessFactor;
    metallic            = material.pbrMetallicFactor;
    if(material.pbrMetallicRoughnessTexture > -1)
    {
      // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
      vec4 mrSample = texture(texturesMap[nonuniformEXT(material.pbrMetallicRoughnessTexture)], uv);
      perceptualRoughness *= mrSample.g;
      metallic *= mrSample.b;
    }

    // The albedo may be defined from a base texture or a flat color
    baseColor = material.pbrBaseColorFactor;
    if(material.pbrBaseColorTexture > -1)
    {
      baseColor *= srgbToLinear(texture(texturesMap[nonuniformEXT(material.pbrBaseColorTexture)], uv));
    }
    vec3 specularCol = mix(vec3(c_MinReflectance), vec3(baseColor), float(metallic));
    f0               = specularCol;
  }
  // Specular-Glossiness which will be converted to metallic-roughness
  else if(material.shadingModel == MATERIAL_SPECULARGLOSSINESS)
  {
    f0                  = material.khrSpecularFactor;
    perceptualRoughness = material.khrGlossinessFactor;

    if(material.khrSpecularGlossinessTexture > -1)
    {
      vec4 sgSample = srgbToLinear(textureLod(texturesMap[nonuniformEXT(material.khrSpecularGlossinessTexture)], uv, 0));
      perceptualRoughness *= sgSample.a;  // glossiness to roughness
      f0 *= sgSample.rgb;                 // specular
    }

    perceptualRoughness = 1.0 - perceptualRoughness;

    vec4 diffuseColor = material.khrDiffuseFactor;
    if(material.khrDiffuseTexture > -1)
    {
      diffuseColor *= srgbToLinear(textureLod(texturesMap[nonuniformEXT(material.khrDiffuseTexture)], uv, 0));
    }

    vec3  specularColor            = f0;  // f0 = specular
    float oneMinusSpecularStrength = 1.0 - max(max(f0.r, f0.g), f0.b);
    baseColor.rgb                  = diffuseColor.rgb * oneMinusSpecularStrength;
    metallic                       = solveMetallic(diffuseColor.rgb, specularColor, oneMinusSpecularStrength);
  }

  // Emissive term
  vec3 emissive = material.emissiveFactor;
  if(material.emissiveTexture > -1)
  {
    emissive *= vec3(srgbToLinear(texture(texturesMap[material.emissiveTexture], uv)));
  }


  // Material Evaluated
  MaterialEval res;
  res.albedo    = baseColor;
  res.f0        = f0;
  res.roughness = clamp(perceptualRoughness, 0.001, 1.0);
  res.metallic  = clamp(metallic, 0.0, 1.0);
  res.emissive  = emissive;
  res.normal    = normal;
  res.tangent   = tangent;
  res.bitangent = bitangent;

  return res;
}


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
// PBR - EVALUATION
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 cosineSampleHemisphere(float r1, float r2)
{
  vec3  dir;
  float r   = sqrt(r1);
  float phi = M_TWO_PI * r2;
  dir.x     = r * cos(phi);
  dir.y     = r * sin(phi);
  dir.z     = sqrt(max(0.0, 1.0 - dir.x * dir.x - dir.y * dir.y));

  return dir;
}

//-----------------------------------------------------------------------
// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
//-----------------------------------------------------------------------
vec3 F_Schlick(vec3 f0, vec3 f90, float VdotH)
{
  return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

float F_Schlick(float f0, float f90, float VdotH)
{
  return f0 + (f90 - f0) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
}

//-----------------------------------------------------------------------
// Smith Joint GGX
// Note: Vis = G / (4 * NdotL * NdotV)
// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
// see Real-Time Rendering. Page 331 to 336.
// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
//-----------------------------------------------------------------------
float V_GGX(float NdotL, float NdotV, float alphaRoughness)
{
  float alphaRoughnessSq = alphaRoughness * alphaRoughness;

  float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);
  float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - alphaRoughnessSq) + alphaRoughnessSq);

  float GGX = GGXV + GGXL;
  if(GGX > 0.0)
  {
    return 0.5 / GGX;
  }
  return 0.0;
}

//-----------------------------------------------------------------------
// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
//-----------------------------------------------------------------------
float D_GGX(float NdotH, float alphaRoughness)
{
  float alphaRoughnessSq = alphaRoughness * alphaRoughness;
  float f                = (NdotH * NdotH) * (alphaRoughnessSq - 1.0) + 1.0;
  return alphaRoughnessSq / (M_PI * f * f);
}

//-----------------------------------------------------------------------
// https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
//-----------------------------------------------------------------------
vec3 BRDF_lambertian(vec3 f0, vec3 f90, vec3 diffuseColor, float VdotH)
{
  // see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
  return (1.0 - F_Schlick(f0, f90, VdotH)) * (diffuseColor / M_PI);
}

vec3 BRDF_lambertian(vec3 diffuseColor, float metallic)
{
  return (1.0 - metallic) * (diffuseColor / M_PI);
}

//-----------------------------------------------------------------------
// https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
//-----------------------------------------------------------------------
vec3 BRDF_specularGGX(vec3 f0, vec3 f90, float alphaRoughness, float VdotH, float NdotL, float NdotV, float NdotH)
{
  vec3  F   = F_Schlick(f0, f90, VdotH);
  float Vis = V_GGX(NdotL, NdotV, alphaRoughness);
  float D   = D_GGX(NdotH, alphaRoughness);

  return F * Vis * D;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 ggxSampling(float specularAlpha, float r1, float r2)
{
  float phi = r1 * 2.0 * M_PI;

  float cosTheta = sqrt((1.0 - r2) / (1.0 + (specularAlpha * specularAlpha - 1.0) * r2));
  float sinTheta = clamp(sqrt(1.0 - (cosTheta * cosTheta)), 0.0, 1.0);
  float sinPhi   = sin(phi);
  float cosPhi   = cos(phi);

  return vec3(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 pbrEval(in MaterialEval state, in vec3 V, in vec3 L, inout float pdf)
{
  // L = Direction from surface point to light
  // V = Direction from surface point to Eye
  vec3  N     = state.normal;
  vec3  H     = normalize(L + V);
  float NdotL = clampedDot(N, L);
  float NdotV = clampedDot(N, V);
  float NdotH = clampedDot(N, H);
  float LdotH = clampedDot(L, H);
  float VdotH = clampedDot(V, H);

  vec3 f_specular = vec3(0.0);
  vec3 f_diffuse  = vec3(0.0);

  if(NdotL > 0.0 || NdotV > 0.0)
  {
    float diffuseRatio  = 0.5 * (1.0 - state.metallic);
    float specularRatio = 1.0 - diffuseRatio;

    vec3  f90            = vec3(1.0);
    float specularWeight = 1.0;
    float alphaRoughness = state.roughness;

    f_diffuse  = BRDF_lambertian(state.albedo.xyz, state.metallic);
    f_specular = BRDF_specularGGX(state.f0, f90, alphaRoughness, VdotH, NdotL, NdotV, NdotH);

    pdf += diffuseRatio * (NdotL * M_1_PI);                                       // diffuse
    pdf += specularRatio * D_GGX(NdotH, alphaRoughness) * NdotH / (4.0 * LdotH);  // specular
  }

  return (f_diffuse + f_specular) * NdotL;
}


//-----------------------------------------------------------------------
// Return a direction for the sampling
//-----------------------------------------------------------------------
vec3 bsdfSample(in MaterialEval state, in vec3 V, in vec3 randVal)
{
  vec3  N           = state.normal;
  float probability = randVal.x;

  float diffuseRatio = 0.5 * (1.0 - state.metallic);
  float r1           = randVal.y;
  float r2           = randVal.z;
  vec3  T            = state.tangent;
  vec3  B            = state.bitangent;

  if(probability <= diffuseRatio)
  {  // Diffuse
    vec3 L = cosineSampleHemisphere(r1, r2);
    return L.x * T + L.y * B + L.z * N;
  }
  else
  {  // Specular
    float specularAlpha = state.roughness;
    vec3  H             = ggxSampling(specularAlpha, r1, r2);
    H                   = T * H.x + B * H.y + N * H.z;
    return reflect(-V, H);
  }
}


//-----------------------------------------------------------------------
// Sample and evaluate
//-----------------------------------------------------------------------
vec3 pbrSample(in MaterialEval state, vec3 V, inout vec3 L, inout float pdf, in vec3 randVal)
{
  // Find bouncing ray
  L = bsdfSample(state, V, randVal);

  vec3  N     = state.normal;
  float dotNL = dot(N, L);

  // Early out
  pdf = 0;
  if(dotNL < 0)
    return vec3(0);

  // Sampling with new light direction
  return pbrEval(state, V, L, pdf);
}

#endif  // PBR_GLTF_GLSL
