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
};

// sRGB to linear approximation, see http://chilliant.blogspot.com/2012/08/srgb-approximations-for-hlsl.html
vec4 srgbToLinear(in vec4 sRGB)
{
  //return vec4(pow(sRGB.xyz, vec3(2.2f)), sRGB.w);
  vec3 RGB = sRGB.xyz * (sRGB.xyz * (sRGB.xyz * 0.305306011 + 0.682171111) + 0.012522878);
  return vec4(RGB, sRGB.a);
}

//-------------------------------------------------------------------------------------------------
// Specular-Glossiness converter
// See: // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_pbrSpecularGlossiness/examples/convert-between-workflows/js/three.pbrUtilities.js#L34
//-------------------------------------------------------------------------------------------------
float getPerceivedBrightness(vec3 vector)
{
  return sqrt(0.299f * vector.x * vector.x + 0.587f * vector.y * vector.y + 0.114f * vector.z * vector.z);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
const float c_MinReflectance = 0.04;

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
  // Metallic and Roughness material properties are packed together. In glTF, these factors can be specified by fixed scalar values or from a metallic-roughness map
  MaterialEval res;
  res.roughness = 0.f;
  res.metallic  = 0.f;
  res.albedo    = vec4(0.f, 0.f, 0.f, 1.f);

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
    res.roughness = material.pbrRoughnessFactor;
    res.metallic  = material.pbrMetallicFactor;
    if(material.pbrMetallicRoughnessTexture > -1)
    {
      // Roughness is stored in the 'g' channel, metallic is stored in the 'b' channel.
      vec4 mrSample = texture(texturesMap[nonuniformEXT(material.pbrMetallicRoughnessTexture)], uv);
      res.roughness *= mrSample.g;
      res.metallic *= mrSample.b;
    }

    // The albedo may be defined from a base texture or a flat color
    res.albedo = material.pbrBaseColorFactor;
    if(material.pbrBaseColorTexture > -1)
    {
      res.albedo *= srgbToLinear(texture(texturesMap[nonuniformEXT(material.pbrBaseColorTexture)], uv));
    }
  }
  // Specular-Glossiness which will be converted to metallic-roughness
  else if(material.shadingModel == MATERIAL_SPECULARGLOSSINESS)
  {
    float roughness = 0.0;
    float metallic  = 0.0;
    vec4  baseColor = vec4(0.0, 0.0, 0.0, 1.0);

    vec3 f0   = material.khrSpecularFactor;
    roughness = 1.0 - material.khrGlossinessFactor;

    if(material.khrSpecularGlossinessTexture > -1)
    {
      vec4 sgSample = srgbToLinear(textureLod(texturesMap[nonuniformEXT(material.khrSpecularGlossinessTexture)], uv, 0));
      roughness     = 1 - material.khrGlossinessFactor * sgSample.a;  // glossiness to roughness
      f0 *= sgSample.rgb;                                             // specular
    }

    vec3  specularColor            = f0;  // f0 = specular
    float oneMinusSpecularStrength = 1.0 - max(max(f0.r, f0.g), f0.b);

    vec4 diffuseColor = material.khrDiffuseFactor;
    if(material.khrDiffuseTexture > -1)
      diffuseColor *= srgbToLinear(textureLod(texturesMap[nonuniformEXT(material.khrDiffuseTexture)], uv, 0));

    baseColor.rgb = diffuseColor.rgb * oneMinusSpecularStrength;
    metallic      = solveMetallic(diffuseColor.rgb, specularColor, oneMinusSpecularStrength);

    res.metallic  = metallic;
    res.albedo    = baseColor;
    res.roughness = roughness;
  }

  // Clamping results
  res.roughness = clamp(res.roughness, 0.001, 1.f);
  res.metallic  = clamp(res.metallic, 0.f, 1.f);

  // Emissive term
  vec3 emissive = material.emissiveFactor;
  if(material.emissiveTexture > -1)
    emissive *= vec3(srgbToLinear(texture(texturesMap[material.emissiveTexture], uv)));

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
//-----------------------------------------------------------------------
vec3 lambertian(vec3 f0, vec3 f90, vec3 diffuseColor, float VdotH)
{
  // see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
  return (1.0 - F_Schlick(f0, f90, VdotH)) * (diffuseColor / M_PI);
}

vec3 lambertian(vec3 diffuseColor, float metallic)
{
  return (1.0 - metallic) * (diffuseColor / M_PI);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 specularGGX(vec3 f0, vec3 f90, float alphaRoughness, float VdotH, float NdotL, float NdotV, float NdotH)
{
  vec3  F = F_Schlick(f0, f90, VdotH);
  float V = V_GGX(NdotL, NdotV, alphaRoughness);
  float D = D_GGX(NdotH, max(0.001, alphaRoughness));

  return F * V * D;
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
vec3 evalDiffuseGltf(MaterialEval state, vec3 f0, vec3 f90, vec3 V, vec3 L, vec3 H, out float pdf)
{
  vec3 N      = state.normal;
  pdf         = 0;
  float NdotV = dot(N, V);
  float NdotL = dot(N, L);

  if(NdotL < 0.0 || NdotV < 0.0)
    return vec3(0.0);

  NdotL = clamp(NdotL, 0.001, 1.0);


  pdf = NdotL * M_1_PI;
  return lambertian(state.albedo.xyz, state.metallic);
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 evalSpecularGltf(MaterialEval state, vec3 f0, vec3 f90, vec3 V, vec3 L, vec3 H, out float pdf)
{
  vec3 N      = state.normal;
  pdf         = 0;
  float NdotL = dot(N, L);

  if(NdotL < 0.0)
    return vec3(0.0);

  float NdotV = dot(N, V);
  float NdotH = clamp(dot(N, H), 0, 1);
  float LdotH = clamp(dot(L, H), 0, 1);
  float VdotH = clamp(dot(V, H), 0, 1);

  NdotL = clamp(NdotL, 0.001, 1.0);
  NdotV = clamp(abs(NdotV), 0.001, 1.0);


  pdf = D_GGX(NdotH, state.roughness) * NdotH / (4.0 * LdotH);
  return specularGGX(f0, f90, state.roughness, VdotH, NdotL, NdotV, NdotH);
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void brdfEvalSeparate(in MaterialEval state, vec3 V, vec3 L, out vec3 brdfDiff, out float pdfDiff, out vec3 brdfSpec, out float pdfSpec)
{
  vec3 N = state.normal;
  vec3 H = normalize(L + V);

  float diffuseRatio  = 0.5 * (1.0 - state.metallic);
  float specularRatio = 1.0 - diffuseRatio;

  // Compute reflectance.
  // Anything less than 2% is physically impossible and is instead considered to be shadowing. Compare to "Real-Time-Rendering" 4th editon on page 325.
  vec3  specularCol = mix(vec3(c_MinReflectance), vec3(state.albedo), float(state.metallic));
  float reflectance = max(max(specularCol.r, specularCol.g), specularCol.b);
  vec3  f0          = specularCol;
  vec3  f90         = vec3(clamp(reflectance * 50.0, 0.0, 1.0));

  // Diffuse
  brdfDiff = evalDiffuseGltf(state, f0, f90, V, L, H, pdfDiff);
  pdfDiff *= diffuseRatio;

  // Specular
  brdfSpec = evalSpecularGltf(state, f0, f90, V, L, H, pdfSpec);
  pdfSpec *= specularRatio;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 pbrEval(in MaterialEval state, in vec3 V, in vec3 L, inout float pdf)
{
  float pdfDiff, pdfSpec;
  vec3  brdfDiff, brdfSpec;
  brdfEvalSeparate(state, V, L, brdfDiff, pdfDiff, brdfSpec, pdfSpec);

  pdf = pdfDiff + pdfSpec;
  return brdfDiff + brdfSpec;
}


// Return a direction for the sampling
vec3 bsdfSample(in MaterialEval state, in vec3 V, in vec3 randVal)
{
  vec3  N           = state.normal;
  float probability = randVal.x;

  float diffuseRatio = 0.5 * (1.0 - state.metallic);
  float r1           = randVal.y;
  float r2           = randVal.z;
  vec3  T            = state.tangent;
  vec3  B            = state.bitangent;

  if(probability < diffuseRatio)
  {  // Diffuse
    vec3 L = cosineSampleHemisphere(r1, r2);
    return L.x * T + L.y * B + L.z * N;
  }
  else
  {  // Specular
    float roughness = state.roughness;
    vec3  H         = ggxSampling(roughness, r1, r2);
    H               = T * H.x + B * H.y + N * H.z;
    return reflect(-V, H);
  }
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 pbrSample(in MaterialEval state, vec3 V, inout vec3 L, inout float pdf, in vec3 randVal)
{
  // Find boucing ray
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
