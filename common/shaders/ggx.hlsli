/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef GGX_HLSLI
#define GGX_HLSLI 1

#include "constants.hlsli"
#include "functions.hlsli"

struct PbrMaterial
{
  float4 albedo;
  float roughness;
  float metallic;
  float3 normal;
  float3 emissive;
  float3 f0;
  // KHR_materials_transmission
  float transmission;
  // KHR_materials_ior
  float ior;
  // KHR_materials_volume
  float3 attenuationColor;
  float attenuationDistance;
  bool thinWalled;
  // KHR_materials_clearcoat
  float clearcoat;
  float clearcoatRoughness;
};


#define BSDF_EVENT_ABSORB 0
#define BSDF_EVENT_DIFFUSE 1
#define BSDF_EVENT_GLOSSY (1 << 1)
#define BSDF_EVENT_SPECULAR (1 << 2)
#define BSDF_EVENT_REFLECTION (1 << 3)
#define BSDF_EVENT_TRANSMISSION (1 << 4)

#define BSDF_EVENT_DIFFUSE_REFLECTION (BSDF_EVENT_DIFFUSE | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_DIFFUSE_TRANSMISSION (BSDF_EVENT_DIFFUSE | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_GLOSSY_REFLECTION (BSDF_EVENT_GLOSSY | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_GLOSSY_TRANSMISSION (BSDF_EVENT_GLOSSY | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_SPECULAR_REFLECTION (BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_SPECULAR_TRANSMISSION (BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION)

#define BSDF_USE_MATERIAL_IOR (-1.0)

struct BsdfEvaluateData
{
  float3 ior1; // [in] inside ior
  float3 ior2; // [in] outside ior
  float3 k1; // [in] Toward the incoming ray
  float3 k2; // [in] Toward the sampled light
  float3 bsdf_diffuse; // [out] Diffuse contribution
  float3 bsdf_glossy; // [out] Specular contribution
  float pdf; // [out] PDF
};

struct BsdfSampleData
{
  float3 ior1; // [in] inside ior
  float3 ior2; // [in] outside ior
  float3 k1; // [in] Toward the incoming ray
  float3 k2; // [in] Toward the sampled light
  float4 xi; // [in] 4 random [0..1]
  float pdf; // [out] PDF
  float3 bsdf_over_pdf; // [out] contribution / PDF
  int event_type; // [out] one of the event above
};




//-----------------------------------------------------------------------
// The following equation models the Fresnel reflectance term of the spec equation (aka F())
// Implementation of fresnel from [4], Equation 15
//-----------------------------------------------------------------------
float3 fresnelSchlick(float3 f0, float3 f90, float VdotH)
{
  float a = 1.0 - VdotH;
  float a2 = a * a;
  return f0 + (f90 - f0) * a2 * a2 * a;
}

//-----------------------------------------------------------------------
// Smith Joint GGX
// Note: Vis = G / (4 * NdotL * NdotV)
// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
// see Real-Time Rendering. Page 331 to 336.
// see https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
//-----------------------------------------------------------------------
float smithJointGGX(float NdotL, float NdotV, float alphaRoughness)
{
  float alphaRoughnessSq = max(alphaRoughness * alphaRoughness, 1e-07);

  float ggxV = NdotL * sqrt(NdotV * NdotV * (1.0F - alphaRoughnessSq) + alphaRoughnessSq);
  float ggxL = NdotV * sqrt(NdotL * NdotL * (1.0F - alphaRoughnessSq) + alphaRoughnessSq);

  float ggx = ggxV + ggxL;
  if (ggx > 0.0F)
  {
    return 0.5F / ggx;
  }
  return 0.0F;
}

//-----------------------------------------------------------------------
// The following equation(s) model the distribution of microfacet normals across the area being drawn (aka D())
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// Follows the distribution function recommended in the SIGGRAPH 2013 course notes from EPIC Games [1], Equation 3.
//-----------------------------------------------------------------------
float distributionGGX(float NdotH, float alphaRoughness)  // alphaRoughness    = roughness * roughness;
{
  float alphaSqr = max(alphaRoughness * alphaRoughness, 1e-07);

  float NdotHSqr = NdotH * NdotH;
  float denom = NdotHSqr * (alphaSqr - 1.0) + 1.0;

  return alphaSqr / (M_PI * denom * denom);
}

//-----------------------------------------------------------------------
// https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
//-----------------------------------------------------------------------
float3 brdfLambertian(float3 diffuseColor, float metallic)
{
  return (1.0F - metallic) * (diffuseColor / M_PI);
}

//https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
float3 brdfLambertian(float3 f0, float3 f90, float3 diffuseColor, float specularWeight, float VdotH)
{
  // see https://seblagarde.wordpress.com/2012/01/08/pi-or-not-to-pi-in-game-lighting-equation/
  return (1.0 - (specularWeight * fresnelSchlick(f0, f90, VdotH))) * (diffuseColor / M_PI);
}

//-----------------------------------------------------------------------
// https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#acknowledgments AppendixB
//-----------------------------------------------------------------------
float3 brdfSpecularGGX(float3 f0, float3 f90, float alphaRoughness, float VdotH, float NdotL, float NdotV, float NdotH)
{
  float3 f = fresnelSchlick(f0, f90, VdotH);
  float vis = smithJointGGX(NdotL, NdotV, alphaRoughness); // Vis = G / (4 * NdotL * NdotV)
  float d = distributionGGX(NdotH, alphaRoughness);

  return f * vis * d;
}

//-----------------------------------------------------------------------
// Sample the GGX distribution
// - Return the half vector
//-----------------------------------------------------------------------
float3 ggxSampling(float alphaRoughness, float r1, float r2)
{
  float alphaSqr = max(alphaRoughness * alphaRoughness, 1e-07);

  float phi      = 2.0 * M_PI * r1;
  float cosTheta = sqrt((1.0 - r2) / (1.0 + (alphaSqr - 1.0) * r2));
  float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

  return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}



float3 ggxEvaluate(float3 V, float3 N, float3 L, float3 albedo, float metallic, float roughness)
{
  float3 H = normalize(L + V);
  float NdotL = saturate(dot(N, L));
  float NdotV = saturate(dot(N, V));
  float NdotH = saturate(dot(N, H));
  float VdotH = saturate(dot(V, H));

  float3 c_min_reflectance = float3(0.04, 0.04, 0.04);
  float3 f0 = lerp(c_min_reflectance, albedo, metallic);
  float3 f90 = float3(1.0, 1.0, 1.0);

  float3 f_diffuse = brdfLambertian(albedo, metallic);
  float3 f_specular = brdfSpecularGGX(f0, f90, roughness, VdotH, NdotL, NdotV, NdotH);

  float3 color = (f_diffuse + f_specular) * NdotL;
  return color;
}



void bsdfEvaluate(inout BsdfEvaluateData data, in PbrMaterial mat)
{
  // Initialization
  float3 surfaceNormal = mat.normal;
  float3 viewDir = data.k1;
  float3 lightDir = data.k2;
  float3 albedo = mat.albedo.rgb;
  float metallic = mat.metallic;
  float roughness = mat.roughness;
  float3 f0 = mat.f0;
  float3 f90 = float3(1.0F,1.0F,1.0F);

  // Specular roughness
  float alpha = roughness * roughness;

  // Compute half vector
  float3 halfVector = normalize(viewDir + lightDir);

  // Compute various "angles" between vectors
  float NdotV = clampedDot(surfaceNormal, viewDir);
  float NdotL = clampedDot(surfaceNormal, lightDir);
  float VdotH = clampedDot(viewDir, halfVector);
  float NdotH = clampedDot(surfaceNormal, halfVector);
  float LdotH = clampedDot(lightDir, halfVector);

  // Contribution
  float3 f_diffuse = brdfLambertian(albedo, metallic);
  float3 f_specular = brdfSpecularGGX(f0, f90, alpha, VdotH, NdotL, NdotV, NdotH);

  // Calculate PDF (probability density function)
  float diffuseRatio = 0.5F * (1.0F - metallic);
  float diffusePDF = (NdotL * M_1_OVER_PI);
  float specularPDF = distributionGGX(NdotH, alpha) * NdotH / (4.0F * LdotH);

  // Results
  data.bsdf_diffuse = f_diffuse * NdotL;
  data.bsdf_glossy = f_specular * NdotL;
  data.pdf = lerp(specularPDF, diffusePDF, diffuseRatio);
}

void bsdfSample(inout BsdfSampleData data, in PbrMaterial mat)
{
  // Initialization
  float3 surfaceNormal = mat.normal;
  float3 viewDir = data.k1;
  float roughness = mat.roughness;
  float metallic = mat.metallic;

  // Random numbers for importance sampling
  float r1 = data.xi.x;
  float r2 = data.xi.y;
  float r3 = data.xi.z;

  // Create tangent space
  float3 tangent, binormal;
  orthonormalBasis(surfaceNormal, tangent, binormal);

  // Specular roughness
  float alpha = roughness * roughness;

  // Find Half vector for diffuse or glossy reflection
  float diffuseRatio = 0.5F * (1.0F - metallic);
  float3 halfVector;
  if (r3 < diffuseRatio)
    halfVector = cosineSampleHemisphere(r1, r2); // Diffuse
  else
    halfVector = ggxSampling(alpha, r1, r2); // Glossy

  // Transform the half vector to the hemisphere's tangent space
  halfVector = tangent * halfVector.x + binormal * halfVector.y + surfaceNormal * halfVector.z;

  // Compute the reflection direction from the sampled half vector and view direction
  float3 reflectVector = reflect(-viewDir, halfVector);

  // Early out: avoid internal reflection
  if (dot(surfaceNormal, reflectVector) < 0.0F)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Evaluate the refection coefficient with this new ray direction
  BsdfEvaluateData evalData;
  evalData.ior1 = data.ior1;
  evalData.ior2 = data.ior2;
  evalData.k1 = viewDir;
  evalData.k2 = reflectVector;
  bsdfEvaluate(evalData, mat);

  // Return values
  data.bsdf_over_pdf = (evalData.bsdf_diffuse + evalData.bsdf_glossy) / evalData.pdf;
  data.pdf = evalData.pdf;
  data.event_type = BSDF_EVENT_GLOSSY_REFLECTION;
  data.k2 = reflectVector;

  // Avoid internal reflection
  if (data.pdf <= 0.0)
    data.event_type = BSDF_EVENT_ABSORB;

  return;
}


#endif
