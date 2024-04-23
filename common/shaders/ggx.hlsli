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
#include "pbr_mat_struct.hlsli"


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

#define BSDF_USE_MATERIAL_IOR (-1.0F)

struct BsdfEvaluateData
{
  float3 k1; // [in] Toward the incoming ray
  float3 k2; // [in] Toward the sampled light
  float3 bsdf_diffuse; // [out] Diffuse contribution
  float3 bsdf_glossy; // [out] Specular contribution
  float pdf; // [out] PDF
};

struct BsdfSampleData
{
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
  float a = 1.0F - VdotH;
  float a2 = a * a;
  return f0 + (f90 - f0) * a2 * a2 * a;
}
float fresnelSchlick(float f0, float f90, float VdotH)
{
  float a = 1.0F - VdotH;
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
  if(ggx > 0.0F)
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
  float denom = NdotHSqr * (alphaSqr - 1.0F) + 1.0F;

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
  return (1.0F - (specularWeight * fresnelSchlick(f0, f90, VdotH))) * (diffuseColor / M_PI);
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

  float phi = 2.0F * M_PI * r1;
  float cosTheta = sqrt((1.0F - r2) / (1.0F + (alphaSqr - 1.0F) * r2));
  float sinTheta = sqrt(1.0F - cosTheta * cosTheta);

  return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

// Return false if it produce a total internal reflection
bool refract(float3 incident, float3 normal, float eta, out float3 transmitted)
{
  float cosTheta = dot(incident, normal);
  float k = 1.0F - eta * eta * (1.0F - cosTheta * cosTheta);
  transmitted = float3(0.0F, 0.0F, 0.0F);
  if(k < 0.0F)
  {
    // Total internal reflection
    return false;
  }
  else
  {
    transmitted = eta * incident - (eta * cosTheta + sqrt(k)) * normal;
    return true;
  }
}


float3 absorptionCoefficient(in PbrMaterial mat)
{
  float tmp1 = mat.attenuationDistance;
  return tmp1 <= 0.0F ? float3(0.0F, 0.0F, 0.0F) :
                       -float3(log(mat.attenuationColor.x), log(mat.attenuationColor.y), log(mat.attenuationColor.z)) / tmp1.xxx;
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
  float3 f90 = float3(1.0F, 1.0F, 1.0F);

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
  float clearcoat = mat.clearcoatFactor;
  float transmission = mat.transmissionFactor;
  float3 f0 = mat.f0;
  float3 f90 = mat.f90;
  float transmissionRatio = (1.0F - metallic) * transmission;

  // Specular roughness
  float alpha = roughness * roughness;

  // Compute half vector
  float3 halfVector = normalize(viewDir + lightDir);

  // Compute various "angles" between vectors
  float NdotV = clampedDot(surfaceNormal, viewDir);
  float NdotL = clampedDot(surfaceNormal, lightDir);
  float VdotH = clampedDot(viewDir, halfVector);
  float NdotH = clampedDot(surfaceNormal, halfVector);
  float LdotH = dot(lightDir, halfVector);

  // Contribution
  float3 f_diffuse = brdfLambertian(albedo, metallic);
  float3 f_specular = brdfSpecularGGX(f0, f90, alpha, VdotH, NdotL, NdotV, NdotH);

  // Calculate PDF (probability density function)
  float diffuseRatio = 0.5F * clamp(1.0F - metallic - transmission, 0, 1);
  float diffusePDF = (NdotL * M_1_OVER_PI);
  float specularPDF = distributionGGX(NdotH, alpha) * NdotH / (4.0F * LdotH);

  float brdfPdf = lerp(specularPDF, diffusePDF, diffuseRatio);
  
   // Calculate transmitted direction using Snell's law
  if(transmission > 0.0F)
  {
    float eta = mat.eta; // Refractive index
    float3 refractedDir;
    bool totalInternalRefraction = refract(lightDir, surfaceNormal, eta, refractedDir);

    if(!totalInternalRefraction)
    {
      // Adjust diffuse and specular components for transmission
      f_diffuse = lerp(f_diffuse, float3(0.0F, 0.0F, 0.0F), transmissionRatio);
      f_specular = lerp(f_specular, float3(0.0F, 0.0F, 0.0F), transmissionRatio);

      // Calculate transmission PDF
      float transmissionPDF = abs(dot(refractedDir, surfaceNormal));

      // Mix PDFs
      float pdf = lerp(brdfPdf, transmissionPDF, transmissionRatio);

      // Results
      data.bsdf_diffuse = f_diffuse * NdotL;
      data.bsdf_glossy = f_specular * NdotL;
      data.pdf = pdf;
      return;
    }
  }

  // If transmission didn't occur or if total internal reflection happened
  // Evaluate BRDF for non-transmissive case
  data.bsdf_diffuse = f_diffuse * NdotL;
  data.bsdf_glossy = f_specular * NdotL;
  data.pdf = brdfPdf;
}

void bsdfSample(inout BsdfSampleData data, in PbrMaterial mat)
{
  // Initialization
  float3 surfaceNormal = mat.normal;
  float3 viewDir = data.k1;
  float3 albedo = mat.albedo.rgb;
  float metallic = mat.metallic;
  float roughness = mat.roughness;
  float clearcoat = mat.clearcoatFactor;
  float transmission = mat.transmissionFactor;
  float f0 = luminance(mat.f0);
  float f90 = luminance(mat.f90);
  float eta = mat.eta;

  // Random numbers for importance sampling
  float r1 = data.xi.x;
  float r2 = data.xi.y;
  float r3 = data.xi.z;

  // Create tangent space
  float3 tangent, bitangent;
  orthonormalBasis(surfaceNormal, tangent, bitangent);

  // Find Half vector for diffuse or glossy reflection
  float diffuseRatio = 0.5F * clamp(1.0F - metallic - transmission, 0 , 1);
  float transmissionRatio = (1.0F - metallic) * transmission;
  float3 sampleDirection = float3(0.0F, 0.0F, 0.0F);


  if(r3 < diffuseRatio)
  {
    sampleDirection = cosineSampleHemisphere(r1, r2); // Diffuse
    sampleDirection = tangent * sampleDirection.x + bitangent * sampleDirection.y + surfaceNormal * sampleDirection.z;
    data.event_type = BSDF_EVENT_DIFFUSE;
  }
  else
  {
    // Specular roughness
    float alpha = roughness * roughness;
    float3 halfVector = ggxSampling(alpha, r1, r2); // Glossy
    
    // Transform the half vector to the hemisphere's tangent space
    halfVector = tangent * halfVector.x + bitangent * halfVector.y + surfaceNormal * halfVector.z;
    
    // Compute the reflection direction from the sampled half vector and view direction
    sampleDirection = reflect(-viewDir, halfVector);
    data.event_type = BSDF_EVENT_SPECULAR;

    // If surface is rough, update surfaceNormal to follow the microfacet distribution for the rest of the calculations
    if(roughness > 0.0F)
    {
      surfaceNormal = halfVector;
    }
  }

  // Calculate if the ray goes through
  if(r3 < transmissionRatio)
  {
    // Calculate transmission direction using Snell's law
    float3 refractedDir;
    bool refracted = refract(-viewDir, surfaceNormal, eta, refractedDir);
    if(eta == 1.f && roughness > 0.0F)
    {
      refractedDir = -sampleDirection;
    }
    // Fresnel term
    float VdotH = dot(viewDir, surfaceNormal);
    float reflectance = fresnelSchlick(f0, f90, VdotH);

    if(!refracted || r3 < reflectance)
    {
      // Total internal reflection or reflection based on Fresnel term
      sampleDirection = reflect(-viewDir, surfaceNormal); // Reflective direction
      data.event_type = BSDF_EVENT_SPECULAR;
    }
    else
    {
      // Transmission
      sampleDirection = refractedDir;
      data.event_type = BSDF_EVENT_TRANSMISSION;
    }

    // Attenuate albedo for transmission
    albedo *= transmission;

    // Result
    data.bsdf_over_pdf = albedo;
    data.pdf = abs(dot(surfaceNormal, sampleDirection)); //transmissionRatio;
    data.k2 = sampleDirection;
    return;
  }

  // Evaluate the reflection coefficient with the new ray direction
  BsdfEvaluateData evalData;
  evalData.k1 = viewDir;
  evalData.k2 = sampleDirection;
  bsdfEvaluate(evalData, mat);

  // Return values
  data.bsdf_over_pdf = (evalData.bsdf_diffuse + evalData.bsdf_glossy) / evalData.pdf;
  data.pdf = evalData.pdf;
  if(all((evalData.bsdf_diffuse + evalData.bsdf_glossy) == float3(0.0F, 0.0F, 0.0F)))
    data.event_type = BSDF_EVENT_ABSORB;
  data.k2 = sampleDirection;

  // Avoid internal reflection
  if(data.pdf <= 0.00001 || any(isnan(data.bsdf_over_pdf)))
    data.event_type = BSDF_EVENT_ABSORB;

  return;
}


#endif
