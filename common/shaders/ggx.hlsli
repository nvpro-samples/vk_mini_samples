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

//-----------------------------------------------------------------------------------------



struct EvalData
{
  float pdf;
  float3 bsdf;
};

void evalDiffuse(in BsdfEvaluateData data, in PbrMaterial mat, out EvalData eval)
{
  // Diffuse reflection
  float NdotL = clampedDot(mat.normal, data.k2);
  eval.bsdf = brdfLambertian(mat.albedo.rgb, mat.metallic) * NdotL;
  eval.pdf = M_1_OVER_PI;
}

void evalSpecular(in BsdfEvaluateData data, in PbrMaterial mat, out EvalData eval)
{
  // Specular reflection
  float3 H = normalize(data.k1 + data.k2);

  float alphaRoughness = mat.roughness * mat.roughness;
  float NdotV = clampedDot(mat.normal, data.k1);
  float NdotL = clampedDot(mat.normal, data.k2);
  float VdotH = clampedDot(data.k1, H);
  float NdotH = clampedDot(mat.normal, H);
  float LdotH = clampedDot(data.k2, H);

  float3 f_specular = brdfSpecularGGX(mat.f0, mat.f90, alphaRoughness, VdotH, NdotL, NdotV, NdotH);
  eval.bsdf = f_specular * NdotL;
  eval.pdf = distributionGGX(NdotH, alphaRoughness) * NdotH / (4.0F * LdotH);
}

void evalTransmission(in BsdfEvaluateData data, in PbrMaterial mat, out EvalData eval)
{
  eval.pdf = 0;
  eval.bsdf = float3(0.0F, 0.0F, 0.0F);

  if(mat.transmissionFactor <= 0.0F)
    return;

  float3 refractedDir;
  bool totalInternalRefraction = refract(data.k2, mat.normal, mat.eta, refractedDir);

  if(!totalInternalRefraction)
  {
    //eval.bsdf = mat.albedo.rgb * mat.transmissionFactor;
    eval.pdf = abs(dot(refractedDir, mat.normal));
  }
}


/** @DOC_START
# Function bsdfEvaluate
>  Evaluate the BSDF for the given material.
@DOC_END */
void bsdfEvaluate(inout BsdfEvaluateData data, in PbrMaterial mat)
{
  // Initialization
  float diffuseRatio = 0.5F * (1.0F - mat.metallic);
  float specularRatio = 1.0F - diffuseRatio;
  float transmissionRatio = (1.0F - mat.metallic) * mat.transmissionFactor;

  // Contribution
  EvalData f_diffuse;
  EvalData f_specular;
  EvalData f_transmission;

  evalDiffuse(data, mat, f_diffuse);
  evalSpecular(data, mat, f_specular);
  evalTransmission(data, mat, f_transmission);

  // Combine the results
  float brdfPdf = 0;
  brdfPdf += f_diffuse.pdf * diffuseRatio;
  brdfPdf += f_specular.pdf * specularRatio;
  brdfPdf = lerp(brdfPdf, f_transmission.pdf, transmissionRatio);

  float3 bsdfDiffuse = lerp(f_diffuse.bsdf, f_transmission.bsdf, transmissionRatio);
  float3 bsdfGlossy = lerp(f_specular.bsdf, f_transmission.bsdf, transmissionRatio);

  // Return results
  data.bsdf_diffuse = bsdfDiffuse;
  data.bsdf_glossy = bsdfGlossy;
  data.pdf = brdfPdf;
}

//-------------------------------------------------------------------------------------------------

float3 sampleDiffuse(inout BsdfSampleData data, in PbrMaterial mat)
{
  float3 surfaceNormal = mat.normal;
  float3 tangent, bitangent;
  orthonormalBasis(surfaceNormal, tangent, bitangent);
  float r1 = data.xi.x;
  float r2 = data.xi.y;
  float3 sampleDirection = cosineSampleHemisphere(r1, r2); // Diffuse
  sampleDirection = tangent * sampleDirection.x + bitangent * sampleDirection.y + surfaceNormal * sampleDirection.z;
  data.event_type = BSDF_EVENT_DIFFUSE;
  return sampleDirection;
}

float3 sampleSpecular(inout BsdfSampleData data, in PbrMaterial mat)
{
  float3 surfaceNormal = mat.normal;
  float3 tangent, bitangent;
  orthonormalBasis(surfaceNormal, tangent, bitangent);
  float alphaRoughness = mat.roughness * mat.roughness;
  float r1 = data.xi.x;
  float r2 = data.xi.y;
  float3 halfVector = ggxSampling(alphaRoughness, r1, r2); // Glossy
  halfVector = tangent * halfVector.x + bitangent * halfVector.y + surfaceNormal * halfVector.z;
  float3 sampleDirection = reflect(-data.k1, halfVector);
  data.event_type = BSDF_EVENT_SPECULAR;

  return sampleDirection;
}

float3 sampleThinTransmission(in BsdfSampleData data, in PbrMaterial mat)
{
  float3 incomingDir = data.k1;
  float r1 = data.xi.x;
  float r2 = data.xi.y;
  float alphaRoughness = mat.roughness * mat.roughness;
  float3 halfVector = ggxSampling(alphaRoughness, r1, r2);
  float3 tangent, bitangent;
  orthonormalBasis(incomingDir, tangent, bitangent);
  float3 transformedHalfVector = tangent * halfVector.x + bitangent * halfVector.y + incomingDir * halfVector.z;
  float3 refractedDir = -transformedHalfVector;

  return refractedDir;
}

float3 sampleSolidTransmission(inout BsdfSampleData data, in PbrMaterial mat, out bool refracted)
{
  float3 surfaceNormal = mat.normal;
  if(mat.roughness > 0.0F)
  {
    float3 tangent, bitangent;
    orthonormalBasis(surfaceNormal, tangent, bitangent);
    float alphaRoughness = mat.roughness * mat.roughness;
    float r1 = data.xi.x;
    float r2 = data.xi.y;
    float3 halfVector = ggxSampling(alphaRoughness, r1, r2); // Glossy
    halfVector = tangent * halfVector.x + bitangent * halfVector.y + surfaceNormal * halfVector.z;
    surfaceNormal = halfVector;
  }

  float3 refractedDir;
  refracted = refract(-data.k1, surfaceNormal, mat.eta, refractedDir);

  return refractedDir;
}

void sampleTransmission(inout BsdfSampleData data, in PbrMaterial mat)
{
  // Calculate transmission direction using Snell's law
  float3 refractedDir;
  float3 sampleDirection;
  bool refracted = true;
  float r4 = data.xi.w;

  // Thin film approximation
  if(mat.thicknessFactor == 0.0F && mat.roughness > 0.0F)
  {
    refractedDir = sampleThinTransmission(data, mat);
  }
  else
  {
    refractedDir = sampleSolidTransmission(data, mat, refracted);
  }

  // Fresnel term
  float VdotH = dot(data.k1, mat.normal);
  float3 reflectance = fresnelSchlick(mat.f0, mat.f90, VdotH);
  float3 surfaceNormal = mat.normal;

  if(!refracted || r4 < luminance(reflectance))
  {
    // Total internal reflection or reflection based on Fresnel term
    sampleDirection = reflect(-data.k1, surfaceNormal); // Reflective direction
    data.event_type = BSDF_EVENT_SPECULAR;
  }
  else
  {
    // Transmission
    sampleDirection = refractedDir;
    data.event_type = BSDF_EVENT_TRANSMISSION;
  }

  // Attenuate albedo for transmission
  float3 bsdf = mat.albedo.rgb; // * mat.transmissionFactor;

  // Result
  data.bsdf_over_pdf = bsdf;
  data.pdf = abs(dot(surfaceNormal, sampleDirection)); //transmissionRatio;
  data.k2 = sampleDirection;
}


/** @DOC_START
# Function bsdfSample
>  Sample the BSDF for the given material
@DOC_END */
void bsdfSample(inout BsdfSampleData data, in PbrMaterial mat)
{
  // Random numbers for importance sampling
  float r3 = data.xi.z;

  // Initialization
  float diffuseRatio = 0.5F * (1.0F - mat.metallic);
  float specularRatio = 1.0F - diffuseRatio;
  float transmissionRatio = (1.0F - mat.metallic) * mat.transmissionFactor;

  // Calculate if the ray goes through
  if(r3 < transmissionRatio)
  {
    sampleTransmission(data, mat);
    return;
  }

  // Choose between diffuse and glossy reflection
  float3 sampleDirection = float3(0.0F, 0.0F, 0.0F);
  if(r3 < diffuseRatio)
  {
    sampleDirection = sampleDiffuse(data, mat);
  }
  else
  {
    // Specular roughness
    sampleDirection = sampleSpecular(data, mat);
  }

  // Evaluate the reflection coefficient with the new ray direction
  BsdfEvaluateData evalData;
  evalData.k1 = data.k1;
  evalData.k2 = sampleDirection;
  bsdfEvaluate(evalData, mat);

  // Return values
  data.pdf = evalData.pdf;
  data.bsdf_over_pdf = (evalData.bsdf_diffuse + evalData.bsdf_glossy) / data.pdf;
  data.k2 = sampleDirection;

  // Avoid internal reflection
  if(data.pdf <= 0.00001F || any(isnan(data.bsdf_over_pdf)))
    data.event_type = BSDF_EVENT_ABSORB;

  return;
}


#endif
