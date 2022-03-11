/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "host_device.h"
#include "common/shaders/ray_common.glsl"
#include "common/shaders/sampling.glsl"


// clang-format off
// Incoming 
layout(location = 1) in vec2 i_texCoord;
layout(location = 2) in vec3 i_normal;
layout(location = 3) in vec3 i_viewDir;
layout(location = 4) in vec3 i_pos;
layout(location = 5) in vec4 i_tangent;

// Outgoing
layout(location = 0) out vec4 outColor;

// Buffers
layout(buffer_reference, scalar) buffer  GltfMaterial { GltfShadeMaterial m[]; };
layout(set = 0, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; } ;
layout(set = 0, binding = eTextures) uniform sampler2D[] texturesMap;

layout(set = 1, binding = 0) uniform sampler2D   u_GGXLUT; // lookup table
layout(set = 1, binding = 1) uniform samplerCube u_LambertianEnvSampler; // 
layout(set = 1, binding = 2) uniform samplerCube u_GGXEnvSampler;  //
// clang-format on


#include "common/shaders/pbr_gltf.glsl"

layout(push_constant) uniform RasterPushConstant_
{
  RasterPushConstant pc;
};

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3 pos;
  vec3 nrm;
  vec2 uv;
  vec3 tangent;
  vec3 bitangent;
};

vec3 getDiffuseLight(vec3 n)
{
  vec3 dir = rotate(n, vec3(0, 1, 0), -frameInfo.envRotation);
  return texture(u_LambertianEnvSampler, dir).rgb * frameInfo.clearColor.rgb;
}

vec4 getSpecularSample(vec3 reflection, float lod)
{
  vec3 dir = rotate(reflection, vec3(0, 1, 0), -frameInfo.envRotation);
  return textureLod(u_GGXEnvSampler, dir, lod) * frameInfo.clearColor;
}


// specularWeight is introduced with KHR_materials_specular
vec3 getIBLRadianceLambertian(vec3 n, vec3 v, float roughness, vec3 diffuseColor, vec3 F0, float specularWeight)
{
  float NdotV           = clampedDot(n, v);
  vec2  brdfSamplePoint = clamp(vec2(NdotV, roughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
  vec2  f_ab            = texture(u_GGXLUT, brdfSamplePoint).rg;

  vec3 irradiance = getDiffuseLight(n);

  // see https://bruop.github.io/ibl/#single_scattering_results at Single Scattering Results
  // Roughness dependent fresnel, from Fdez-Aguera

  vec3 Fr     = max(vec3(1.0 - roughness), F0) - F0;
  vec3 k_S    = F0 + Fr * pow(1.0 - NdotV, 5.0);
  vec3 FssEss = specularWeight * k_S * f_ab.x + f_ab.y;  // <--- GGX / specular light contribution (scale it down if the specularWeight is low)

  // Multiple scattering, from Fdez-Aguera
  float Ems    = (1.0 - (f_ab.x + f_ab.y));
  vec3  F_avg  = specularWeight * (F0 + (1.0 - F0) / 21.0);
  vec3  FmsEms = Ems * FssEss * F_avg / (1.0 - F_avg * Ems);
  vec3  k_D    = diffuseColor * (1.0 - FssEss + FmsEms);  // we use +FmsEms as indicated by the formula in the blog post (might be a typo in the implementation)

  return (FmsEms + k_D) * irradiance;
}


vec3 getIBLRadianceGGX(vec3 n, vec3 v, float roughness, vec3 F0, float specularWeight)
{
  int u_MipCount = textureQueryLevels(u_GGXEnvSampler);

  float NdotV      = clampedDot(n, v);
  float lod        = roughness * float(u_MipCount - 1);
  vec3  reflection = normalize(reflect(-v, n));

  vec2 brdfSamplePoint = clamp(vec2(NdotV, roughness), vec2(0.0, 0.0), vec2(1.0, 1.0));
  vec2 f_ab            = texture(u_GGXLUT, brdfSamplePoint).rg;
  vec4 specularSample  = getSpecularSample(reflection, lod);

  vec3 specularLight = specularSample.rgb;

  // see https://bruop.github.io/ibl/#single_scattering_results at Single Scattering Results
  // Roughness dependent fresnel, from Fdez-Aguera
  vec3 Fr     = max(vec3(1.0 - roughness), F0) - F0;
  vec3 k_S    = F0 + Fr * pow(1.0 - NdotV, 5.0);
  vec3 FssEss = k_S * f_ab.x + f_ab.y;

  return specularWeight * specularLight * FssEss;
}


void main()
{
  // Material of the object
  GltfMaterial      gltfMat = GltfMaterial(sceneDesc.materialAddress);
  GltfShadeMaterial mat     = gltfMat.m[pc.materialId];

  HitState hit;
  hit.pos       = i_pos;
  hit.nrm       = normalize(i_normal);
  hit.uv        = i_texCoord;
  hit.tangent   = normalize(i_tangent.xyz);
  hit.bitangent = cross(hit.nrm, hit.tangent) * i_tangent.w;

  MaterialEval matEval = evaluateMaterial(mat, hit.nrm, hit.tangent, hit.bitangent, hit.uv);

  vec3 toEye = normalize(-i_viewDir);

  // Result
  vec3 result = vec3(0);  //(matEval.albedo.xyz * frameInfo.clearColor.xyz);  // ambient

  float ambientFactor = 0.3;
  result += getIBLRadianceGGX(matEval.normal, toEye, matEval.roughness, matEval.f0, 1.0) * ambientFactor;
  result += getIBLRadianceLambertian(matEval.normal, toEye, matEval.roughness, matEval.albedo.rgb.rgb, matEval.f0, 1.0) * ambientFactor;

  result += matEval.emissive;  // emissive

  // All lights
  for(int i = 0; i < NB_LIGHTS; i++)
  {
    Light light = frameInfo.light[i];
    vec3  lightDir;
    vec3  lightContrib = lightContribution(light, hit.pos, matEval.normal, lightDir);

    float pdf      = 0;
    vec3  brdf     = pbrEval(matEval, toEye, lightDir, pdf);
    vec3  radiance = brdf * lightContrib;
    result += radiance;
  }


  outColor = vec4(result, 1);
}
