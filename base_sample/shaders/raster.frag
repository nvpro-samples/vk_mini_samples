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
#include "raycommon.glsl"
#include "sampling.glsl"


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
// clang-format on


#include "pbr_gltf.glsl"

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
  vec3 result = (matEval.albedo.xyz * frameInfo.clearColor.xyz);  // ambient
  result += matEval.emissive;                                     // emissive

  // All lights
  for(int i = 0; i < NB_LIGHTS; i++)
  {
    Light light = frameInfo.light[i];
    vec3  lightDir;
    vec3  lightContrib = lightContribution(light, hit.pos, hit.nrm, lightDir);
    float dotNL        = dot(lightDir, hit.nrm);

    float pdf      = 0;
    vec3  brdf     = pbrEval(matEval, toEye, lightDir, pdf);
    vec3  radiance = brdf * dotNL * lightContrib;
    result += radiance;
  }

  outColor = vec4(result, 1);
}
