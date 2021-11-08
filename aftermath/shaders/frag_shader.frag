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
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "host_device.h"

#define M_PI 3.141592
// clang-format off
// Incoming 
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal;
layout(location = 3) in vec3 viewDir;
layout(location = 4) in vec3 worldPos;
// Outgoing
layout(location = 0) out vec4 outColor;
// Buffers

layout(buffer_reference, scalar) buffer  GltfMaterial { ShadingMaterial m[]; };

layout(set = 0, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; } ;
layout(set = 0, binding = eTextures) uniform sampler2D[] textureSamplers;
// clang-format on

layout(push_constant) uniform RasterPushConstant_
{
  RasterPushConstant pc;
};


void main()
{
  // Material of the object
  GltfMaterial    gltfMat = GltfMaterial(sceneDesc.materialAddress);
  ShadingMaterial mat     = gltfMat.m[pc.materialId + 100000000];

  vec3 N = normalize(fragNormal);

  Light light = frameInfo.light[0];

  // Vector toward light
  vec3  L              = light.position;
  float lightIntensity = light.intensity;
  if(light.type == 0 /*point*/)
  {
    L -= worldPos;
    float d = length(L);
    lightIntensity /= (d * d);
  }
  L = normalize(L);

  // Diffuse
  float dotNL   = max(dot(N, L), 0.0);
  vec3  diffuse = mat.pbrBaseColorFactor.xyz * dotNL;
  if(mat.pbrBaseColorTexture > -1)
  {
    uint txtId      = mat.pbrBaseColorTexture;
    vec3 diffuseTxt = texture(textureSamplers[nonuniformEXT(txtId)], fragTexCoord).xyz;
    diffuse *= diffuseTxt;
  }

  // Result
  vec3 result = (mat.pbrBaseColorFactor.xyz * frameInfo.clearColor.xyz);  // ambient
  result += lightIntensity * light.color * (diffuse / M_PI);              // diffuse
  result += mat.emissiveFactor;                                           // emissive

  outColor = vec4(result, 1);
}
