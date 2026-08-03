/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_descriptor_heap : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "shaderio.h"

layout(location = 0) in vec3 inFragPos;
layout(location = 1) in vec3 inFragNrm;

layout(location = 0) out vec4 outColor;

// Bindless: descriptor-heap storage buffer accessed by local buffer index (heap slot 0 = FrameInfo).
layout(descriptor_heap, scalar) readonly buffer FrameInfo_
{
  FrameInfo info;
}
heapFrameInfo[];
layout(push_constant) uniform PushConstant_
{
  PushConstant pushC;
};

vec3 simpleShading(in vec3 toEye, in vec3 normal)
{
  vec3  color    = vec3(0.8);
  vec3  wUpDir   = vec3(0, 1, 0);
  vec3  lightDir = normalize(toEye);
  vec3  eyeDir   = normalize(toEye);
  vec3  reflDir  = normalize(-reflect(lightDir, normal));
  float lt       = abs(dot(normal, lightDir)) + pow(max(0, dot(reflDir, eyeDir)), 16.0);
  color          = color * (lt);
  color += mix(vec3(0.1, 0.1, 0.4), vec3(0.8, 0.6, 0.2), dot(normal, wUpDir.xyz) * 0.5 + 0.5) * 0.2;
  return color;
}

void main()
{
  FrameInfo frameInfo = heapFrameInfo[kHeapBufFrameInfo].info;

  vec3 toEye = frameInfo.camPos - inFragPos;
  vec3 color = simpleShading(toEye, inFragNrm) * pushC.color.xyz;
  outColor   = vec4(color, pushC.color.w);
}
