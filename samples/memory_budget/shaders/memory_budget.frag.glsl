/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#version 450


#extension GL_GOOGLE_include_directive : enable

#include "shaderio.h"
#include "nvshaders/functions.h.slang"
#include "nvshaders/simple_shading.h.slang"


layout(location = 0) in vec3 inFragPos;
layout(location = 1) in vec3 inFragNrm;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};
layout(push_constant) uniform PushConstant_
{
  PushConstant pushConst;
};



void main()
{
  vec3 V = normalize(frameInfo.camPos - inFragPos);  // vector that goes from the hit position towards the origin of the ray
  vec3 color = simpleShading(V, V, inFragNrm, pushConst.color.xyz, 16.0);
  color      = pow(color, vec3(1.0 / 2.2));  // Gamma correction

  // Darker in the center
  float nb_subdiv = pow(3.0F, MENGER_SUBDIV);
  vec3  cx        = floor(abs((inFragPos * nb_subdiv) + vec3(0.001))) / nb_subdiv * 2.0;
  float factor    = pow(max(cx.x, max(cx.y, cx.z)), 1.5);
  color *= factor;
  outColor = vec4(color, 1);
}