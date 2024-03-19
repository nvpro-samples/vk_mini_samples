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
#version 450


#extension GL_GOOGLE_include_directive : enable

#include "device_host.h"


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

vec3 simpleShading(vec3 viewDir, vec3 lightDir, vec3 normal, vec3 color, float expo)
{
  // Diffuse + Specular
  vec3 reflDir = normalize(-reflect(lightDir, normal));
  float lt = clamp(dot(normal, lightDir), 0, 1) + pow(max(0., dot(reflDir, viewDir)), expo);
  color *= lt;

  // Slight ambient term (sky effect)
  vec3 skyUpDir = vec3(0, 1, 0);
  vec3 groundColor = vec3(0.1, 0.1, 0.4);
  vec3 skyColor = vec3(0.8, 0.6, 0.2);
  color += mix(skyColor, groundColor, dot(normal, skyUpDir.xyz) * 0.5 + 0.5) * 0.2;

  return color;
}


void main()
{
  vec3 V = normalize(frameInfo.camPos - inFragPos); // vector that goes from the hit position towards the origin of the ray  
  vec3 color = simpleShading(V, V, inFragNrm, pushConst.color.xyz, 16.0);
  color = pow(color, vec3(1.0/2.2)); // Gamma correction

  // Darker in the center
  float nb_subdiv = pow(3.0F, MENGER_SUBDIV);
  vec3 cx = floor(abs((inFragPos * nb_subdiv) + vec3(0.001))) / nb_subdiv * 2.0;
  float factor = pow(max(cx.x, max(cx.y, cx.z)), 1.5);
  color *= factor;
  outColor = vec4(color, 1);
}