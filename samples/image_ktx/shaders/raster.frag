/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450


#extension GL_GOOGLE_include_directive : enable

#include "device_host.h"


layout(location = 0) in vec3 inFragPos;
layout(location = 1) in vec3 inFragNrm;
layout(location = 2) in vec2 inFragUv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = BKtxFrameInfo) uniform FrameInfo_
{
  FrameInfo frameInfo;
};
layout(set = 0, binding = BKtxTex) uniform sampler2D inTexture;

layout(push_constant) uniform PushConstant_
{
  PushConstant pushC;
};

//layout(set = 0, binding = 0) uniform sampler2D inTexture;

vec3 simpleShading(in vec3 toEye, in vec3 normal)
{
  vec3 color    = vec3(0.8);
  vec3 wUpDir   = vec3(0, 1, 0);
  vec3 lightDir = normalize(toEye);
  vec3 eyeDir   = normalize(toEye);
  //vec3 normal   = normalize(-cross(dFdx(i_pos), dFdy(i_pos))); // Geometric normals
  vec3 reflDir = normalize(-reflect(lightDir, normal));
  // Diffuse + Specular
  float lt = abs(dot(normal, lightDir)) + pow(max(0, dot(reflDir, eyeDir)), 16.0);
  color    = color * (lt);
  // Ambient term (sky effect)
  color += mix(vec3(0.1, 0.1, 0.4), vec3(0.8, 0.6, 0.2), dot(normal, wUpDir.xyz) * 0.5 + 0.5) * 0.2;
  // Gamma correction
  //color = pow(color, vec3(2.2));
  return color;
}


void main()
{
  vec3 toEye = frameInfo.camPos - inFragPos;
  vec3 color = simpleShading(toEye, inFragNrm) * pushC.color.xyz;
  color *= texture(inTexture, inFragUv).xyz;
  outColor = vec4(color, pushC.color.w);
}