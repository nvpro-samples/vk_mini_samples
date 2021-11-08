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

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require


#include "host_device.h"

layout(location = 0) in vec2 i_uv;
layout(location = 0) out vec4 o_color;

layout(set = 0, binding = 0) uniform sampler2D g_image;

layout(push_constant) uniform shaderInformation
{
  Tonemapper tm;
};

// http://filmicworlds.com/blog/filmic-tonemapping-operators/
vec3 tonemapFilmic(vec3 x)
{
  vec3 X      = max(vec3(0.0), x - vec3(0.004));
  vec3 result = (X * (6.2 * X + 0.5)) / (X * (6.2 * X + 1.7) + 0.06);
  return result;
}


void main()
{
  vec4 R = texture(g_image, i_uv);
  // Exposure
  R *= tm.exposure;
  // Tonemap
  vec3 C = tonemapFilmic(R.rgb);
  //contrast
  C = clamp(mix(vec3(0.5), C, tm.contrast), 0, 1);
  // brighness
  C = pow(C, vec3(1.0 / tm.brightness));
  // saturation
  vec3 i = vec3(dot(C, vec3(0.299, 0.587, 0.114)));
  C      = mix(i, C, tm.saturation);
  // vignette
  vec2 uv = ((i_uv)-0.5) * 2.0;
  C *= 1.0 - dot(uv, uv) * tm.vignette;


  o_color = vec4(C, R.a);
}
