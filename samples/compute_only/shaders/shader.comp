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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require

#include "device_host.h"
#include "nvvkhl/shaders/dh_comp.h"  // WORKGROUP_SIZE

layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE, local_size_z = 1) in;
layout(set = 0, binding = 0) writeonly uniform image2D outImage;
layout(push_constant) uniform PushConstant_
{
  PushConstant pushConst;
};

//https://iquilezles.org/articles/palettes/
vec3 palette(float t)
{
  vec3 a = vec3(0.5, 0.5, 0.5);
  vec3 b = vec3(0.5, 0.5, 0.5);
  vec3 c = vec3(1.0, 1.0, 0.5);
  vec3 d = vec3(0.0, 0.15, 0.2);

  return a + b * cos(6.28318f * (c * t + d));
}


void main()
{
  vec2 fragCoord   = gl_GlobalInvocationID.xy;
  vec2 iResolution = imageSize(outImage);
  if(fragCoord.x >= iResolution.x || fragCoord.y >= iResolution.y)
    return;
  float iTime = pushConst.time;

  vec2 uv         = (fragCoord * 2.0f - iResolution) / iResolution.y;
  vec2 uv0        = uv;
  vec3 finalColor = vec3(0.0);

  for(float i = 0.0; i < pushConst.iter; i++)
  {
    uv = fract(uv * pushConst.zoom) - 0.5f;

    float d = length(uv) * exp(-length(uv0));

    vec3 col = palette(length(uv0) + i + iTime * .6);

    d = sin(d * 8 + iTime) / 2;
    d = abs(d);
    d = pow(0.01 / d, 1.2);

    finalColor += col * d;
  }

  imageStore(outImage, ivec2(fragCoord), vec4(finalColor, 1.0F));
}
