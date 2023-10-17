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

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable


#include "device_host.h"

layout(location = 0) in vec3 inFragColor;
layout(location = 1) in vec2 inUv;
layout(location = 2) in float alpha;

layout(location = 0) out vec4 outColor;


layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};
layout(set = 0, binding = 1) buffer Testing_
{
  float values[];
};
layout(set = 0, binding = 2) uniform sampler2D inTexture[];

struct BadStruct
{
  float bad_value;
  mat4  bad_mat;
};
layout(buffer_reference, scalar) buffer BadStruct_
{
  BadStruct s[];
};


layout(constant_id = 0) const int CRASH_TEST = 0;

vec3 IntegerToColor(uint val)
{
  const vec3 freq = vec3(1.33333f, 2.33333f, 3.33333f);
  return vec3(sin(freq * val) * .5 + .5);
}


void main()
{
  float iTime = frameInfo.time[0];

  // Ripple color
  vec3  col = vec3(inUv, 0.5 + 0.5 * sin(iTime * 6.5));
  float r   = length(vec2(0.5, 0.5) - inUv);
  float z   = 1.0 + 0.5 * sin((r + iTime * 0.05) / 0.005);
  col *= z;


  if(CRASH_TEST == 2)
  {
    // Entering an infinite loop
    col = min(col, vec3(1.0));
    while(col.x <= 10.0)
    {
      col.x *= sin(col.x);
    }
  }

  if(CRASH_TEST == 3)
  {
    // This is not good, but might not crash
    values[frameInfo.badOffset] = 0.5 + 0.5 * abs(sin(gl_FragCoord.y + iTime) + sin(gl_FragCoord.x + iTime * 0.5));
    col *= values[frameInfo.badOffset];

    // Bad access
    BadStruct_ tbuf = BadStruct_(frameInfo.bufferAddr);
    BadStruct  buff = tbuf.s[frameInfo.badOffset + 100000000];  // <------ Make it crash !!!
    col += buff.bad_value;
  }

  if(CRASH_TEST == 4)
  {
    col += texture(inTexture[frameInfo.badOffset + 100000000], inUv).xyz;
    col *= IntegerToColor(frameInfo.badOffset);
  }

  outColor = vec4(col, alpha);
}