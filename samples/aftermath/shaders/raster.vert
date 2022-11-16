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
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "device_host.h"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(location = 0) out vec3 outFragColor;
layout(location = 1) out vec2 outUv;
layout(location = 2) out float outAlpha;

//#define ENDLESSLOOP_VS 1
layout(constant_id = 0) const int CRASH_TEST = 0;

layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

void main()
{
  gl_Position  = frameInfo.mpv * vec4(inPosition, 1.0);
  outFragColor = 0.5 + 0.5 * inNormal;
  outUv        = inUv;

  outAlpha = 1.0;
  if(CRASH_TEST == 1)
  {
    while(outAlpha > 0.0)
    {
      outAlpha -= inNormal.x;
      outAlpha = abs(outAlpha);
    }
    outAlpha *= 0.1 * outAlpha;
  }
}
