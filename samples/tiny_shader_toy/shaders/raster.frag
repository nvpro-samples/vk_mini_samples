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

//---------------------------------------------
layout(location = 0) out vec4 fragColor;

// Shaders are supplied with static information per frame using the following variables
// See https://www.shadertoy.com/howto
layout(push_constant) uniform _InputUniforms
{
  vec3  iResolution;
  float iTime;
  vec4  iMouse;
  float iTimeDelta;
  int   iFrame;
  int   iFrameRate;
  int   _pad0;
  float iChannelTime[1];
  int   _pad1;
  int   _pad2;
  int   _pad3;
  vec3  iChannelResolution[1];
};

layout(set = 0, binding = 0) uniform sampler2D iChannel0;


// Shared accross all shaders
#include "common.glsl"

//---------------------------------------------
// On compilation, using the right shader code
#ifdef INCLUDE_IMAGE
#include "image.glsl"
#else
#include "buffer_a.glsl"
#endif
//---------------------------------------------

void main()
{
  // Initialization
  fragColor = vec4(0, 0, 0, 1);

  // Inverting fragCoord from OpenGL
  vec2 invFragCoord = gl_FragCoord.xy;
  invFragCoord.y    = iResolution.y - gl_FragCoord.y;

  // Calling the main function
  mainImage(fragColor, invFragCoord.xy);

  // Image is written to the attached G-Buffer
}