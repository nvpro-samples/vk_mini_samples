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

// #version 450 <--- This will slowdown the compilation

// Opengl 4.5
typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;

float fract(float value) { return frac(value); }
float mix(float a, float b, float t) { return lerp(a, b, t); }

[[vk::push_constant]]
cbuffer MyPush // Layout std140 :-(
{
  float3 iResolution;
  float iTime;
  float4 iMouse;    // 16
  float iTimeDelta; // 32
  int iFrame;       // 36
  int iFrameRate;   // 40
  int _pad0;
  float iChannelTime[1];
  // int _pad1;
  // int _pad2;
  // int _pad3;
  float3 iChannelResolution[1]; // 48
};

[[vk::binding(0)]]
Sampler2D iChannel0;

// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin {
  [[vk::location(0)]]
  float2 position : POSITION;
};

// Output of the vertex shader
struct VSout {
  float4 sv_position : SV_Position;
};

// Vertex  Shader
[shader("vertex")]
float4 vertexMain(float2 position: POSITION) : SV_Position {
  return float4(position, 0.0, 1.0);
}

#include "buffer_a.glsl"
#include "image.glsl"

// Fragment Shader
[shader("pixel")]
float4 fragmentMain(in float4 fragCoord: SV_Position) : SV_Target {
  // Inverting fragCoord for OpenGL
  vec2 invFragCoord = fragCoord.xy;
  invFragCoord.y = iResolution.y - fragCoord.y;

  float4 fragColor;
  mainImage(fragColor, invFragCoord.xy);

  return fragColor;
}

[shader("pixel")]
float4 fragmentBuffer(in float4 fragCoord: SV_Position) : SV_Target {
  // Inverting fragCoord for OpenGL
  vec2 invFragCoord = fragCoord.xy;
  invFragCoord.y = iResolution.y - fragCoord.y;

  float4 fragColor;
  mainBufferA(fragColor, invFragCoord.xy);

  return fragColor;
}
