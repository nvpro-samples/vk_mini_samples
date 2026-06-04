/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"

// Shared push data and vertex path for the ConstantOffset and DirectAccess
// modes. Both render all cubes with a single instanced draw; each instance
// derives its world position and base texture index from SV_InstanceID.
struct PushBlock
{
  FrameInfo         frame;
  InstancedPushData instanced;
};
[[vk::push_constant]]
ConstantBuffer<PushBlock> push;

struct VSin
{
  [[vk::location(0)]]
  float3 position;
  [[vk::location(1)]]
  float3 normal;
};

struct VSout
{
  float2               uv : TEXCOORD0;
  float3               normal : NORMAL;
  nointerpolation int  faceIdx : TEXCOORD1;
  nointerpolation uint baseFaceTexIdx : TEXCOORD2;
  float4               sv_position : SV_Position;
};

struct PSout
{
  float4 color : SV_Target;
};

VSout instancedVertex(VSin input, int instanceIndex)
{
  // Use instanceIndex to compute this cube's 3D grid position (ix, iy, iz).
  int N  = int(push.instanced.gridSize);
  int ix = instanceIndex % N;
  int iy = (instanceIndex / N) % N;
  int iz = instanceIndex / (N * N);

  float  spacing = 1.1;
  float  off     = float(N - 1) * spacing * 0.5;
  float3 cubePos = float3(ix, iy, iz) * spacing - off;

  uint baseFaceTexIdx = uint(instanceIndex) * 6u;

  // Compute cycle length and loop
  float totalStagger = float(push.frame.numCubes - 1u) * animationCubeDelay;
  float fallInEnd    = totalStagger + animationFallDuration;
  float fallOutStart = fallInEnd + animationRestDuration;
  float cycleTime    = fallOutStart + totalStagger + animationFallDuration;
  float loopTime     = fmod(push.frame.time, cycleTime);

  // Per-cube fall-in and fall-out progress
  float tIn = clamp((loopTime - float(instanceIndex) * animationCubeDelay) / animationFallDuration, 0.0, 1.0);
  float tOut = clamp((loopTime - fallOutStart - float(instanceIndex) * animationCubeDelay) / animationFallDuration, 0.0, 1.0);

  // Invisible before fall-in or after fall-out: degenerate vertex
  bool visible = (tIn > 0.0) && (tOut < 1.0);

  // Y offset: fall in from above, then fall out below
  float h       = push.frame.dropHeight;
  float yOffset = h * (1.0 - tIn * tIn) - h * tOut * tOut;

  float3 worldPos = input.position + cubePos;
  worldPos.y += yOffset;

  VSout output;
  output.sv_position    = visible ? mul(mul(float4(worldPos, 1.0), push.frame.view), push.frame.proj) : float4(0.0);
  output.normal         = input.normal;
  output.baseFaceTexIdx = baseFaceTexIdx;

  // Face index from vertex normal
  float3 a       = abs(input.normal);
  int    axis    = (a.x > a.y && a.x > a.z) ? 0 : (a.y > a.z) ? 1 : 2;
  int    s       = (axis == 0 ? input.normal.x : (axis == 1 ? input.normal.y : input.normal.z)) > 0.0 ? 0 : 1;
  output.faceIdx = axis * 2 + s;

  // Compute UVs from the two non-dominant axes
  float3 p = input.position + 0.5;
  if(axis == 0)
    output.uv = s == 0 ? float2(1.0 - p.z, 1.0 - p.y) : float2(p.z, 1.0 - p.y);
  else if(axis == 1)
    output.uv = s == 0 ? float2(p.x, 1.0 - p.z) : float2(p.x, p.z);
  else
    output.uv = s == 0 ? float2(p.x, 1.0 - p.y) : float2(1.0 - p.x, 1.0 - p.y);

  return output;
}

float3 unpackColor(uint c)
{
  return float3(float(c & 0xFFu), float((c >> 8) & 0xFFu), float((c >> 16) & 0xFFu)) / 255.0;
}
