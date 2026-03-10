/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "shaderio.h"


// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin
{
  [[vk::location(0)]]
  float3 position : POSITION;
  [[vk::location(1)]]
  float3 normal : NORMAl;
  [[vk::location(2)]]
  float2 uv : TEXCOORD0;
};

// Output of the vertex shader, and input to the fragment shader.
struct PSin
{
  float3 color : COLOR0;
  float2 uv : TEXCOORD0;
};

// Output of the vertex shader
struct VSout
{
  PSin   stage;
  float4 sv_position : SV_Position;
};

// Output of the fragment shader
struct PSout
{
  float4 color : SV_Target;
};

// clang-format off
[[vk::constant_id(0)]] int CRASH_TEST;
[[vk::binding(0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(1)]] RWStructuredBuffer<float> values;
// clang-format on

// Vertex  Shader
[shader("vertex")]
VSout vertexMain(VSin input)
{
  VSout output;
  output.sv_position   = mul(float4(input.position, 1), frameInfo.mpv);
  output.stage.color   = 0.5 + 0.5 * input.normal;
  output.stage.uv      = input.uv;
  return output;
}


// Fragment Shader
[shader("pixel")]
PSout fragmentMain(PSin stage, float4 fragCoord: SV_Position)
{
  float iTime = frameInfo.time.x;

  // Ripple color
  float3 color = float3(stage.uv, 0.5 + 0.5 * sin(iTime * 6.5));
  float  r     = length(float2(0.5, 0.5) - stage.uv);
  float  z     = 1.0 + 0.5 * sin((r + iTime * 0.05) / 0.005);
  color *= z;

  // ---------------------------------------------------------------------------
  // TDR CRASH: Infinite loop using sin()
  // ---------------------------------------------------------------------------
  if(CRASH_TEST == 1)
  {
    // TDR: sin(x) for x in (0,1) returns a value in (0,1), so multiplying
    // color.x by sin(color.x) keeps color.x small and it never exceeds 10.
    // The GPU cannot complete this shader, triggering Windows TDR
    // (Timeout Detection and Recovery) after ~2 seconds -> VK_ERROR_DEVICE_LOST.
    color = min(color, float3(1.0, 1.0, 1.0));
    while(color.x <= 10.0)
    {
      color.x *= sin(color.x);
    }
  }

  // ---------------------------------------------------------------------------
  // TDR CRASH: Infinite loop with SSBO writes
  // ---------------------------------------------------------------------------
  if(CRASH_TEST == 2)
  {
    // TDR: v starts in (0,1) and v *= sin(v) keeps shrinking it toward 0.
    // Once v reaches 0.0 in float32, it stays there forever (0 * sin(0) = 0),
    // but "v <= 10.0" remains true. The SSBO write each iteration is a visible
    // side-effect that prevents the compiler from optimizing the loop away
    // -> guaranteed TDR -> VK_ERROR_DEVICE_LOST.
    float v = min(color.x, 0.99);
    while(v <= 10.0)
    {
      v *= sin(v);
      values[0] = v;
    }
    color *= values[0];
  }

  // ---------------------------------------------------------------------------
  // PAGE FAULT CRASH: Write through BDA to unmapped GPU virtual memory
  // ---------------------------------------------------------------------------
  if(CRASH_TEST == 3)
  {
    // PAGE FAULT: bufferAddr is set by the CPU to an unmapped GPU virtual
    // address -- either a valid buffer base + huge offset (overrun past the
    // buffer end), NULL (address 0), or the address of a previously destroyed
    // buffer (use-after-free).
    // Writing through this BDA pointer causes a GPU page fault that the
    // driver cannot recover from -> VK_ERROR_DEVICE_LOST.
    float* ptr = (float*)(frameInfo.bufferAddr);
    *ptr = 1.0;
    color += *ptr;
  }

  // ---------------------------------------------------------------------------
  // PAGE FAULT CRASH: Wild pointer spray -- each fragment writes to a
  // different pseudo-random address in the GPU virtual address space
  // ---------------------------------------------------------------------------
  if(CRASH_TEST == 4)
  {
    // WILD POINTER SPRAY: Each fragment hashes its screen coordinates to
    // produce a different address in the GPU's ~40-bit virtual address space.
    // With thousands of fragments writing to scattered addresses, we're
    // guaranteed to hit unmapped pages -> GPU page fault -> VK_ERROR_DEVICE_LOST.
    uint     h    = uint(fragCoord.x) * 2654435761u ^ uint(fragCoord.y) * 340573321u;
    uint64_t addr = uint64_t(h) << 8;  // spread across 40-bit VA range
    addr &= ~uint64_t(3);              // 4-byte alignment
    addr += 4096u;                     // skip null page region
    float* ptr = (float*)(addr);
    *ptr = 1.0;
    color += *ptr;
  }

  // ---------------------------------------------------------------------------
  // PAGE FAULT CRASH: Indirect use-after-free via BDA pointer chain
  // ---------------------------------------------------------------------------
  if(CRASH_TEST == 5)
  {
    // INDIRECT USE-AFTER-FREE: bufferAddr points to an "indirect" buffer that
    // holds the BDA of a second "target" buffer. The target buffer has been
    // destroyed on the CPU side. The shader follows the pointer chain:
    //   bufferAddr -> indirect buffer -> [stale target address] -> WRITE
    // The write through the dangling inner pointer causes a GPU page fault
    // -> VK_ERROR_DEVICE_LOST.
    uint64_t* indirect   = (uint64_t*)(frameInfo.bufferAddr);
    uint64_t  targetAddr = *indirect;
    float*    target     = (float*)(targetAddr);
    *target = 1.0;
    color += *target;
  }

  PSout output;
  output.color = float4(color, 1.0);

  return output;
}


// Compute Shader
[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 tid: SV_DispatchThreadID)
{
  // TDR: v starts at 0.5 and v *= sin(v) keeps shrinking it toward 0.
  // Once v reaches 0.0 in float32, it stays there forever (0 * sin(0) = 0),
  // but "v <= 10.0" remains true. The SSBO write each iteration is a visible
  // side-effect that prevents the compiler from optimizing the loop away.
  // All 64 threads in the workgroup are stuck, blocking the compute queue
  // -> Windows TDR -> VK_ERROR_DEVICE_LOST.
  float v = 0.5;
  while(v <= 10.0)
  {
    v *= sin(v);
    values[tid.x] = v;
  }
}
