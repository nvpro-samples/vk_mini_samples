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

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio.h"

layout(location = 0) in vec3 inFragColor;
layout(location = 1) in vec2 inUv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};
layout(set = 0, binding = 1) buffer Testing_
{
  float values[];
};

layout(buffer_reference, scalar) buffer WriteTarget_
{
  float v;
};

layout(buffer_reference, scalar) buffer IndirectPtr_
{
  uint64_t targetAddr;
};

layout(constant_id = 0) const int CRASH_TEST = 0;


void main()
{
  float iTime = frameInfo.time.x;

  // Ripple color
  vec3  col = vec3(inUv, 0.5 + 0.5 * sin(iTime * 6.5));
  float r   = length(vec2(0.5, 0.5) - inUv);
  float z   = 1.0 + 0.5 * sin((r + iTime * 0.05) / 0.005);
  col *= z;

  // ---------------------------------------------------------------------------
  // TDR CRASH: Infinite loop using sin()
  // ---------------------------------------------------------------------------
  if(CRASH_TEST == 1)
  {
    // TDR: sin(x) for x in (0,1) returns a value in (0,1), so multiplying
    // col.x by sin(col.x) keeps col.x small and it never exceeds 10.
    // The GPU cannot complete this shader, triggering Windows TDR
    // (Timeout Detection and Recovery) after ~2 seconds -> VK_ERROR_DEVICE_LOST.
    col = min(col, vec3(1.0));
    while(col.x <= 10.0)
    {
      col.x *= sin(col.x);
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
    float v = min(col.x, 0.99);
    while(v <= 10.0)
    {
      v *= sin(v);
      values[0] = v;
    }
    col *= values[0];
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
    WriteTarget_ ptr = WriteTarget_(frameInfo.bufferAddr);
    ptr.v = 1.0;
    col += ptr.v;
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
    uint  h    = uint(gl_FragCoord.x) * 2654435761u ^ uint(gl_FragCoord.y) * 340573321u;
    uint64_t addr = uint64_t(h) << 8;  // spread across 40-bit VA range
    addr &= ~uint64_t(3);              // 4-byte alignment
    addr += 4096u;                     // skip null page region
    WriteTarget_ ptr = WriteTarget_(addr);
    ptr.v = 1.0;
    col += ptr.v;
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
    IndirectPtr_ indirect = IndirectPtr_(frameInfo.bufferAddr);
    WriteTarget_ target   = WriteTarget_(indirect.targetAddr);
    target.v = 1.0;
    col += target.v;
  }

  outColor = vec4(col, 1.0);
}
