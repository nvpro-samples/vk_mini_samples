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
#version 450

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_descriptor_heap : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_nonuniform_qualifier : enable  // variable indexing into the heap-mapped arrays

#include "shaderio.h"

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 outFragPos;

layout(descriptor_heap, scalar) uniform FrameInfo_
{
  FrameInfo frameInfo;
}
heapFrameInfo[];

layout(push_constant) uniform PushConstant_
{
  PushConstant pushC;
};

void main()
{
  FrameInfo frameInfo = heapFrameInfo[pushC.bufferHeapBase + kHeapBufFrameInfo].frameInfo;
  vec4      pos       = pushC.transfo * vec4(inPosition.xyz, 1.0);
  gl_Position         = frameInfo.proj * frameInfo.view * vec4(pos);
  outFragPos          = pos.xyz;
}
