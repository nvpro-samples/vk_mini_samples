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

#include "shaderio.h"

layout(local_size_x = 64) in;

layout(set = 0, binding = 1) buffer Testing_
{
  float values[];
};

void main()
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
    values[gl_LocalInvocationIndex] = v;
  }
}
