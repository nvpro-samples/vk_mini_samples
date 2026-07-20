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

#ifndef SHADERIO_H
#define SHADERIO_H

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

#define WORKGROUP_SIZE 16

// Descriptor-heap slot layout (local indices within each heap region).
static const uint kHeapBufFrameInfo = 0;  // buffer region: per-frame camera UBO
static const uint kHeapImgVolume    = 0;  // image region: 3D volume texture
static const uint kHeapSmpVolume    = 0;  // sampler region: volume sampler

struct FrameInfo
{
  float4x4 proj;
  float4x4 view;
  float3   camPos;
  float3   toLight;
  int      headlight;
};

struct PushConstant
{
  float4x4 transfo;
  float4   color;
  float    threshold;
  int      steps;
  int      size;
  uint     bufferHeapBase;
};


struct PerlinSettings
{
  int octave      SLANG_DEFAULT(3);
  float power     SLANG_DEFAULT(1.0f);
  float frequency SLANG_DEFAULT(1.0f);
};

NAMESPACE_SHADERIO_END()

#endif
