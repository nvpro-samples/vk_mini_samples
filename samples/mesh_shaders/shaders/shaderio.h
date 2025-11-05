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

#ifndef SHADERIO_H
#define SHADERIO_H

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

// Constants - can be overridden at compile time
#ifndef MESHSHADER_WORKGROUP_SIZE
#define MESHSHADER_WORKGROUP_SIZE 32U
#endif

// Max boxes per mesh shader workgroup
#ifndef BOXES_PER_MESH
#define BOXES_PER_MESH 8U
#endif

static const uint VERTICES_PER_BOX = 8;
static const uint LINES_PER_BOX    = 12;

struct PushConstant
{
  uint32_t totalBoxesX;  // Total number of boxes in X dimension
  uint32_t totalBoxesZ;  // Total number of boxes in Z dimension
  float    boxSize;
  float    spacing;
  float    time;       // Animation time
  float    animSpeed;  // Animation speed multiplier
};

struct FrameInfo
{
  float4x4 proj;
  float4x4 view;
  float3   camPos;
  float    _pad0;
};

NAMESPACE_SHADERIO_END()

//--------------------------------------------------------------------------------------------------
// Animation helper functions (shader-only, shared between GLSL and Slang)
//--------------------------------------------------------------------------------------------------
#ifdef __cplusplus
// C++ host code - skip these functions
#else
// Shader code - define animation helpers

// Calculate gentle wave animation (ocean-like motion)
// Used in mesh_shaders sample for all boxes
float calculateGentleWaveOffset(float time, float animSpeed, float boxX, float boxZ)
{
  float wave = sin(time * 2.0f + boxX * 0.2f) * cos(time * 1.5f + boxZ * 0.2f);
  return wave * 0.5f * animSpeed;
}

#endif  // __cplusplus

#endif
