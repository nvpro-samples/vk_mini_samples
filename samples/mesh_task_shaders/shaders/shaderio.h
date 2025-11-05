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

// Constants - are overridden at compile time by the application
#ifndef MESHSHADER_WORKGROUP_SIZE
#define MESHSHADER_WORKGROUP_SIZE 32U
#endif

#ifndef TASKSHADER_WORKGROUP_SIZE
#define TASKSHADER_WORKGROUP_SIZE 32U
#endif


static const uint BOXES_PER_TASK = TASKSHADER_WORKGROUP_SIZE;  // Number of boxes per task workgroup (1:1 mapping with threads)
static const uint BOXES_PER_MESH = 8U;  // Max boxes per mesh shader workgroup (limited by max_vertices = 64 / 8 vertices per box)
static const uint VERTICES_PER_BOX = 8;
static const uint LINES_PER_BOX    = 12;

struct PushConstant
{
  uint32_t totalBoxesX;  // Total number of boxes in X dimension
  uint32_t totalBoxesZ;  // Total number of boxes in Z dimension
  float    boxSize;
  float    spacing;
  uint64_t statisticsAddr;  // Buffer device address for statistics buffer
  float    time;            // Animation time
  float    animSpeed;       // Animation speed multiplier
};

struct FrameInfo
{
  float4x4 proj;
  float4x4 view;
  float3   camPos;
  float    _pad0;
  float4   frustumPlanes[6];  // Left, Right, Bottom, Top, Near, Far (xyz=normal, w=distance)
};

// Task mesh payload (shared between task and mesh shader)
struct TaskPayload
{
  uint    gridX;
  uint    gridZ;
  uint    numSurvivingBoxes;                    // Number of boxes that passed frustum culling
  uint8_t survivingBoxIndices[BOXES_PER_TASK];  // Local indices (0-31) of boxes that survived
};

// Statistics buffer for atomic counters
struct Statistics
{
  uint32_t boxesDrawn;  // Total number of boxes that passed frustum culling and were drawn
};

NAMESPACE_SHADERIO_END()

//--------------------------------------------------------------------------------------------------
// Animation helper functions (shader-only, shared between GLSL and Slang)
//--------------------------------------------------------------------------------------------------
#ifdef __cplusplus
// C++ host code - skip these functions
#else
// Shader code - define animation helpers

// Calculate ripple-from-center animation (radial wave effect)
// Used in mesh_task_shaders sample - boxes rise and fall in circular waves
float calculateRippleOffset(float time, float animSpeed, float distanceFromCenter)
{
  // Radial wave propagating outward from center
  float wave = sin(time * 3.0 - distanceFromCenter * 0.5) * 2.0;
  return max(0.0, wave) * animSpeed;  // Only positive (boxes rise up, not go down)
}

// Calculate box rotation angle based on distance and time
// Boxes spin faster when they're at wave peaks
float calculateBoxRotation(float time, float animSpeed, float distanceFromCenter, float yOffset)
{
  // Base rotation speed varies with distance
  float baseRotation = time * 2.0 * animSpeed;
  // Extra spin when box is elevated (at wave peak)
  float elevationBoost = yOffset;  // * 0.5;
  return baseRotation + elevationBoost + distanceFromCenter * 0.3;
}

// Calculate bounding sphere radius for a box
// This is the radius of a sphere that exactly encompasses a cube
// Add padding if box is rotating
float calculateBoxBoundingRadius(float boxSize, bool rotating)
{
  float baseRadius = boxSize * sqrt(3.0) * 0.5;
  // When rotating, we need a slightly larger sphere to ensure box stays inside
  // The diagonal of the box in XZ plane is boxSize * sqrt(2)
  return rotating ? (boxSize * sqrt(2.0) * 0.5) : baseRadius;
}

// Rotate a 3D point around Y axis
// Used to rotate boxes on themselves
float3 rotateY(float3 point, float angle)
{
  float c = cos(angle);
  float s = sin(angle);
  return float3(point.x * c - point.z * s, point.y, point.x * s + point.z * c);
}

#endif  // __cplusplus

#endif
