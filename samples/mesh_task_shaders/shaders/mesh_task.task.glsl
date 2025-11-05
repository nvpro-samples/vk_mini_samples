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

#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_buffer_reference : require

#include "shaderio.h"

// Buffer reference for statistics
layout(buffer_reference, std430, buffer_reference_align = 4) buffer StatisticsRef
{
  uint boxesDrawn;
};

taskPayloadSharedEXT TaskPayload payload;

layout(push_constant) uniform PushConst_
{
  PushConstant pushConst;
};

layout(binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

layout(local_size_x = TASKSHADER_WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;

// Test if a sphere (center + radius) is inside frustum
// Returns true if visible (inside or intersecting frustum)
bool isSphereInFrustum(vec3 center, float radius)
{
  for(int i = 0; i < 6; i++)
  {
    vec3  planeNormal   = frameInfo.frustumPlanes[i].xyz;
    float planeDistance = frameInfo.frustumPlanes[i].w;

    // Distance from plane to sphere center
    float distance = dot(planeNormal, center) + planeDistance;

    // If sphere is completely outside any plane, it's not visible
    if(distance < -radius)
    {
      return false;
    }
  }
  return true;
}

void main()
{
  uint threadID = gl_LocalInvocationID.x;  // 0 to 31

  // Get grid position for this task shader workgroup
  uint gridX = gl_WorkGroupID.x;
  uint gridZ = gl_WorkGroupID.y;

  // Check if this workgroup is within bounds
  uint startBoxX = gridX * BOXES_PER_TASK;
  if(startBoxX >= pushConst.totalBoxesX || gridZ >= pushConst.totalBoxesZ)
  {
    return;  // Outside grid bounds
  }

  // Pass grid position to mesh shader via payload
  if(threadID == 0)
  {
    payload.gridX = gridX;
    payload.gridZ = gridZ;
  }
  barrier();  // Ensure payload is initialized before all threads use it

  // Calculate how many boxes this workgroup will actually test
  uint boxesInThisTile = min(BOXES_PER_TASK, pushConst.totalBoxesX - startBoxX);

  // Each thread tests one box (1:1 mapping - all 32 threads utilized when 32 boxes present)
  uint localBoxIndex = threadID;
  bool boxSurvives   = false;

  if(localBoxIndex < boxesInThisTile)
  {
    // Calculate global box position
    uint globalBoxX = startBoxX + localBoxIndex;

    // Calculate box center position in world space
    float xOffset   = (float(globalBoxX) - float(pushConst.totalBoxesX - 1) * 0.5) * pushConst.spacing;
    float zOffset   = (float(gridZ) - float(pushConst.totalBoxesZ - 1) * 0.5) * pushConst.spacing;
    
    // Ripple animation: calculate distance from grid center
    float distanceFromCenter = length(vec2(xOffset, zOffset));
    float yOffset = calculateRippleOffset(pushConst.time, pushConst.animSpeed, distanceFromCenter);
    
    vec3  boxCenter = vec3(xOffset, yOffset, zOffset);

    // Bounding sphere radius (larger for rotating boxes)
    float boundingRadius = calculateBoxBoundingRadius(pushConst.boxSize, true);

    // Test if this box is visible
    boxSurvives = isSphereInFrustum(boxCenter, boundingRadius);
  }

  // Perform subgroup-wide ballot to collect visibility results from all threads.
  // Each bit in the resulting ballot represents one thread's boxSurvives value.
  uvec4 voteSurviving = subgroupBallot(boxSurvives);

  // Compact surviving boxes into a contiguous array with no gaps.
  // Each thread with a visible box computes its output position by counting how many
  // threads with lower IDs also have visible boxes (exclusive prefix sum of the ballot).
  if(boxSurvives)
  {
    payload.survivingBoxIndices[subgroupBallotExclusiveBitCount(voteSurviving)] = uint8_t(localBoxIndex);
  }

  // Count total number of surviving boxes by counting set bits in the ballot.
  // Only thread 0 performs this work to avoid redundant operations.
  if(threadID == 0)
  {
    payload.numSurvivingBoxes = subgroupBallotBitCount(voteSurviving);
    // Atomically add the number of surviving boxes to the global counter
    StatisticsRef stats = StatisticsRef(pushConst.statisticsAddr);
    atomicAdd(stats.boxesDrawn, payload.numSurvivingBoxes);
  }

  // Emit mesh shader workgroups to process surviving boxes
  // Each mesh shader handles up to BOXES_PER_MESH boxes, so emit ceil(numSurvivingBoxes / BOXES_PER_MESH) workgroups
  if(threadID == 0 && payload.numSurvivingBoxes > 0)
  {
    uint numMeshWorkgroups = (payload.numSurvivingBoxes + BOXES_PER_MESH - 1) / BOXES_PER_MESH;
    EmitMeshTasksEXT(numMeshWorkgroups, 1, 1);
  }
}
