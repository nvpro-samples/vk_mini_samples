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

#include "shaderio.h"

[[vk::push_constant]]
ConstantBuffer<PushConstant> pushConst;
[[vk::binding(0)]]
ConstantBuffer<FrameInfo> frameInfo;


// Prepare payload to pass to mesh shader
groupshared TaskPayload taskPayload;

// Test if a sphere (center + radius) is inside frustum
// Returns true if visible (inside or intersecting frustum)
bool isSphereInFrustum(float3 center, float radius)
{
  for(int i = 0; i < 6; i++)
  {
    float3 planeNormal   = frameInfo.frustumPlanes[i].xyz;
    float  planeDistance = frameInfo.frustumPlanes[i].w;

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

//--------------------------------------------------------------------------------------------------
// Task Shader - Per-box GPU culling with compaction
// Each thread tests one box against the frustum (32 threads test 32 boxes)
//--------------------------------------------------------------------------------------------------
[shader("amplification")]
[numthreads(TASKSHADER_WORKGROUP_SIZE, 1, 1)]
void taskMain(uint3 groupThreadID: SV_GroupThreadID, uint3 groupID: SV_GroupID)
{
  uint threadID = groupThreadID.x;  // 0 to 31

  // Get grid position for this task shader workgroup
  uint gridX = groupID.x;
  uint gridZ = groupID.y;

  // Check if this workgroup is within bounds
  uint startBoxX = gridX * BOXES_PER_TASK;
  if(startBoxX >= pushConst.totalBoxesX || gridZ >= pushConst.totalBoxesZ)
  {
    return;  // Outside grid boundsB
  }


  // Pass grid position to mesh shader via payload
  if(threadID == 0)
  {
    taskPayload.gridX = gridX;
    taskPayload.gridZ = gridZ;
  }
  GroupMemoryBarrierWithGroupSync();  // Ensure payload is initialized before all threads use it

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
    float  xOffset   = (float(globalBoxX) - float(pushConst.totalBoxesX - 1) * 0.5) * pushConst.spacing;
    float  zOffset   = (float(gridZ) - float(pushConst.totalBoxesZ - 1) * 0.5) * pushConst.spacing;
    
    // Ripple animation: calculate distance from grid center
    float distanceFromCenter = length(float2(xOffset, zOffset));
    float yOffset = calculateRippleOffset(pushConst.time, pushConst.animSpeed, distanceFromCenter);
    
    float3 boxCenter = float3(xOffset, yOffset, zOffset);

    // Bounding sphere radius (larger for rotating boxes)
    float boundingRadius = calculateBoxBoundingRadius(pushConst.boxSize, true);

    // Test if this box is visible
    boxSurvives = isSphereInFrustum(boxCenter, boundingRadius);
  }

  uint  numSurvive    = WaveActiveCountBits(boxSurvives);
  uint  offset        = WavePrefixCountBits(boxSurvives);

  if(boxSurvives)
  {
    taskPayload.survivingBoxIndices[offset] = uint8_t(threadID);
  }

  if(threadID == 0)
  {
    taskPayload.numSurvivingBoxes = numSurvive;
    // Atomically add the number of surviving boxes to the global counter using buffer device address
    Statistics *stats = (Statistics *)(pushConst.statisticsAddr);
    InterlockedAdd(stats->boxesDrawn, numSurvive);
  }

  // Emit mesh shader workgroups to process surviving boxes
  // Each mesh shader handles up to BOXES_PER_MESH boxes, so emit ceil(numSurvivingBoxes / BOXES_PER_MESH) workgroups
  if(threadID == 0 && taskPayload.numSurvivingBoxes > 0)
  {
    uint numMeshWorkgroups = (taskPayload.numSurvivingBoxes + BOXES_PER_MESH - 1) / BOXES_PER_MESH;
    DispatchMesh(numMeshWorkgroups, 1, 1, taskPayload);
  }
}
