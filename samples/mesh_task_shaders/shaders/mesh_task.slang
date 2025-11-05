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

// Output from mesh shader to fragment shader
struct MeshOutput
{
  float4 position : SV_Position;
  float3 color : COLOR;
};

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
    return;  // Outside grid bounds
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
    float xOffset = (float(globalBoxX) - float(pushConst.totalBoxesX - 1) * 0.5) * pushConst.spacing;
    float zOffset = (float(gridZ) - float(pushConst.totalBoxesZ - 1) * 0.5) * pushConst.spacing;

    // Ripple animation: calculate distance from grid center
    float distanceFromCenter = length(float2(xOffset, zOffset));
    float yOffset            = calculateRippleOffset(pushConst.time, pushConst.animSpeed, distanceFromCenter);

    float3 boxCenter = float3(xOffset, yOffset, zOffset);

    // Bounding sphere radius (larger for rotating boxes)
    float boundingRadius = calculateBoxBoundingRadius(pushConst.boxSize, true);

    // Test if this box is visible
    boxSurvives = isSphereInFrustum(boxCenter, boundingRadius);
  }

  if(boxSurvives)
  {
    // Compact surviving boxes into a contiguous array with no gaps.
    // Each thread with a visible box computes its output position by counting how many
    // threads with lower lane IDs also have visible boxes (exclusive prefix sum).
    uint offset                             = WavePrefixCountBits(boxSurvives);
    taskPayload.survivingBoxIndices[offset] = uint8_t(localBoxIndex);
  }

  // Count total number of surviving boxes across the entire wave.
  uint numSurvive = WaveActiveCountBits(boxSurvives);

  // Store total count of surviving boxes.
  // Only thread 0 performs this work to avoid redundant operations.
  if(threadID == 0)
  {
    taskPayload.numSurvivingBoxes = numSurvive;
    // Atomically add the number of surviving boxes to the global counter using buffer device address
    Statistics* stats = (Statistics*)(pushConst.statisticsAddr);
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

//--------------------------------------------------------------------------------------------------
// Mesh Shader - generates up to 8 bounding boxes from compacted survivor list
// Uses proper work distribution across all threads in the workgroup
//--------------------------------------------------------------------------------------------------
[shader("mesh")]
[outputtopology("line")]
[numthreads(MESHSHADER_WORKGROUP_SIZE, 1, 1)]
void meshMain(uint3 groupThreadID: SV_GroupThreadID,
              uint3 groupID: SV_GroupID,
              // NOTE: Payload parameter is commented out for multi-entry point Slang compilation.
              // In single-file mode, both shaders access the groupshared 'taskPayload' directly (line 32).
              // In separate-file mode (mesh_boxes.mesh.slang), the payload is passed as a function parameter.
              // This might be a Slang compiler limitation with multi-entry point files.
              //in payload TaskPayload         taskPayload,
              OutputVertices<MeshOutput, BOXES_PER_MESH * VERTICES_PER_BOX> verts,   // 8 boxes * 8 unique vertices
              OutputIndices<uint2, BOXES_PER_MESH * LINES_PER_BOX>          indices  // 8 boxes * 12 lines
)
{
  uint threadID        = groupThreadID.x;  // 0 to 31
  uint meshWorkgroupID = groupID.x;        // Which mesh shader workgroup (0-3, since we can emit up to 4)

  // Get grid position from task shader payload
  uint gridX = taskPayload.gridX;
  uint gridZ = taskPayload.gridZ;

  // Each mesh workgroup processes up to BOXES_PER_MESH boxes from the compacted survivor list
  uint baseBoxOffset = meshWorkgroupID * BOXES_PER_MESH;

  // Calculate how many boxes this mesh workgroup should render (0-8)
  uint numBoxes = min(BOXES_PER_MESH, taskPayload.numSurvivingBoxes - baseBoxOffset);

  // Calculate output counts
  uint totalVertices   = numBoxes * VERTICES_PER_BOX;
  uint totalPrimitives = numBoxes * LINES_PER_BOX;

  float boxSize = pushConst.boxSize;
  float spacing = pushConst.spacing;

  // Set primitive count - all threads must reach this point before outputting
  SetMeshOutputCounts(totalVertices, totalPrimitives);

  // Distribute vertex work across all threads
  uint startBoxX = gridX * BOXES_PER_TASK;

  for(uint vertexIndex = threadID; vertexIndex < totalVertices; vertexIndex += MESHSHADER_WORKGROUP_SIZE)
  {
    uint boxIndex    = vertexIndex / VERTICES_PER_BOX;
    uint cornerIndex = vertexIndex % VERTICES_PER_BOX;

    // Get the local box index from the surviving boxes list (offset by baseBoxOffset)
    uint localBoxIndex = taskPayload.survivingBoxIndices[baseBoxOffset + boxIndex];

    // Calculate global box position in the grid
    uint globalBoxX = startBoxX + localBoxIndex;

    // Procedural positioning: center the entire grid
    float xOffset = (float(globalBoxX) - float(pushConst.totalBoxesX - 1) * 0.5f) * spacing;
    float zOffset = (float(gridZ) - float(pushConst.totalBoxesZ - 1) * 0.5f) * spacing;

    // Ripple animation: calculate distance from grid center
    float distanceFromCenter = length(float2(xOffset, zOffset));
    float yOffset            = calculateRippleOffset(pushConst.time, pushConst.animSpeed, distanceFromCenter);

    float3 boxCenter = float3(xOffset, yOffset, zOffset);
    float  halfSize  = boxSize * 0.5f;

    // Calculate corner position (relative to box center) using bit manipulation
    float3 cornerLocal;
    cornerLocal.x = ((cornerIndex & 1) != 0 ? halfSize : -halfSize);
    cornerLocal.y = ((cornerIndex & 2) != 0 ? halfSize : -halfSize);
    cornerLocal.z = ((cornerIndex & 4) != 0 ? halfSize : -halfSize);

    // Apply rotation around box's own Y axis
    float  rotationAngle = calculateBoxRotation(pushConst.time, pushConst.animSpeed, distanceFromCenter, yOffset);
    float3 cornerRotated = rotateY(cornerLocal, rotationAngle);

    // Final world position
    float3 corner = boxCenter + cornerRotated;

    float4 clipPos = mul(mul(float4(corner, 1.0f), frameInfo.view), frameInfo.proj);

    // Color based on global grid position
    float3 color;
    color.r = (globalBoxX % 2 == 0) ? 0.8f : 0.3f;
    color.g = (gridZ % 2 == 0) ? 0.8f : 0.3f;
    color.b = ((globalBoxX + gridZ) % 2 == 0) ? 0.8f : 0.3f;

    verts[vertexIndex].position = clipPos;
    verts[vertexIndex].color    = color;
  }

  // Distribute primitive work across all threads
  uint corners[4] = { 0, 1, 3, 2 };
  for(uint primitiveIndex = threadID; primitiveIndex < totalPrimitives; primitiveIndex += MESHSHADER_WORKGROUP_SIZE)
  {
    uint boxIndex   = primitiveIndex / LINES_PER_BOX;
    uint edgeIndex  = primitiveIndex % LINES_PER_BOX;
    uint baseVertex = boxIndex * VERTICES_PER_BOX;

    uint v0, v1;
    if(edgeIndex < 4)
    {
      // Bottom face: 0-1, 1-3, 3-2, 2-0
      v0 = corners[edgeIndex];
      v1 = corners[(edgeIndex + 1) & 3];
    }
    else if(edgeIndex < 8)
    {
      // Top face: 4-5, 5-7, 7-6, 6-4
      uint localEdge = edgeIndex - 4;
      v0             = corners[localEdge] + 4;
      v1             = corners[(localEdge + 1) & 3] + 4;
    }
    else
    {
      // Vertical edges: 0-4, 1-5, 3-7, 2-6
      uint localEdge = edgeIndex - 8;
      v0             = corners[localEdge];
      v1             = v0 + 4;
    }
    v0 += baseVertex;
    v1 += baseVertex;

    indices[primitiveIndex] = uint2(v0, v1);
  }
}

//--------------------------------------------------------------------------------------------------
// Fragment Shader - simple color output
//--------------------------------------------------------------------------------------------------
[shader("pixel")]
float4 fragmentMain(MeshOutput input)
    : SV_Target
{
  return float4(input.color, 1.0f);
}
