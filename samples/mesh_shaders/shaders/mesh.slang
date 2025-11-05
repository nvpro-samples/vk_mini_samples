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

// Output from mesh shader to fragment shader
struct MeshOutput
{
  float4 position : SV_Position;
  float3 color : COLOR;
};

//--------------------------------------------------------------------------------------------------
// Mesh Shader - generates up to 8 bounding boxes directly from workgroup ID
// No task shader, no culling - renders all boxes
//--------------------------------------------------------------------------------------------------
[shader("mesh")]
[outputtopology("line")]
[numthreads(MESHSHADER_WORKGROUP_SIZE, 1, 1)]
void meshMain(uint3                                                         groupThreadID: SV_GroupThreadID,
              uint3                                                         groupID: SV_GroupID,
              OutputVertices<MeshOutput, BOXES_PER_MESH * VERTICES_PER_BOX> verts,   // 8 boxes * 8 unique vertices
              OutputIndices<uint2, BOXES_PER_MESH * LINES_PER_BOX>          indices  // 8 boxes * 12 lines
)
{
  uint threadID = groupThreadID.x;  // 0 to 31 (MESHSHADER_WORKGROUP_SIZE-1)

  // Calculate base box index from workgroup ID
  // Each workgroup renders up to BOXES_PER_MESH boxes in a 2D grid pattern
  uint baseBoxX = groupID.x * BOXES_PER_MESH;
  uint boxZ     = groupID.y;

  // Calculate how many boxes this workgroup should render (0-8)
  // Need to handle partial workgroups at the edge of the grid
  uint numBoxes = min(BOXES_PER_MESH, pushConst.totalBoxesX - baseBoxX);

  // Early out if this workgroup is completely outside the grid
  if(baseBoxX >= pushConst.totalBoxesX || boxZ >= pushConst.totalBoxesZ)
  {
    SetMeshOutputCounts(0, 0);
    return;
  }

  // Calculate output counts
  uint totalVertices   = numBoxes * VERTICES_PER_BOX;
  uint totalPrimitives = numBoxes * LINES_PER_BOX;

  float boxSize = pushConst.boxSize;
  float spacing = pushConst.spacing;

  // Set primitive count - all threads must reach this point before outputting
  SetMeshOutputCounts(totalVertices, totalPrimitives);

  // Distribute vertex work across all threads
  for(uint vertexIndex = threadID; vertexIndex < totalVertices; vertexIndex += MESHSHADER_WORKGROUP_SIZE)
  {
    uint boxIndex    = vertexIndex / VERTICES_PER_BOX;
    uint cornerIndex = vertexIndex % VERTICES_PER_BOX;

    // Calculate global box position in the grid
    uint globalBoxX = baseBoxX + boxIndex;

    // Procedural positioning: center the entire grid
    float xOffset = (float(globalBoxX) - float(pushConst.totalBoxesX - 1) * 0.5f) * spacing;
    float zOffset = (float(boxZ) - float(pushConst.totalBoxesZ - 1) * 0.5f) * spacing;

    // Gentle wave animation: ocean-like motion (shared function from shaderio.h)
    float yOffset = calculateGentleWaveOffset(pushConst.time, pushConst.animSpeed, float(globalBoxX), float(boxZ));

    float3 boxCenter = float3(xOffset, yOffset, zOffset);
    float  halfSize  = boxSize * 0.5f;

    // Calculate corner position using bit manipulation
    float3 corner;
    corner.x = boxCenter.x + ((cornerIndex & 1) != 0 ? halfSize : -halfSize);
    corner.y = boxCenter.y + ((cornerIndex & 2) != 0 ? halfSize : -halfSize);
    corner.z = boxCenter.z + ((cornerIndex & 4) != 0 ? halfSize : -halfSize);

    float4 clipPos = mul(mul(float4(corner, 1.0f), frameInfo.view), frameInfo.proj);

    // Color based on global grid position
    float3 color;
    color.r = (globalBoxX % 2 == 0) ? 0.8f : 0.3f;
    color.g = (boxZ % 2 == 0) ? 0.8f : 0.3f;
    color.b = ((globalBoxX + boxZ) % 2 == 0) ? 0.8f : 0.3f;

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
