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

#include "shaderio.h"


taskPayloadSharedEXT TaskPayload payload;


layout(push_constant) uniform PushConst_
{
  PushConstant pushConst;
};

layout(binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};


layout(local_size_x = MESHSHADER_WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
layout(lines) out;
layout(max_vertices = BOXES_PER_MESH * VERTICES_PER_BOX, max_primitives = BOXES_PER_MESH * LINES_PER_BOX) out;

layout(location = 0) out vec3 outColor[];

void main()
{
  uint threadID        = gl_LocalInvocationID.x;  // 0 to 31
  uint meshWorkgroupID = gl_WorkGroupID.x;        // Which mesh shader workgroup (0-3, since we can emit up to 4)

  // Get grid position from task shader payload
  uint gridX = payload.gridX;
  uint gridZ = payload.gridZ;

  // Each mesh workgroup processes up to BOXES_PER_MESH boxes from the compacted survivor list
  uint baseBoxOffset = meshWorkgroupID * BOXES_PER_MESH;

  // Calculate how many boxes this mesh workgroup should render (0-BOXES_PER_MESH)
  uint numBoxes = min(BOXES_PER_MESH, payload.numSurvivingBoxes - baseBoxOffset);

  // Calculate output counts
  uint totalVertices   = numBoxes * VERTICES_PER_BOX;
  uint totalPrimitives = numBoxes * LINES_PER_BOX;

  // Calculate box position and geometry
  float boxSize = pushConst.boxSize;
  float spacing = pushConst.spacing;


  // Distribute work across all threads in the workgroup
  uint startBoxX = gridX * BOXES_PER_TASK;

  for(uint vertexIndex = threadID; vertexIndex < totalVertices; vertexIndex += MESHSHADER_WORKGROUP_SIZE)
  {
    uint boxIndex    = vertexIndex / VERTICES_PER_BOX;
    uint cornerIndex = vertexIndex % VERTICES_PER_BOX;

    // Get the local box index from the surviving boxes list (offset by baseBoxOffset)
    uint localBoxIndex = payload.survivingBoxIndices[baseBoxOffset + boxIndex];

    // Calculate global box position in the grid
    uint globalBoxX = startBoxX + localBoxIndex;

    // Procedural positioning: center the entire grid
    float xOffset = (float(globalBoxX) - float(pushConst.totalBoxesX - 1) * 0.5) * spacing;
    float zOffset = (float(gridZ) - float(pushConst.totalBoxesZ - 1) * 0.5) * spacing;

    // Ripple animation: calculate distance from grid center
    float distanceFromCenter = length(vec2(xOffset, zOffset));
    float yOffset = calculateRippleOffset(pushConst.time, pushConst.animSpeed, distanceFromCenter);

    vec3  boxCenter = vec3(xOffset, yOffset, zOffset);
    float halfSize  = boxSize * 0.5;

    // Calculate corner position (relative to box center) using bit manipulation
    vec3 cornerLocal;
    cornerLocal.x = ((cornerIndex & 1) != 0 ? halfSize : -halfSize);
    cornerLocal.y = ((cornerIndex & 2) != 0 ? halfSize : -halfSize);
    cornerLocal.z = ((cornerIndex & 4) != 0 ? halfSize : -halfSize);
    
    // Apply rotation around box's own Y axis
    float rotationAngle = calculateBoxRotation(pushConst.time, pushConst.animSpeed, distanceFromCenter, yOffset);
    vec3 cornerRotated = rotateY(cornerLocal, rotationAngle);
    
    // Final world position
    vec3 corner = boxCenter + cornerRotated;

    vec4 clipPos = frameInfo.proj * frameInfo.view * vec4(corner, 1.0);

    // Color based on global grid position (creates a checkerboard pattern)
    vec3 color;
    color.r = (globalBoxX % 2 == 0) ? 0.8 : 0.3;
    color.g = (gridZ % 2 == 0) ? 0.8 : 0.3;
    color.b = ((globalBoxX + gridZ) % 2 == 0) ? 0.8 : 0.3;

    gl_MeshVerticesEXT[vertexIndex].gl_Position = clipPos;
    outColor[vertexIndex]                       = color;
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

    gl_PrimitiveLineIndicesEXT[primitiveIndex] = uvec2(v0, v1);
  }

  // Set primitive count (only first thread)
  if(threadID == 0)
  {
    SetMeshOutputsEXT(totalVertices, totalPrimitives);
  }
}
