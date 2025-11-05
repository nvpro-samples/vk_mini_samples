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

#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_mesh_shader : require
#extension GL_GOOGLE_include_directive : require

#include "shaderio.h"

layout(push_constant) uniform PushConstantBlock
{
  PushConstant pushConst;
};

layout(binding = 0) uniform FrameInfoBlock
{
  FrameInfo frameInfo;
};

// Output from mesh shader to fragment shader
layout(location = 0) out vec3 outColor[];

//--------------------------------------------------------------------------------------------------
// Mesh Shader - generates up to 8 bounding boxes directly from workgroup ID
// No task shader, no culling - renders all boxes
//--------------------------------------------------------------------------------------------------
layout(local_size_x = MESHSHADER_WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
layout(lines) out;
layout(max_vertices = BOXES_PER_MESH * VERTICES_PER_BOX, max_primitives = BOXES_PER_MESH * LINES_PER_BOX) out;

void main()
{
  uint threadID = gl_LocalInvocationID.x;  // 0 to 31

  // Calculate base box index from workgroup ID
  // Each workgroup renders up to BOXES_PER_MESH boxes in a 2D grid pattern
  uint baseBoxX = gl_WorkGroupID.x * BOXES_PER_MESH;
  uint boxZ     = gl_WorkGroupID.y;

  // Calculate how many boxes this workgroup should render (0-8)
  // Need to handle partial workgroups at the edge of the grid
  uint numBoxes = min(BOXES_PER_MESH, pushConst.totalBoxesX - baseBoxX);

  // Early out if this workgroup is completely outside the grid
  if(baseBoxX >= pushConst.totalBoxesX || boxZ >= pushConst.totalBoxesZ)
  {
    SetMeshOutputsEXT(0, 0);
    return;
  }

  // Calculate output counts
  uint totalVertices   = numBoxes * VERTICES_PER_BOX;
  uint totalPrimitives = numBoxes * LINES_PER_BOX;

  float boxSize = pushConst.boxSize;
  float spacing = pushConst.spacing;

  // Distribute vertex work across all threads
  for(uint vertexIndex = threadID; vertexIndex < totalVertices; vertexIndex += MESHSHADER_WORKGROUP_SIZE)
  {
    uint boxIndex    = vertexIndex / VERTICES_PER_BOX;
    uint cornerIndex = vertexIndex % VERTICES_PER_BOX;

    // Calculate global box position in the grid
    uint globalBoxX = baseBoxX + boxIndex;

    // Procedural positioning: center the entire grid
    float xOffset = (float(globalBoxX) - float(pushConst.totalBoxesX - 1) * 0.5) * spacing;
    float zOffset = (float(boxZ) - float(pushConst.totalBoxesZ - 1) * 0.5) * spacing;

    // Gentle wave animation: ocean-like motion (shared function from shaderio.h)
    float yOffset = calculateGentleWaveOffset(pushConst.time, pushConst.animSpeed, float(globalBoxX), float(boxZ));

    vec3  boxCenter = vec3(xOffset, yOffset, zOffset);
    float halfSize  = boxSize * 0.5;

    // Calculate corner position using bit manipulation
    vec3 corner;
    corner.x = boxCenter.x + ((cornerIndex & 1u) != 0u ? halfSize : -halfSize);
    corner.y = boxCenter.y + ((cornerIndex & 2u) != 0u ? halfSize : -halfSize);
    corner.z = boxCenter.z + ((cornerIndex & 4u) != 0u ? halfSize : -halfSize);

    vec4 clipPos = frameInfo.proj * frameInfo.view * vec4(corner, 1.0);

    // Color based on global grid position
    vec3 color;
    color.r = (globalBoxX % 2u == 0u) ? 0.8 : 0.3;
    color.g = (boxZ % 2u == 0u) ? 0.8 : 0.3;
    color.b = ((globalBoxX + boxZ) % 2u == 0u) ? 0.8 : 0.3;

    gl_MeshVerticesEXT[vertexIndex].gl_Position = clipPos;
    outColor[vertexIndex]                       = color;
  }

  // Distribute primitive work across all threads
  uint corners[4] = uint[4](0u, 1u, 3u, 2u);
  for(uint primitiveIndex = threadID; primitiveIndex < totalPrimitives; primitiveIndex += MESHSHADER_WORKGROUP_SIZE)
  {
    uint boxIndex   = primitiveIndex / LINES_PER_BOX;
    uint edgeIndex  = primitiveIndex % LINES_PER_BOX;
    uint baseVertex = boxIndex * VERTICES_PER_BOX;

    uint v0, v1;
    if(edgeIndex < 4u)
    {
      // Bottom face: 0-1, 1-3, 3-2, 2-0
      v0 = corners[edgeIndex];
      v1 = corners[(edgeIndex + 1u) & 3u];
    }
    else if(edgeIndex < 8u)
    {
      // Top face: 4-5, 5-7, 7-6, 6-4
      uint localEdge = edgeIndex - 4u;
      v0             = corners[localEdge] + 4u;
      v1             = corners[(localEdge + 1u) & 3u] + 4u;
    }
    else
    {
      // Vertical edges: 0-4, 1-5, 3-7, 2-6
      uint localEdge = edgeIndex - 8u;
      v0             = corners[localEdge];
      v1             = v0 + 4u;
    }
    v0 += baseVertex;
    v1 += baseVertex;

    gl_PrimitiveLineIndicesEXT[primitiveIndex] = uvec2(v0, v1);
  }

  // Set output counts (must be called by thread 0 after all work is done)
  if(threadID == 0u)
  {
    SetMeshOutputsEXT(totalVertices, totalPrimitives);
  }
}
