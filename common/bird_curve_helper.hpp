/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <vector>
#include "nvmath/nvmath.h"

// Interpolate the 3 values with bary; using auto as a templated function
inline auto getInterpolated = [](const auto& v0, const auto& v1, const auto& v2, const nvmath::vec3f& bary) {
  return v0 * bary.x + v1 * bary.y + v2 * bary.z;
};


//--------------------------------------------------------------------------------------------------
// Holds the barycentric values for each level, in the order the GPU is expecting them
//
class BirdCurveHelper
{
public:
  struct SubTriangle
  {
    nvmath::vec3f w;
    nvmath::vec3f u;
    nvmath::vec3f v;
  };
  using BaryCoordinates   = std::vector<nvmath::vec3f>;  // Barycentric coordinates in Bird Curve order
  using BaryIndices       = std::vector<nvmath::vec3i>;  // Array of triangle indices in Bird Curve order
  using DisplacementBlock = std::vector<uint32_t>;       // Indices of BaryCoordinates for a subdivision level.
  using DisplacementBlocks = std::vector<DisplacementBlock>;  // Level 0..3 uses one block and is linear, Level 4 needs four blocks, level 5 needs sixteen blocks

  // Constructor, initialize up to `maxLevel` subdivision
  explicit BirdCurveHelper(uint32_t             maxLevel = 5U,
                           const nvmath::vec3f& w        = {1, 0, 0},
                           const nvmath::vec3f& u        = {0, 1, 0},
                           const nvmath::vec3f& v        = {0, 0, 1});

  static constexpr inline uint32_t getNumMicroVertices(uint32_t level)
  {
    return (((1 << level) + 1) * ((1 << level) + 2)) >> 1;
  }
  static constexpr inline uint32_t getNumMicroTriangles(uint32_t level) { return 1 << (level << 1u); }


  const BaryCoordinates& getVertexCoord(uint16_t level);      // Returns all bary coordinate values of the `level`
  const BaryIndices&     getTriangleIndices(uint16_t level);  // Returns triplet of indices making sub-triangles
  const std::vector<SubTriangle>& getTriangleCoord(uint16_t level);  // Returns triplet of coordinates making sub-triangles

  DisplacementBlocks createDisplacementBlocks(uint32_t level);  // Return the displacement blocks: block of indices for unorm11 uncompressed

  static void micro2bary(uint32_t index, uint32_t subdivisionLevel, nvmath::vec3f& uv0, nvmath::vec3f& uv1, nvmath::vec3f& uv2);

private:
  void init(const nvmath::vec3f& w, const nvmath::vec3f& u, const nvmath::vec3f& v);
  void birdLevel(int level, bool triPointUp, bool counterClockwise, const nvmath::vec3f& w, const nvmath::vec3f& u, const nvmath::vec3f& v);
  static std::vector<SubTriangle> splitTriangles(const nvmath::vec3f& w, const nvmath::vec3f& u, const nvmath::vec3f& v);

  uint32_t                              m_maxLevel{0};
  std::vector<BaryCoordinates>          m_birdValues;   // Barycentric coordinates (vertex) per level
  std::vector<BaryIndices>              m_birdIndices;  // Triplet of indices making a triangle
  std::vector<std::vector<SubTriangle>> m_triBary;      // Barycentric coordinates (triplet of sub-triangle) per level
};
