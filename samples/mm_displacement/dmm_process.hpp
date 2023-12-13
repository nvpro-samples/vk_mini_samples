/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <vector>
#include <vulkan/vulkan_core.h>

#include "nvvk/context_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvh/primitives.hpp"

// Setting for the terrain generator
struct Terrain
{
  float seed{0.0F};
  float freq{2.0F};
  float power{2.0F};
  int   octave{4};
};


class MicromapProcess
{

public:
  MicromapProcess(nvvk::Context* ctx, nvvk::ResourceAllocator* allocator);
  ~MicromapProcess();

  bool createMicromapData(VkCommandBuffer cmd, const nvh::PrimitiveMesh& mesh, uint16_t subdivLevel, const Terrain& terrain);
  void createMicromapBuffers(VkCommandBuffer cmd, const nvh::PrimitiveMesh& mesh, const glm::vec2& biasScale);
  void cleanBuildData();

  const nvvk::Buffer&                    primitiveFlags() { return m_primitiveFlags; }
  const nvvk::Buffer&                    displacementDirections() { return m_displacementDirections; }
  const nvvk::Buffer&                    displacementBiasAndScale() { return m_displacementBiasAndScale; }
  const VkMicromapEXT&                   micromap() { return m_micromap; }
  const std::vector<VkMicromapUsageEXT>& usages() { return m_usages; }

private:
  struct MicromapData
  {
    std::vector<uint8_t>               values;
    std::vector<VkMicromapTriangleEXT> triangles;
    std::vector<VkMicromapUsageEXT>    usages;
  };

  // Raw values per triangles
  struct RawTriangle
  {
    uint32_t           subdivLevel{0};
    std::vector<float> values;
  };

  struct MicroDistances
  {
    std::vector<RawTriangle> rawTriangles;
  };


  bool        buildMicromap(VkCommandBuffer cmd);
  static void barrier(VkCommandBuffer cmd);
  static MicroDistances createDisplacements(const nvh::PrimitiveMesh& mesh, uint16_t subdivLevel, const Terrain& terrain);

  VkDevice                 m_device;
  nvvk::ResourceAllocator* m_alloc;

  nvvk::Buffer m_inputData;
  nvvk::Buffer m_microData;
  nvvk::Buffer m_trianglesBuffer;
  nvvk::Buffer m_primitiveFlags;
  nvvk::Buffer m_displacementDirections;
  nvvk::Buffer m_displacementBiasAndScale;
  nvvk::Buffer m_scratchBuffer;

  VkMicromapEXT                   m_micromap{VK_NULL_HANDLE};
  std::vector<VkMicromapUsageEXT> m_usages;
};
