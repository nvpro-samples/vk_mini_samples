/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <vector>
#include <vulkan/vulkan_core.h>

#include "nvvk/resource_allocator.hpp"
#include "nvutils/primitives.hpp"
#include "nvvk/staging.hpp"


class MicromapProcess
{

public:
  // useKHR: use VK_KHR_opacity_micromap build path (VkAccelerationStructureKHR),
  //         otherwise use VK_EXT_opacity_micromap (VkMicromapEXT).
  MicromapProcess(nvvk::ResourceAllocator* allocator, bool useKHR = false);
  ~MicromapProcess();

  bool createMicromapData(VkCommandBuffer               cmd,
                          nvvk::StagingUploader&        uploader,
                          const nvutils::PrimitiveMesh& mesh,
                          uint16_t                      subdivLevel,
                          float                         radius,
                          uint16_t                      micromapFormat);
  void cleanBuildData();

  bool                             useKHR() const { return m_useKHR; }
  const VkMicromapEXT&             micromap() const { return m_micromap; }
  const VkAccelerationStructureKHR micromapAS() const { return m_micromapAS; }
  const nvvk::Buffer&              indexBuffer() const { return m_indexBuffer; }

private:
  // Raw values per triangles
  struct RawTriangle
  {
    uint32_t         subdivLevel{0};
    std::vector<int> values;
  };

  struct MicroOpacity
  {
    std::vector<RawTriangle> rawTriangles;
  };


  bool                buildMicromapEXT(VkCommandBuffer cmd, VkMicromapTypeEXT type);
  bool                buildMicromapKHR(VkCommandBuffer cmd);
  void                barrier(VkCommandBuffer cmd);
  static MicroOpacity createOpacity(const nvutils::PrimitiveMesh& mesh, uint16_t subdivLevel, float radius);

  VkDevice                 m_device;
  nvvk::ResourceAllocator* m_alloc;
  bool                     m_useKHR{false};

  nvvk::Buffer m_inputData;
  nvvk::Buffer m_microData;
  nvvk::Buffer m_trianglesBuffer;
  nvvk::Buffer m_scratchBuffer;
  nvvk::Buffer m_indexBuffer;

  // EXT path
  VkMicromapEXT                   m_micromap{VK_NULL_HANDLE};
  std::vector<VkMicromapUsageEXT> m_usages;

  // KHR path
  VkAccelerationStructureKHR      m_micromapAS{VK_NULL_HANDLE};
  std::vector<VkMicromapUsageKHR> m_usagesKHR;

  VkPhysicalDeviceOpacityMicromapPropertiesKHR m_oppacityProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_KHR};
};
