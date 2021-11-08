/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once
//////////////////////////////////////////////////////////////////////////


#include <array>
#include <vector>

#include "nvvk/debug_util_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "shaders/host_device.h"


//--------------------------------------------------------------------------------------------------
// Load an environment image (HDR) and create an acceleration structure for
// important light sampling.
class HdrEnv
{
public:
  HdrEnv() = default;

  void  setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator);
  void  loadEnvironment(const std::string& hrdImage);
  void  destroy();
  float getIntegral() { return m_integral; }
  float getAverage() { return m_average; }

  inline const VkDescriptorSetLayout getDescLayout() { return m_descSetLayout; }
  inline const VkDescriptorSet       getDescSet() { return m_descSet; }


private:
  VkDevice                 m_device{VK_NULL_HANDLE};
  uint32_t                 m_familyIndex{0};
  nvvk::ResourceAllocator* m_alloc{nullptr};
  nvvk::DebugUtil          m_debug;

  float m_integral{1.f};
  float m_average{1.f};

  // Resources
  nvvk::Texture         m_texHdr;
  nvvk::Buffer          m_accelImpSmpl;
  VkDescriptorPool      m_descPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_descSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet       m_descSet{VK_NULL_HANDLE};


  void createDescriptorSetLayout();
};
