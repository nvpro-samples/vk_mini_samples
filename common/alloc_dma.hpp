/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/memorymanagement_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"


//-------------------------------------------------
// Using Device Memory Allocator, similar to VMA
// but using a slightly different approach
class AllocDma : public nvvk::ResourceAllocator
{
public:
  explicit AllocDma(VkDevice device, VkPhysicalDevice physicalDevice) { init(device, physicalDevice); }
  ~AllocDma() override { deinit(); }

private:
  void init(VkDevice device, VkPhysicalDevice physicalDevice)
  {
    m_dma = std::make_unique<nvvk::DeviceMemoryAllocator>();
    m_dma->init(device, physicalDevice, NVVK_DEFAULT_MEMORY_BLOCKSIZE);
    nvvk::ResourceAllocator::init(device, physicalDevice, m_dma.get());
  }
  void deinit()
  {
    releaseStaging();
    m_dma->deinit();
    nvvk::ResourceAllocator::deinit();
  }

  std::unique_ptr<nvvk::DeviceMemoryAllocator> m_dma{};
};
