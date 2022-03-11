/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common/src/vulkan_sample.hpp"
#include "common/src/hdr_env.hpp"
#include "common/src/hdr_env_dome.hpp"
#include "nvvk/memallocator_vma_vk.hpp"


//--------------------------------------------------------------------------------------------------
// Simple rasterizer and raytracer of glTF scenes
//
class HdrSample : public VulkanSample
{
public:
  void createHdr(const std::string& hdrFilename);
  void drawDome(VkCommandBuffer cmdBuf);

  // Override
  void destroy() override;
  void createGraphicPipeline() override;
  void rasterize(VkCommandBuffer cmdBuf) override;
  void recordRendering() override;
  void onResize(int /*w*/, int /*h*/) override;
  void createRtPipeline() override;
  void raytrace(VkCommandBuffer cmdBuf) override;
  void onFileDrop(const char* filename) override;
  void renderUI() override;
  void create(const nvvk::AppBaseVkCreateInfo& info) override;
  void createScene(const std::string& filename) override;

private:
  HdrEnv     m_hdrEnv;
  HdrEnvDome m_hdrDome;

  VmaAllocator                              m_vmaAlloc;
  std::unique_ptr<nvvk::VMAMemoryAllocator> m_vma;  // The memory allocator
};
