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


//--------------------------------------------------------------------------------------------------
// Simple rasterizer and raytracer of glTF scenes
//
class HdrSample : public VulkanSample
{
public:
  // Load HDR and set image
  void createHdr(const std::string& hdrFilename);
  // Draw the HDR dome for raster
  void drawDome(VkCommandBuffer cmdBuf);

  void destroy() override;
  void createGraphicPipeline() override;
  void rasterize(VkCommandBuffer cmdBuf) override;
  void onResize(int /*w*/, int /*h*/) override;
  void createOffscreenRender() override;
  void createRtPipeline() override;
  void raytrace(VkCommandBuffer cmdBuf) override;
  void onFileDrop(const char* filename) override;
  void renderUI() override;
  void create(const nvvk::AppBaseVkCreateInfo& info) override;
  void createScene(const std::string& filename) override;

private:
  //nvvk::ResourceAllocatorVma m_alloc;  // Allocator for buffer, images, acceleration structures

  //VkClearColorValue m_clearColor{0.5f, 0.5f, 0.5f, 1.f};

  HdrEnv     m_hdrEnv;
  HdrEnvDome m_hdrDome;
};
