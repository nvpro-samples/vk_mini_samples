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
#include <filesystem>

#include "common/src/vulkan_sample.hpp"
#include "nvvk/profiler_vk.hpp"

//--------------------------------------------------------------------------------------------------
// Simple rasterizer and raytracer of glTF scenes
//
class KtxSample : public VulkanSample
{
public:
  void create(const nvvk::AppBaseVkCreateInfo& info) override;
  void loadScene(const std::string& filename) override;
  void updateUniformBuffer(VkCommandBuffer cmdBuf) override;
  void createTextureImages(VkCommandBuffer cmdBuf, tinygltf::Model& gltfModel, const std::string& filename);
  void destroy() override;
  void raytrace(VkCommandBuffer cmdBuf) override;
  void renderUI() override;
  void prepareFrame() override
  {
    nvvk::AppBaseVk::prepareFrame();
    m_profiler.beginFrame();
  }

  void submitFrame() override
  {
    m_profiler.endFrame();
    nvvk::AppBaseVk::submitFrame();
  }


private:
  FrameInfo m_hostUBO{{}, {}, {}, {}, {}, {{}}, true};
  bool loadCreateImage(const VkCommandBuffer& cmdBuf, const std::filesystem::path& basedir, tinygltf::Image& gltfImage);

  nvvk::ProfilerVK m_profiler;
};
