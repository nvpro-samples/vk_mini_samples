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


//--------------------------------------------------------------------------------------------------
// Simple rasterizer of glTF objects with MSAA
//
class MsaaSample : public VulkanSample
{
public:
  void create(const nvvk::AppBaseVkCreateInfo& info) override;
  void createScene(const std::string& filename) override;
  void createGraphicPipeline() override;
  void freeResources() override;
  void rasterize(VkCommandBuffer cmdBuf) override;
  void recordRendering() override;
  void onResize(int /*w*/, int /*h*/) override;
  void createMsaaImages(VkSampleCountFlagBits samples);
  void renderUI() override;

private:
  // #MSAA
  nvvk::Image           m_msaaColor;
  nvvk::Image           m_msaaDepth;
  VkImageView           m_msaaColorIView{VK_NULL_HANDLE};
  VkImageView           m_msaaDepthIView{VK_NULL_HANDLE};
  VkSampleCountFlagBits m_msaaSamples{VK_SAMPLE_COUNT_4_BIT};
};
