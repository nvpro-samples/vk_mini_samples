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
// Simple rasterizer of glTF objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class InheritedSample : public VulkanSample
{
public:
  void create(const nvvk::AppBaseVkCreateInfo& info) override;
  void createScene(const std::string& filename) override;
  void rasterize(VkCommandBuffer cmdBuff, VkRect2D renderArea);
  void recordRendering() override;
  void onResize(int /*w*/, int /*h*/) override;
  void renderUI() override;
  void updateUniformBuffer(VkCommandBuffer cmdBuf, const VkExtent2D& size);
  void fourViews(VkCommandBuffer cmdBuf);


  VkPhysicalDeviceInheritedViewportScissorFeaturesNV m_inheritedViewport{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INHERITED_VIEWPORT_SCISSOR_FEATURES_NV};
  VkViewport    m_viewportDepth{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};  // Only care about minDepth and maxDepth
  nvmath::vec2f m_viewCenter{0.3f, 0.3f};
};
