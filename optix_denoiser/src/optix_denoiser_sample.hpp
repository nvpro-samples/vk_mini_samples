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

// #OPTIX_D
#include "common/src/hdr_env.hpp"
#include "denoiser.hpp"
constexpr int FRAMES_IN_FLIGHT = 3;  // We are using this to change the image to display on the fly

//--------------------------------------------------------------------------------------------------
// Simple rasterizer and raytracer of glTF scenes
//
class OptixDenoiserSample : public VulkanSample
{
public:
  void create(const nvvk::AppBaseVkCreateInfo& info) override;
  void createScene(const std::string& filename) override;
  void createHdr(const std::string& hdrFilename);  // #OPTIX_D
  void renderUI() override;
  void createPostPipeline() override;
  void drawPost(VkCommandBuffer cmdBuf) override;
  void raytrace(VkCommandBuffer cmdBuf) override;
  void freeResources() override;
  void destroy() override;
  // Overriding to create 2x more command buffer per frame
  void createSwapchain(const VkSurfaceKHR& surface,
                       uint32_t            width,
                       uint32_t            height,
                       VkFormat            colorFormat = VK_FORMAT_B8G8R8A8_UNORM,
                       VkFormat            depthFormat = VK_FORMAT_UNDEFINED,
                       bool                vsync       = false) override;

  // #OPTIX_D
  void submitWithTLSemaphore(const VkCommandBuffer& cmdBuf);
  void submitFrame(const VkCommandBuffer& cmdBuf);
  void createGbuffers();
  void denoise();
  void setImageToDisplay();
  bool needToDenoise();
  void copyImagesToCuda(const VkCommandBuffer& cmdBuf);
  void copyCudaImagesToVulkan(const VkCommandBuffer& cmdBuf);

private:
  void updatePostDescriptorSet(const VkDescriptorImageInfo& descriptor) override;
  void createRtPipeline() override;
  void updateRtDescriptorSet() override;
  void onResize(int /*w*/, int /*h*/) override;
  void onFileDrop(const char* filename) override;

  //////////////////////////////////////////////////////////////////////////
  // #OPTIX_D
  std::array<VkDescriptorPool, FRAMES_IN_FLIGHT> m_postDstPool;
  std::array<VkDescriptorSet, FRAMES_IN_FLIGHT>  m_postDstSet;
  uint32_t                                       m_postFrame{0};

  nvvk::Texture m_gAlbedo;
  nvvk::Texture m_gNormal;
  nvvk::Texture m_gDenoised;

  HdrEnv m_hdrEnv;

#ifdef NVP_SUPPORTS_OPTIX7
  DenoiserOptix m_denoiser;
#endif  // NVP_SUPPORTS_OPTIX7

  // Timeline semaphores
  uint64_t m_fenceValue{0};
  bool     m_denoiseApply{true};
  bool     m_denoiseFirstFrame{false};
  int      m_denoiseEveryNFrames{100};
};
