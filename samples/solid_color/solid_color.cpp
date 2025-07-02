/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#define VMA_IMPLEMENTATION

#include <volk.h>
#include <span>

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_vulkan.h>

#include <nvapp/application.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>

#define SHOW_MENU 1      // Enabling the standard Window menu.
#define SHOW_SETTINGS 1  // Show the setting panel
#define ANIMATE 1        // Modify the image at each frame

class EmptyElement : public nvapp::IAppElement
{
public:
  EmptyElement()           = default;
  ~EmptyElement() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app                                = app;
    VmaAllocatorCreateInfo allocatorInfo = {
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    };
    m_alloc.init(allocatorInfo);

    m_stagingUploader.init(&m_alloc, true);

    m_samplerPool.init(app->getDevice());

    // Create a 1x1 Vulkan 2D texture
    VkImageCreateInfo imageInfo = DEFAULT_VkImageCreateInfo;
    imageInfo.format            = VK_FORMAT_R32G32B32A32_SFLOAT;
    imageInfo.extent            = {1, 1, 1};
    imageInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT;

    std::array<float, 4> imageData = {0.46F, 0.72F, 0, 1};  // NVIDIA green

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    NVVK_CHECK(m_alloc.createImage(m_image, imageInfo, DEFAULT_VkImageViewCreateInfo));
    NVVK_CHECK(m_stagingUploader.appendImage(m_image, std::span<float>(imageData.data(), imageData.size()),
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
    NVVK_DBG_NAME(m_image.image);
    NVVK_DBG_NAME(m_image.descriptor.imageView);

    m_stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Default sampler for the texture
    NVVK_CHECK(m_samplerPool.acquireSampler(m_image.descriptor.sampler));

    // Add image to ImGui, for display
    m_targetImage = ImGui_ImplVulkan_AddTexture(m_image.descriptor.sampler, m_image.descriptor.imageView,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    NVVK_CHECK(m_alloc.createBuffer(m_stagingBuffer, std::span<float>(imageData).size_bytes(),
                                    VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT));
    NVVK_DBG_NAME(m_stagingBuffer.buffer);
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
    ImGui_ImplVulkan_RemoveTexture(m_targetImage);
    m_alloc.destroyImage(m_image);
    m_alloc.destroyBuffer(m_stagingBuffer);
    m_stagingUploader.deinit();
    m_alloc.deinit();
    m_samplerPool.deinit();
  }

  void onUIRender() override
  {
#if SHOW_SETTINGS
    // [optional] convenient setting panel
    ImGui::Begin("Settings");
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
    ImGui::End();
#endif

    // Rendered image displayed fully in 'Viewport' window
    ImGui::Begin("Viewport");
    ImGui::Image(ImTextureID(m_targetImage), ImGui::GetContentRegionAvail());
    ImGui::End();
  }

  void onRender(VkCommandBuffer cmd)
  {
#if ANIMATE
    ImVec4                                  rgb = {0, 0, 0, 1};
    const std::array<VkBufferImageCopy2, 1> copyRegion{{{
        .sType            = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
        .imageExtent      = m_image.extent,
    }}};  // Copy the whole image

    ImGui::ColorConvertHSVtoRGB((float)ImGui::GetTime() * 0.25f, 1, 1, rgb.x, rgb.y, rgb.z);
    nvvk::cmdImageMemoryBarrier(cmd, {m_image.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL});
    std::memcpy(m_stagingBuffer.mapping, &rgb, sizeof(ImVec4));  // Update the staging buffer (single pixel)

    VkCopyBufferToImageInfo2 copyBufferToImageInfo{
        .sType          = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
        .srcBuffer      = m_stagingBuffer.buffer,
        .dstImage       = m_image.image,
        .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .regionCount    = uint32_t(copyRegion.size()),
        .pRegions       = copyRegion.data(),
    };

    vkCmdCopyBufferToImage2(cmd, &copyBufferToImageInfo);  // Copy the staging buffer to the image
    nvvk::cmdImageMemoryBarrier(cmd, {m_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
#endif
  }

  // [optional] Called if showMenu is true
  void onUIMenu() override
  {
    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
        m_app->close();
      ImGui::EndMenu();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
      m_app->close();
  }

private:
  nvapp::Application*     m_app             = {};
  nvvk::Image             m_image           = {};
  nvvk::ResourceAllocator m_alloc           = {};
  nvvk::StagingUploader   m_stagingUploader = {};
  nvvk::Buffer            m_stagingBuffer   = {};
  nvvk::SamplerPool       m_samplerPool     = {};
  VkDescriptorSet         m_targetImage     = {};
};


int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  nvvk::ContextInitInfo vkSetup{.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME}};
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  appInfo.name           = "The Empty Example";
  appInfo.useMenu        = SHOW_MENU ? true : false;
  appInfo.instance       = vkContext.getInstance();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.device         = vkContext.getDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);
  app.addElement(std::make_shared<EmptyElement>());
  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
