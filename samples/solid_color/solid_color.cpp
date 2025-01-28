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


/*
  This sample is not using any shaders, it is simply copying a color in a texture and displaying that image.
  
  The example `SolidColor` is attached to the nvvk::Application as an engine, and will be called for various 
  events:
    - onAttach : called at when adding the engine
    - onDetach: called when application stops
    - onUIRender: called at each frame, before a frame render begins
    - onUIMenu: called by application to add a menubar

  See nvvk::Application for more details, but in a nutshell, the application is 
   - creates a GLFW window
   - creates Vulkan context: VkInstance, VkDevice, VkPhysicalDevice
   - creates the swapchains (using ImGui code)
   - loops and call onUIRender, onUIMenu, onRender (not used here)
   - commit the frame command buffer and submit the frame

   This example and a few others are using MicroVk. This is creating the Vulkan resource manager, 
   it is the allocator of memory, creation of images and buffers. We are wrapping this in a
   class, since the allocator will be use every everywhere. 


*/


#include <array>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include "backends/imgui_impl_vulkan.h"             // ImGui_ImplVulkan_AddTexture
#include "common/vk_context.hpp"                    // Vulkan context
#include "nvvk/debug_util_vk.hpp"                   // Vulkan debug names
#include "nvvk/images_vk.hpp"                       // Image creation helpers
#include "nvvkhl/alloc_vma.hpp"                     // VMA allocator
#include "nvvkhl/application.hpp"                   // The application framework
#include "nvvkhl/element_benchmark_parameters.hpp"  // Tests and benchmarks
#include "nvvk/extensions_vk.hpp"

class SolidColor : public nvvkhl::IAppElement
{
public:
  SolidColor()           = default;
  ~SolidColor() override = default;

  // Implementation of nvvk::IApplication interface
  void onAttach(nvvkhl::Application* app) override
  {
    m_app = app;

    // Create the Vulkan allocator (VMA)
    m_alloc = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    });  // Allocator
    m_dutil = std::make_unique<nvvk::DebugUtil>(m_app->getDevice());
    createTexture();
  };

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
    destroyResources();
  };


  void onUIRender() override
  {
    // Settings
    ImGui::Begin("Settings");
    if(ImGui::ColorEdit3("Color", m_imageData.data()))
    {
      m_dirty = true;
    }
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
    ImGui::End();

    // Using viewport Window
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    // Display the colored image
    if(m_texture.image != nullptr)
    {
      ImGui::Image(m_descriptorSet, ImGui::GetContentRegionAvail());
    }

    ImGui::End();
    ImGui::PopStyleVar();
  }

  void onRender(VkCommandBuffer cmd) override
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    if(m_dirty)
    {
      setData(cmd);
    }
  }

  void onUIMenu() override
  {
    bool close_app{false};

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
      {
        close_app = true;
      }
      ImGui::EndMenu();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      close_app = true;
    }

    if(close_app)
    {
      m_app->close();
    }
  }


private:
  void createTexture()
  {
    assert(!m_texture.image);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    const VkImageCreateInfo   create_info = nvvk::makeImage2DCreateInfo({1, 1}, VK_FORMAT_R32G32B32A32_SFLOAT);
    const VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_texture = m_alloc->createTexture(cmd, m_imageData.size() * sizeof(float), m_imageData.data(), create_info, sampler_info);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_dutil->setObjectName(m_texture.image, "Image");
    m_dutil->setObjectName(m_texture.descriptor.sampler, "Sampler");
    m_descriptorSet = ImGui_ImplVulkan_AddTexture(m_texture.descriptor.sampler, m_texture.descriptor.imageView,
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  void setData(VkCommandBuffer cmd)
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    assert(m_texture.image);
    const VkOffset3D               offset{0};
    const VkImageSubresourceLayers subresource{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    const VkExtent3D               extent{1, 1, 1};
    nvvk::cmdBarrierImageLayout(cmd, m_texture.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    nvvk::StagingMemoryManager* staging = m_alloc->getStaging();
    staging->cmdToImage(cmd, m_texture.image, offset, extent, subresource, m_imageData.size() * sizeof(float),
                        m_imageData.data());

    nvvk::cmdBarrierImageLayout(cmd, m_texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_dirty = false;
  }

  void destroyResources() { m_alloc->destroy(m_texture); }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_texture.image, {1, 1}, nvh::getExecutablePath().replace_extension(".jpg").string());
  }

  // Local data
  nvvk::Texture        m_texture;
  std::array<float, 4> m_imageData{1, 1, 0, 1};
  VkDescriptorSet      m_descriptorSet{};
  bool                 m_dirty{false};

  std::unique_ptr<nvvkhl::AllocVma> m_alloc;
  std::unique_ptr<nvvk::DebugUtil>  m_dutil;
  nvvkhl::Application*              m_app{nullptr};
};

int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo appInfo;

  nvh::CommandLineParser cli(PROJECT_NAME);
  cli.addArgument({"--headless"}, &appInfo.headless, "Run in headless mode");
  cli.parse(argc, argv);

  VkContextSettings vkSetup;
  if(!appInfo.headless)
  {
    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  // Create Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);  // Loading the Vulkan extension pointers

  // Application setup
  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext->getInstance();
  appInfo.device         = vkContext->getDevice();
  appInfo.physicalDevice = vkContext->getPhysicalDevice();
  appInfo.queues         = vkContext->getQueueInfos();

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appInfo);

  // Create this example
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);
  app->addElement(test);
  app->addElement(std::make_shared<SolidColor>());

  app->run();

  return test->errorCode();
}
