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


/** @DOC_START

This sample demonstrate how to use compute shader in a separate thread to render an image. 
The image is then copied to the main thread for display. The compute shader is rendering 
as fast as possible, while the main thread is displaying the image at the screen refresh rate.

Note: The amount of compute frame it can do per display iteration depends on the GPU. On laptops, 
      and integrated GPUs, the difference is minimal. On high-end GPUs, the difference can be significant.

@DOC_END */

#include <thread>
#include <condition_variable>
#include <iostream>


#include "common/vk_context.hpp"                    // Our Vulkan context
#include "nvvk/descriptorsets_vk.hpp"               // Descriptor set helper
#include "nvvkhl/application.hpp"                   // For Application and IAppElememt
#include "nvvkhl/gbuffer.hpp"                       // G-Buffer helper
#include "nvvkhl/shaders/dh_comp.h"                 // Workgroup size and count
#include "nvvkhl/element_benchmark_parameters.hpp"  // For testing
#include "nvvk/commands_vk.hpp"                     // For command pool
#include "nvvk/extensions_vk.hpp"                   // For vkCreateShadersEXT

#ifdef NVP_SUPPORTS_NVAPI
#include "common/nvapi_manager.hpp"
#endif

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#define SHOW_MENU 1  // Enabling the standard Window menu.
constexpr auto g_defaultWindowSize = VkExtent2D{1024, 1024};


// Shader source code, compiled from CMake
#if USE_HLSL
#include "_autogen/shader_computeMain.spirv.h"
const auto& comp_shd = std::vector<uint8_t>{std::begin(shader_computeMain), std::end(shader_computeMain)};
#elif USE_SLANG
#include "_autogen/shader_slang.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(shaderSlang), std::end(shaderSlang)};
#else
#include "_autogen/shader.comp.glsl.h"  // Generated compiled shader
const auto& comp_shd = std::vector<uint32_t>{std::begin(shader_comp_glsl), std::end(shader_comp_glsl)};
#endif

template <typename T>  // Return memory usage size
size_t getShaderSize(const std::vector<T>& vec)
{
  using baseType = typename std::remove_reference<T>::type;
  return sizeof(baseType) * vec.size();
}

DH::PushConstant g_pushC = {.zoom = 1.5f, .iter = 2};

class MultiThreadedSample : public nvvkhl::IAppElement
{
public:
  MultiThreadedSample()           = default;
  ~MultiThreadedSample() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app   = app;
    m_alloc = std::make_unique<nvvk::ResourceAllocatorDma>(app->getDevice(), app->getPhysicalDevice());
    m_dset  = std::make_unique<nvvk::DescriptorSetContainer>(m_app->getDevice());
    createShaderObjectAndLayout();

    m_gCompBuffers =
        std::make_unique<nvvkhl::GBuffer>(m_app->getDevice(), m_alloc.get(), g_defaultWindowSize, VK_FORMAT_R8G8B8A8_UNORM);

    // Create a thread to render the compute shader
    startRenderThread();
  }

  void onDetach() override
  {
    stopRenderThread();

    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
    for(auto shader : m_shaders)
      vkDestroyShaderEXT(m_app->getDevice(), shader, NULL);
  }

  // Displaying the rendered image and some text
  void onUIRender() override
  {
    static float stats = 0;

    ImGui::Begin("Viewport");
    ImVec2 pos = ImGui::GetCursorPos();  // Remember position to put back text
    ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());
    ImGui::SetCursorPos(pos);
    m_frameCounter++;
    if(m_frameCounter > 30)
    {
      stats           = m_threadCounter / float(m_frameCounter);
      m_threadCounter = 0;
      m_frameCounter  = 0;
    }
    ImGui::Text("Number of compute per frame: %f", stats);
    ImGui::Text("Framerate: %f FPS / %f IPS", ImGui::GetIO().Framerate, stats * ImGui::GetIO().Framerate);
    ImGui::End();
  }

  // Copying the rendered image to the main thread
  void onRender(VkCommandBuffer cmd) override
  {
    VkImageSubresourceLayers srcSubresource{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1};

    VkImageCopy imgCopy    = {};
    imgCopy.srcSubresource = srcSubresource;
    imgCopy.dstSubresource = srcSubresource;
    imgCopy.extent         = {std::min(m_gBuffers->getSize().width, m_gCompBuffers->getSize().width),
                              std::min(m_gBuffers->getSize().height, m_gCompBuffers->getSize().height), 1};
    vkCmdCopyImage(cmd, m_gCompBuffers->getColorImage(), VK_IMAGE_LAYOUT_GENERAL, m_gBuffers->getColorImage(),
                   VK_IMAGE_LAYOUT_GENERAL, 1, &imgCopy);
  }

  // Utility to start and stop the render thread
  void startRenderThread()
  {
    if(m_isRunning.load())
    {
      std::cerr << "Thread already running.\n";
      return;
    }

    m_shouldRun.store(true);
    m_workerThread = std::thread(&MultiThreadedSample::renderThread, this);
  }

  void stopRenderThread()
  {
    if(!m_isRunning.load())
    {
      std::cerr << "Thread is not running.\n";
      return;
    }

    m_shouldRun.store(false);
    {
      std::unique_lock lock(m_mutex);
      m_cv.wait(lock, [this]() { return !m_isRunning.load(); });
    }

    if(m_workerThread.joinable())
    {
      m_workerThread.join();
    }
  }

  // The render thread: rendering the compute shader in a infinite loop
  void renderThread()
  {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_isRunning.store(true);
    }

    constexpr int32_t computeQueueIndex = 1;
    nvvk::CommandPool cmdPool(m_app->getDevice(), m_app->getQueue(computeQueueIndex).familyIndex,
                              VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_app->getQueue(computeQueueIndex).queue);
    while(m_shouldRun.load())
    {
      auto       cmd          = cmdPool.createCommandBuffer();
      VkExtent2D group_counts = getGroupCounts(m_gCompBuffers->getSize());

      // Wait for the frame to be consumed
      const VkDescriptorImageInfo       in_desc = m_gCompBuffers->getDescriptorImageInfo();
      std::vector<VkWriteDescriptorSet> writes;
      writes.push_back(m_dset->makeWrite(0, 0, &in_desc));
      vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dset->getPipeLayout(), 0,
                                static_cast<uint32_t>(writes.size()), writes.data());

      // Bind compute shader
      const VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
      vkCmdBindShadersEXT(cmd, 1, stages, m_shaders.data());

      // Pushing constants
      g_pushC.time += 0.001f;  //static_cast<float>(ImGui::GetTime());
      vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &g_pushC);

      // Dispatch compute shader
      vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);

      cmdPool.submitAndWait(cmd);
      vkResetCommandPool(m_app->getDevice(), cmdPool.getCommandPool(), 0);
      m_threadCounter++;
    };

    cmdPool.deinit();

    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_isRunning.store(false);
      m_cv.notify_all();
    }
  }


  // Called if showMenu is true
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
    bool v_sync = m_app->isVsync();
    if(ImGui::BeginMenu("View"))
    {
      ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &v_sync);
      ImGui::EndMenu();
    }

    if(!m_isRunning.load())
    {
      // Stopped by v-sync change, must be started again, but a frame later
      // because the v-sync change will invoke vkDeviceWaitIdle and cannot be done
      // while the render thread is running.
      startRenderThread();
    }

    if(m_app->isVsync() != v_sync)
    {
      stopRenderThread();
      m_app->setVsync(v_sync);
    }
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    // Re-creating the G-Buffer (RGBA8) when the viewport size change
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_app->getDevice(), m_alloc.get(), VkExtent2D{width, height}, VK_FORMAT_R8G8B8A8_UNORM);
  }

  //-------------------------------------------------------------------------------------------------
  // Creating the pipeline layout and shader object
  void createShaderObjectAndLayout()
  {
    VkPushConstantRange push_constant_ranges = {.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(DH::PushConstant)};

    // Create the layout used by the shader
    m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_dset->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
    m_dset->initPipeLayout(1, &push_constant_ranges);

    // Compute shader description
    std::vector<VkShaderCreateInfoEXT> shaderCreateInfos;
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT{
        .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .pNext                  = NULL,
        .flags                  = VK_SHADER_CREATE_DISPATCH_BASE_BIT_EXT,
        .stage                  = VK_SHADER_STAGE_COMPUTE_BIT,
        .nextStage              = 0,
        .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .codeSize               = getShaderSize(comp_shd),
        .pCode                  = comp_shd.data(),
        .pName                  = USE_GLSL ? "main" : "computeMain",
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_dset->getLayout(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant_ranges,
        .pSpecializationInfo    = NULL,
    });
    // Create the shader
    NVVK_CHECK(vkCreateShadersEXT(m_app->getDevice(), 1, shaderCreateInfos.data(), NULL, m_shaders.data()));
  }

private:
  nvvkhl::Application*                          m_app = {nullptr};
  std::unique_ptr<nvvk::ResourceAllocatorDma>   m_alloc{};
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset{};
  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffers{};
  std::unique_ptr<nvvkhl::GBuffer>              m_gCompBuffers{};
  std::array<VkShaderEXT, 1>                    m_shaders       = {};
  int                                           m_threadCounter = 0;
  int                                           m_frameCounter  = 0;

  std::thread             m_workerThread{};
  std::atomic<bool>       m_shouldRun{false};
  std::atomic<bool>       m_isRunning{false};
  std::mutex              m_mutex{};
  std::condition_variable m_cv{};
};

int main(int argc, char** argv)
{
#ifdef NVP_SUPPORTS_NVAPI
  NVAPIManager nvapiManager;
  nvapiManager.init();
  nvapiManager.pushSetting({
      .version         = NVDRS_SETTING_VER,
      .settingId       = OGL_CPL_PREFER_DXPRESENT_ID,
      .settingType     = NVDRS_DWORD_TYPE,
      .u32CurrentValue = OGL_CPL_PREFER_DXPRESENT_PREFER_DISABLED,
  });
#endif

  // Required extra extensions
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  VkContextSettings vkSetup;
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature});
  vkSetup.deviceExtensions.push_back({VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
  vkSetup.queues.push_back(VK_QUEUE_COMPUTE_BIT);  // Adding an extra compute queue
  // Create Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  // Loading the Vulkan extension pointers
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);


  nvvkhl::ApplicationCreateInfo appSetup;
  appSetup.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appSetup.useMenu        = SHOW_MENU ? true : false;
  appSetup.width          = g_defaultWindowSize.width;
  appSetup.height         = g_defaultWindowSize.height;
  appSetup.instance       = vkContext->getInstance();
  appSetup.device         = vkContext->getDevice();
  appSetup.physicalDevice = vkContext->getPhysicalDevice();
  appSetup.queues         = vkContext->getQueueInfos();

  auto app  = std::make_unique<nvvkhl::Application>(appSetup);                   // Create the application
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);  // Create the test framework
  app->addElement(test);                                                         // Add the test element (--test ...)
  app->addElement(std::make_shared<MultiThreadedSample>());                      // Add our sample to the application

  glfwSetWindowAttrib(app->getWindowHandle(), GLFW_RESIZABLE, GLFW_FALSE);


  app->run();  // Loop infinitely, and call IAppElement virtual functions at each frame

#ifdef NVP_SUPPORTS_NVAPI
  nvapiManager.popSettings();
  nvapiManager.deinit();
#endif

  app.reset();
  vkContext.reset();

  return test->errorCode();
}
