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


// ImGui - standalone example application for Glfw + Vulkan, using programmable
// pipeline If you are new to ImGui, see examples/README.txt and documentation
// at the top of imgui.cpp.

#include <array>

#include "backends/imgui_impl_glfw.h"
#include "imgui.h"

#include "vulkan_sample.hpp"
#include "imgui/imgui_camera_widget.h"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvpsystem.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvh/inputparser.h"
#include "nvvk/gizmos_vk.hpp"


//////////////////////////////////////////////////////////////////////////
#define UNUSED(x) (void)(x)
//////////////////////////////////////////////////////////////////////////

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;


// GLFW Callback functions
static void onErrorCallback(int error, const char* description)
{
  fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
static int const SAMPLE_WIDTH  = 1280;
static int const SAMPLE_HEIGHT = 720;


//--------------------------------------------------------------------------------------------------
// Application Entry
//
int main(int argc, char** argv)
{
  InputParser parser(argc, argv);
  std::string sceneFile = parser.getString("-f", "media/cornellBox.gltf");
  std::string hdrFile   = parser.getString("-h", "media/std_env.hdr");


  // Setup GLFW window
  glfwSetErrorCallback(onErrorCallback);
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME, nullptr, nullptr);

  // Setup camera
  CameraManip.setWindowSize(SAMPLE_WIDTH, SAMPLE_HEIGHT);
  CameraManip.setLookat(nvmath::vec3f(0, 0, 15), nvmath::vec3f(0, 0, 0), nvmath::vec3f(0, 1, 0));

  // Setup Vulkan
  if(!glfwVulkanSupported())
  {
    printf("GLFW: Vulkan Not Supported\n");
    return 1;
  }

  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  // Search path for shaders and other media
  defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      NVPSystem::exePath() + PROJECT_RELDIRECTORY "..",
      std::string(PROJECT_NAME),
  };

  // Find the files
  sceneFile = nvh::findFile(sceneFile, defaultSearchPaths, true);
  hdrFile   = nvh::findFile(hdrFile, defaultSearchPaths, true);


  // Vulkan required extensions
  assert(glfwVulkanSupported() == 1);
  uint32_t count{0};
  auto     reqExtensions = glfwGetRequiredInstanceExtensions(&count);

  // Requesting Vulkan extensions and layers
  nvvk::ContextCreateInfo contextInfo;
  contextInfo.setVersion(1, 2);                       // Using Vulkan 1.2
  for(uint32_t ext_id = 0; ext_id < count; ext_id++)  // Adding required extensions (surface, win32, linux, ..)
    contextInfo.addInstanceExtension(reqExtensions[ext_id]);
  contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, true);  // Allow debug names
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);            // Enabling ability to present rendering

  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);  // To use vkCmdTraceRaysKHR
  VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);  // Used for picking
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  contextInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);


  // Semaphores - interop Vulkan/Cuda
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME);
#ifdef WIN32
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME);
#else
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
#endif

  // Synchronization (mix of timeline and binary semaphores)
  contextInfo.addDeviceExtension(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, false);
  VkPhysicalDeviceSynchronization2FeaturesKHR syncFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR};
  contextInfo.addDeviceExtension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, false, &syncFeature);

  // Buffer - interop
  contextInfo.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);


  // Creating Vulkan base application
  nvvk::Context vkctx{};
  vkctx.initInstance(contextInfo);
  // Find all compatible devices
  auto compatibleDevices = vkctx.getCompatibleDevices(contextInfo);
  assert(!compatibleDevices.empty());
  // Use a compatible device
  vkctx.initDevice(compatibleDevices[0], contextInfo);

  // Create example
  VulkanSample vkSample;

  // Window need to be opened to get the surface on which to draw
  const VkSurfaceKHR surface = vkSample.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  // Creation of the application
  nvvk::AppBaseVkCreateInfo info;
  info.instance       = vkctx.m_instance;
  info.device         = vkctx.m_device;
  info.physicalDevice = vkctx.m_physicalDevice;
  info.size           = {SAMPLE_WIDTH, SAMPLE_HEIGHT};
  info.surface        = surface;
  info.window         = window;
  info.queueIndices.push_back(vkctx.m_queueGCT.familyIndex);
  vkSample.create(info);
  // Loading and creating the scene
  vkSample.createHdr(hdrFile);
  vkSample.createScene(sceneFile);

  ImGui_ImplGlfw_InitForVulkan(window, true);


  nvvk::AxisVK vkAxis;
  vkAxis.init(vkctx.m_device, vkSample.getRenderPass());

  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(vkSample.isMinimized())
      continue;

    // Start the Dear ImGui frame
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();


    // UI and info
    vkSample.titleBar();
    vkSample.renderUI();

    // Start rendering the scene
    vkSample.prepareFrame();

    // Start command buffer of this frame
    auto curFrame = vkSample.getCurFrame();
    //const VkCommandBuffer& cmdBuf   = vkSample.getCommandBuffers()[curFrame];

    // Two command buffer in a frame, before and after denoiser
    const VkCommandBuffer& cmdBuf1 = vkSample.getCommandBuffers()[curFrame * 2 + 0];
    const VkCommandBuffer& cmdBuf2 = vkSample.getCommandBuffers()[curFrame * 2 + 1];

    vkSample.setImageToDisplay();

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf1, &beginInfo);

    // Updating frame information
    vkSample.updateUniformBuffer(cmdBuf1);

    // Clearing screen
    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color        = vkSample.clearColor();
    clearValues[1].depthStencil = {1.0f, 0};

    // Offscreen Rendering Scene
    {
      if(vkSample.m_renderMode == VulkanSample::RenderMode::eRayTracer)
      {
        // Ray tracing don't need any rendering pass
        vkSample.raytrace(cmdBuf1);
      }
      else
      {
        // Raster
        VkRenderPassBeginInfo offscreenRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        offscreenRenderPassBeginInfo.clearValueCount = 2;
        offscreenRenderPassBeginInfo.pClearValues    = clearValues.data();
        offscreenRenderPassBeginInfo.renderPass      = vkSample.offscreenRenderPass();
        offscreenRenderPassBeginInfo.framebuffer     = vkSample.offscreenFramebuffer();
        offscreenRenderPassBeginInfo.renderArea      = {{0, 0}, vkSample.getSize()};

        vkCmdBeginRenderPass(cmdBuf1, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
        vkSample.rasterize(cmdBuf1);
        vkCmdEndRenderPass(cmdBuf1);
      }
    }

    vkSample.copyImagesToCuda(cmdBuf1);

    vkEndCommandBuffer(cmdBuf1);
    vkSample.submitWithTLSemaphore(cmdBuf1);

    // ---- Cuda part --
    vkSample.denoise();
    // ----

    // SECOND PART
    vkBeginCommandBuffer(cmdBuf2, &beginInfo);


    vkSample.copyCudaImagesToVulkan(cmdBuf2);

    // 2nd rendering pass: tone mapper, UI
    {
      VkRenderPassBeginInfo postRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
      postRenderPassBeginInfo.clearValueCount = 2;
      postRenderPassBeginInfo.pClearValues    = clearValues.data();
      postRenderPassBeginInfo.renderPass      = vkSample.getRenderPass();
      postRenderPassBeginInfo.framebuffer     = vkSample.getFramebuffers()[curFrame];
      postRenderPassBeginInfo.renderArea      = {{0, 0}, vkSample.getSize()};

      // Rendering to the swapchain framebuffer the rendered image and apply a tonemapper
      vkCmdBeginRenderPass(cmdBuf2, &postRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
      vkSample.drawPost(cmdBuf2);

      // Rendering UI
      ImGui::Render();
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf2);

      // Display axis in the lower left corner.
      vkAxis.display(cmdBuf2, CameraManip.getMatrix(), vkSample.getSize());

      vkCmdEndRenderPass(cmdBuf2);
    }

    // Submit for display
    vkEndCommandBuffer(cmdBuf2);
    vkSample.submitFrame(cmdBuf2);
  }

  // Cleanup
  vkDeviceWaitIdle(vkSample.getDevice());

  vkSample.destroy();
  vkAxis.deinit();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
