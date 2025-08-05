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


/*----------------------------------------------------------------------------------
  This sample is not using any shaders, it is simply showing the GPU usage 
  ---------------------------------------------------------------------------------*/

#include <nvapp/application.hpp>
#include <nvgpu_monitor/elem_gpu_monitor.hpp>
#include <nvutils/logger.hpp>
#include <nvvk/context.hpp>

int main(int argc, char** argv)
{
  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}},
  };
  nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);

  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Set how the application should be
  nvapp::ApplicationCreateInfo appSetup;
  appSetup.name                  = TARGET_NAME;
  appSetup.vSync                 = true;
  appSetup.hasUndockableViewport = false;
  appSetup.windowSize            = {750, 400};
  appSetup.instance              = vkContext.getInstance();
  appSetup.device                = vkContext.getDevice();
  appSetup.physicalDevice        = vkContext.getPhysicalDevice();
  appSetup.queues                = vkContext.getQueueInfos();

  // Setting up the layout of the application. Docking the NVML monitor in the center
  appSetup.dockSetup = [](ImGuiID viewportID) { ImGui::DockBuilderDockWindow("NVML Monitor", viewportID); };

  // Create the application
  nvapp::Application app;
  app.init(appSetup);

  // The NVML (GPU) monitor
  app.addElement(std::make_shared<nvgpu_monitor::ElementGpuMonitor>(true));
  app.run();

  app.deinit();
  vkContext.deinit();

  return 0;
}
