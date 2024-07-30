/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


/*
  This sample is not using any shaders, it is simply showing the GPU usage
  
*/


#include <array>
#include <vulkan/vulkan_core.h>

#include "nvvkhl/application.hpp"
#include "nvvkhl/element_nvml.hpp"
#include "common/vk_context.hpp"

int main(int argc, char** argv)
{
  VkContextSettings vkSetup;
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  vkSetup.apiVersion             = VK_API_VERSION_1_3;
  vkSetup.enableAllFeatures      = false;
  vkSetup.enableValidationLayers = false;
  vkSetup.verbose                = false;

  // Create the Vulkan context
  VkContext vkContext(vkSetup);
  if(!vkContext.isValid())
    std::exit(0);

  // Set how the application should be
  nvvkhl::ApplicationCreateInfo appSetup;
  appSetup.name                  = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appSetup.vSync                 = true;
  appSetup.hasUndockableViewport = false;
  appSetup.width                 = 750;
  appSetup.height                = 400;
  appSetup.instance              = vkContext.getInstance();
  appSetup.device                = vkContext.getDevice();
  appSetup.physicalDevice        = vkContext.getPhysicalDevice();
  appSetup.queues                = vkContext.getQueueInfos();

  // Setting up the layout of the application. Docking the NVML monitor in the center
  appSetup.dockSetup = [](ImGuiID viewportID) { ImGui::DockBuilderDockWindow("NVML Monitor", viewportID); };

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appSetup);

  // The NVML (GPU) monitor
  app->addElement(std::make_shared<nvvkhl::ElementNvml>(true));
  app->run();

  app.reset();
  vkContext.deinit();

  return 0;
}
