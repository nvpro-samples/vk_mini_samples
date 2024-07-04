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

int main(int argc, char** argv)
{
  nvvk::ContextCreateInfo vkSetup;  // Vulkan creation context information (see nvvk::Context)
  vkSetup.setVersion(1, 3);
  vkSetup.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);

  nvvk::Context vkContext;
  vkContext.init(vkSetup);

  nvvkhl::ApplicationCreateInfo spec;
  spec.name                  = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  spec.vSync                 = true;
  spec.hasUndockableViewport = false;
  spec.width                 = 750;
  spec.height                = 400;
  spec.instance              = vkContext.m_instance;
  spec.device                = vkContext.m_device;
  spec.physicalDevice        = vkContext.m_physicalDevice;
  spec.queues                = {vkContext.m_queueGCT};

  // Setting up the layout of the application. Docking the NVML monitor in the center
  spec.dockSetup = [](ImGuiID viewportID) { ImGui::DockBuilderDockWindow("NVML Monitor", viewportID); };

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // The NVML (GPU) monitor
  app->addElement(std::make_shared<nvvkhl::ElementNvml>(true));
  app->run();

  app.reset();
  vkContext.deinit();

  return 0;
}
