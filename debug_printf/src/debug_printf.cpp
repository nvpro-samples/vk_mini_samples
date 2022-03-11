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


#include "debug_printf.hpp"

//--------------------------------------------------------------------------------------------------
// Override to add mouse click information
//
void DebugPrintf::updateUniformBuffer(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  CameraManip.updateAnim();

  // Prepare new UBO contents on host.
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  auto&       clip        = CameraManip.getClipPlanes();

  m_frameInfo.view       = CameraManip.getMatrix();
  m_frameInfo.proj       = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, clip.x, clip.y);
  m_frameInfo.viewInv    = nvmath::invert(m_frameInfo.view);
  m_frameInfo.projInv    = nvmath::invert(m_frameInfo.proj);
  m_frameInfo.light[0]   = m_lights[0];
  m_frameInfo.light[1]   = m_lights[1];
  m_frameInfo.clearColor = m_clearColor.float32;
  m_frameInfo.coord      = nvmath::vec2f(-1, -1);

  if(ImGui::IsMouseClicked(ImGuiMouseButton_Left))
  {
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);
    m_frameInfo.coord = nvmath::vec2f(x, y);
  }

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_frameInfoBuf.buffer, 0, sizeof(FrameInfo), &m_frameInfo);
}
