/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <vector>
#include <string>
#include <filesystem>
#include "nvpsystem.hpp"
#include "vulkan/vulkan_core.h"


inline std::vector<std::string> getMediaDirs()
{
  return {
      NVPSystem::exePath() + std::string("../media"),  //
      NVPSystem::exePath() + std::string("/media"),    //
      NVPSystem::exePath()                             //
  };
}

// Find the full path of the shader file
inline std::string getFilePath(const std::string& file, const std::vector<std::string>& paths)
{
  namespace fs = std::filesystem;

  if(fs::exists(fs::path(file)))
  {
    return file;
  }

  std::string directoryPath;
  for(const auto& dir : paths)
  {
    fs::path p = fs::path(dir) / fs::path(file);
    if(fs::exists(p))
    {
      directoryPath = p.string();
      break;
    }
  }
  return directoryPath;
}


inline void memoryBarrier(VkCommandBuffer cmd)
{
  VkMemoryBarrier mb{
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
  };
  VkPipelineStageFlags srcDstStage{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
  vkCmdPipelineBarrier(cmd, srcDstStage, srcDstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
}