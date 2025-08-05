/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include <nvutils/file_operations.hpp>
#include <vulkan/vulkan_core.h>

namespace nvsamples {

inline static std::vector<std::filesystem::path> getResourcesDirs()
{
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  return {
      std::filesystem::absolute(exePath / TARGET_EXE_TO_ROOT_DIRECTORY / "resources"),
      std::filesystem::absolute(exePath / "resources")  //
  };
}

inline static std::vector<std::filesystem::path> getShaderDirs()
{
  std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
  return {
      std::filesystem::absolute(exePath / TARGET_EXE_TO_SOURCE_DIRECTORY / "shaders"),
      std::filesystem::absolute(exePath / TARGET_EXE_TO_NVSHADERS_DIRECTORY),
      std::filesystem::absolute(exePath / TARGET_EXE_TO_ROOT_DIRECTORY / "common"),
      std::filesystem::absolute(NVSHADERS_DIR),
      std::filesystem::absolute(exePath / TARGET_NAME "_files" / "shaders"),
      std::filesystem::absolute(exePath),
  };
}

inline static VkShaderModuleCreateInfo getShaderModuleCreateInfo(const std::span<const uint32_t>& spirv)
{
  return VkShaderModuleCreateInfo{
      .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = spirv.size_bytes(),
      .pCode    = spirv.data(),
  };
}

}  // namespace nvsamples
