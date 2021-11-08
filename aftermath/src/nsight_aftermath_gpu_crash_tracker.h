/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

#include <cassert>
#include <mutex>
#include <string>
#include <iomanip>

#include <vulkan/vulkan.h>
#include "GFSDK_Aftermath.h"
#include "GFSDK_Aftermath_GpuCrashDump.h"
#include "GFSDK_Aftermath_GpuCrashDumpDecoding.h"


//--------------------------------------------------------------------------------------------------
// Implements GPU crash dump tracking using the Nsight Aftermath API.
//
class GpuCrashTracker
{
public:
  GpuCrashTracker() = default;
  ~GpuCrashTracker();

  void Initialize();  // Initialize the GPU crash dump tracker.

private:
  // Callback handlers for GPU crash dumps and related data.
  void OnCrashDump(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize);
  void OnShaderDebugInfo(const void* pShaderDebugInfo, const uint32_t shaderDebugInfoSize);
  void OnDescription(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription);
  void WriteGpuCrashDumpToFile(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize);
  void WriteShaderDebugInformationToFile(GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier,
                                         const void*                               pShaderDebugInfo,
                                         const uint32_t                            shaderDebugInfoSize);

  // Static callback wrappers.
  static void GpuCrashDumpCallback(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize, void* pUserData);
  static void ShaderDebugInfoCallback(const void* pShaderDebugInfo, const uint32_t shaderDebugInfoSize, void* pUserData);
  static void CrashDumpDescriptionCallback(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription, void* pUserData);

  // GPU crash tracker state.
  bool               m_initialized{false};
  mutable std::mutex m_mutex{};  // For thread-safe access of GPU crash tracker state.
};


//--------------------------------------------------------------------------------------------------
// Some std::to_string overloads for some Nsight Aftermath API types.
//
namespace std {
template <typename T>
inline std::string to_hex_string(T n)
{
  std::stringstream stream;
  stream << std::setfill('0') << std::setw(2 * sizeof(T)) << std::hex << n;
  return stream.str();
}

inline std::string to_string(GFSDK_Aftermath_Result result)
{
  return std::string("0x") + to_hex_string(static_cast<uint32_t>(result));
}

inline std::string to_string(const GFSDK_Aftermath_ShaderDebugInfoIdentifier& identifier)
{
  return to_hex_string(identifier.id[0]) + "-" + to_hex_string(identifier.id[1]);
}

inline std::string to_string(const GFSDK_Aftermath_ShaderHash& hash)
{
  return to_hex_string(hash.hash);
}

inline std::string to_string(const GFSDK_Aftermath_ShaderInstructionsHash& hash)
{
  return to_hex_string(hash.hash) + "-" + to_hex_string(hash.hash);
}
}  // namespace std


#define AFTERMATH_CHECK_ERROR(FC)                                                                                      \
  [&]() {                                                                                                              \
    GFSDK_Aftermath_Result _result = FC;                                                                               \
    assert(GFSDK_Aftermath_SUCCEED(_result));                                                                          \
  }()
