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

#include <fstream>
#include <sstream>
#include <vector>
#include <iostream>

#include "nsight_aftermath_gpu_crash_tracker.h"
#include "nvh/nvprint.hpp"

//*********************************************************
// GpuCrashTracker implementation
//*********************************************************

GpuCrashTracker::~GpuCrashTracker()
{
  if(m_initialized)
  {
    GFSDK_Aftermath_DisableGpuCrashDumps();
  }
}

// Initialize the GPU Crash Dump Tracker
void GpuCrashTracker::Initialize()
{
  // Enable GPU crash dumps and set up the callbacks for crash dump notifications,
  // shader debug information notifications, and providing additional crash
  // dump description data. Only the crash dump callback is mandatory. The other two
  // callbacks are optional and can be omitted, by passing nullptr, if the corresponding
  // functionality is not used.
  // The DeferDebugInfoCallbacks flag enables caching of shader debug information data
  // in memory. If the flag is set, ShaderDebugInfoCallback will be called only
  // in the event of a crash, right before GpuCrashDumpCallback. If the flag is not set,
  // ShaderDebugInfoCallback will be called for every shader that is compiled.
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_EnableGpuCrashDumps(
      GFSDK_Aftermath_Version_API, GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
      GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks,  // Let the Nsight Aftermath library cache shader debug information.
      GpuCrashDumpCallback,                                              // Register callback for GPU crash dumps.
      ShaderDebugInfoCallback,       // Register callback for shader debug information.
      CrashDumpDescriptionCallback,  // Register callback for GPU crash dump description.
      this));                        // Set the GpuCrashTracker object as user data for the above callbacks.

  m_initialized = true;
}

// Handler for GPU crash dump callbacks from Nsight Aftermath
void GpuCrashTracker::OnCrashDump(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize)
{
  // Make sure only one thread at a time...
  std::lock_guard<std::mutex> lock(m_mutex);

  // Write to file for later in-depth analysis with Nsight Graphics.
  WriteGpuCrashDumpToFile(pGpuCrashDump, gpuCrashDumpSize);
}

// Handler for shader debug information callbacks
void GpuCrashTracker::OnShaderDebugInfo(const void* pShaderDebugInfo, const uint32_t shaderDebugInfoSize)
{
  // Make sure only one thread at a time...
  std::lock_guard<std::mutex> lock(m_mutex);

  // Get shader debug information identifier
  GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier = {};
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetShaderDebugInfoIdentifier(GFSDK_Aftermath_Version_API, pShaderDebugInfo,
                                                                     shaderDebugInfoSize, &identifier));

  // Write to file for later in-depth analysis of crash dumps with Nsight Graphics
  WriteShaderDebugInformationToFile(identifier, pShaderDebugInfo, shaderDebugInfoSize);
}

// Handler for GPU crash dump description callbacks
void GpuCrashTracker::OnDescription(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription)
{
  // Add some basic description about the crash. This is called after the GPU crash happens, but before
  // the actual GPU crash dump callback. The provided data is included in the crash dump and can be
  // retrieved using GFSDK_Aftermath_GpuCrashDump_GetDescription().
  addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName, PROJECT_NAME);
  addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationVersion, "v1.0");
  addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_UserDefined, "This is a GPU crash dump example.");
  addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_UserDefined + 1, "Engine State: Rendering.");
  addDescription(GFSDK_Aftermath_GpuCrashDumpDescriptionKey_UserDefined + 2, "More user-defined information...");
}

// Helper for writing a GPU crash dump to a file
void GpuCrashTracker::WriteGpuCrashDumpToFile(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize)
{
  // Create a GPU crash dump decoder object for the GPU crash dump.
  GFSDK_Aftermath_GpuCrashDump_Decoder decoder = {};
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_CreateDecoder(GFSDK_Aftermath_Version_API, pGpuCrashDump,
                                                                   gpuCrashDumpSize, &decoder));

  // Use the decoder object to read basic information, like application
  // name, PID, etc. from the GPU crash dump.
  GFSDK_Aftermath_GpuCrashDump_BaseInfo baseInfo = {};
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetBaseInfo(decoder, &baseInfo));

  // Use the decoder object to query the application name that was set
  // in the GPU crash dump description.
  uint32_t applicationNameLength = 0;
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetDescriptionSize(decoder, GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName,
                                                                        &applicationNameLength));

  std::vector<char> applicationName(applicationNameLength, '\0');

  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetDescription(decoder, GFSDK_Aftermath_GpuCrashDumpDescriptionKey_ApplicationName,
                                                                    uint32_t(applicationName.size()), applicationName.data()));

  // Create a unique file name for writing the crash dump data to a file.
  // Note: due to an Nsight Aftermath bug (will be fixed in an upcoming
  // driver release) we may see redundant crash dumps. As a workaround,
  // attach a unique count to each generated file name.
  static int        count = 0;
  const std::string baseFileName =
      std::string(applicationName.data()) + "-" + std::to_string(baseInfo.pid) + "-" + std::to_string(++count);

  // Write the the crash dump data to a file using the .nv-gpudmp extension
  // registered with Nsight Graphics.
  const std::string crashDumpFileName = baseFileName + ".nv-gpudmp";
  std::ofstream     dumpFile(crashDumpFileName, std::ios::out | std::ios::binary);
  if(dumpFile)
  {
    dumpFile.write((const char*)pGpuCrashDump, gpuCrashDumpSize);
    dumpFile.close();
  }

  std::cerr << "\n\n**********\nDump file under: " << crashDumpFileName << "\n**********\n";

  // Decode the crash dump to a JSON string.
  // Step 1: Generate the JSON and get the size.
  uint32_t jsonSize = 0;
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GenerateJSON(
      decoder, GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO, GFSDK_Aftermath_GpuCrashDumpFormatterFlags_NONE,
      nullptr /*ShaderDebugInfoLookupCallback*/, nullptr /*ShaderLookupCallback*/, nullptr,
      nullptr /*ShaderSourceDebugInfoLookupCallback*/, this, &jsonSize));

  // Step 2: Allocate a buffer and fetch the generated JSON.
  std::vector<char> json(jsonSize);
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetJSON(decoder, uint32_t(json.size()), json.data()));

  // Write the the crash dump data as JSON to a file.
  const std::string jsonFileName = crashDumpFileName + ".json";
  std::ofstream     jsonFile(jsonFileName, std::ios::out | std::ios::binary);
  if(jsonFile)
  {
    jsonFile.write(json.data(), json.size());
    jsonFile.close();
  }

  // Destroy the GPU crash dump decoder object.
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(decoder));
}

// Helper for writing shader debug information to a file
void GpuCrashTracker::WriteShaderDebugInformationToFile(GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier,
                                                        const void*                               pShaderDebugInfo,
                                                        const uint32_t                            shaderDebugInfoSize)
{
  // Create a unique file name.
  const std::string filePath = "shader-" + std::to_string(identifier) + ".nvdbg";

  std::ofstream f(filePath, std::ios::out | std::ios::binary);
  if(f)
  {
    f.write((const char*)pShaderDebugInfo, shaderDebugInfoSize);
  }
}

// Static callback wrapper for OnCrashDump
void GpuCrashTracker::GpuCrashDumpCallback(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize, void* pUserData)
{
  GpuCrashTracker* pGpuCrashTracker = reinterpret_cast<GpuCrashTracker*>(pUserData);
  pGpuCrashTracker->OnCrashDump(pGpuCrashDump, gpuCrashDumpSize);
}

// Static callback wrapper for OnShaderDebugInfo
void GpuCrashTracker::ShaderDebugInfoCallback(const void* pShaderDebugInfo, const uint32_t shaderDebugInfoSize, void* pUserData)
{
  GpuCrashTracker* pGpuCrashTracker = reinterpret_cast<GpuCrashTracker*>(pUserData);
  pGpuCrashTracker->OnShaderDebugInfo(pShaderDebugInfo, shaderDebugInfoSize);
}

// Static callback wrapper for OnDescription
void GpuCrashTracker::CrashDumpDescriptionCallback(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription, void* pUserData)
{
  GpuCrashTracker* pGpuCrashTracker = reinterpret_cast<GpuCrashTracker*>(pUserData);
  pGpuCrashTracker->OnDescription(addDescription);
}
