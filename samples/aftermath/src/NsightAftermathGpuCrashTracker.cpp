//*********************************************************
//
// Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//*********************************************************

#ifdef USE_NSIGHT_AFTERMATH

#include <fstream>
#include <iomanip>
#include <string>
#include <array>

#include "NsightAftermathGpuCrashTracker.h"

//*********************************************************
// GpuCrashTracker implementation
//*********************************************************

GpuCrashTracker::GpuCrashTracker(const MarkerMap& markerMap)
    : m_initialized(false)
    , m_markerMap(markerMap)
{
}

GpuCrashTracker::~GpuCrashTracker()
{
  // If initialized, disable GPU crash dumps
  if(m_initialized)
  {
    GFSDK_Aftermath_DisableGpuCrashDumps();
  }
}

// Initialize the GPU Crash Dump Tracker
void GpuCrashTracker::initialize()
{
  // Enable GPU crash dumps and set up the callbacks for crash dump notifications,
  // shader debug information notifications, and providing additional crash
  // dump description data.Only the crash dump callback is mandatory. The other two
  // callbacks are optional and can be omitted, by passing nullptr, if the corresponding
  // functionality is not used.
  // The DeferDebugInfoCallbacks flag enables caching of shader debug information data
  // in memory. If the flag is set, ShaderDebugInfoCallback will be called only
  // in the event of a crash, right before GpuCrashDumpCallback. If the flag is not set,
  // ShaderDebugInfoCallback will be called for every shader that is compiled.
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_EnableGpuCrashDumps(
      GFSDK_Aftermath_Version_API, GFSDK_Aftermath_GpuCrashDumpWatchedApiFlags_Vulkan,
      GFSDK_Aftermath_GpuCrashDumpFeatureFlags_DeferDebugInfoCallbacks,  // Let the Nsight Aftermath library cache shader debug information.
      gpuCrashDumpCallback,                                              // Register callback for GPU crash dumps.
      shaderDebugInfoCallback,       // Register callback for shader debug information.
      crashDumpDescriptionCallback,  // Register callback for GPU crash dump description.
      resolveMarkerCallback,         // Register callback for resolving application-managed markers.
      this));                        // Set the GpuCrashTracker object as user data for the above callbacks.

  m_initialized = true;
}

// Handler for GPU crash dump callbacks from Nsight Aftermath
void GpuCrashTracker::onCrashDump(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize)
{
  // Make sure only one thread at a time...
  std::lock_guard<std::mutex> lock(m_mutex);

  // Write to file for later in-depth analysis with Nsight Graphics.
  writeGpuCrashDumpToFile(pGpuCrashDump, gpuCrashDumpSize);
}

// Handler for shader debug information callbacks
void GpuCrashTracker::onShaderDebugInfo(const void* pShaderDebugInfo, const uint32_t shaderDebugInfoSize)
{
  // Make sure only one thread at a time...
  std::lock_guard<std::mutex> lock(m_mutex);

  // Get shader debug information identifier
  GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier = {};
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetShaderDebugInfoIdentifier(GFSDK_Aftermath_Version_API, pShaderDebugInfo,
                                                                     shaderDebugInfoSize, &identifier));

  // Store information for decoding of GPU crash dumps with shader address mapping
  // from within the application.
  std::vector<uint8_t> data((uint8_t*)pShaderDebugInfo, (uint8_t*)pShaderDebugInfo + shaderDebugInfoSize);
  m_shaderDebugInfo[identifier].swap(data);

  // Write to file for later in-depth analysis of crash dumps with Nsight Graphics
  writeShaderDebugInformationToFile(identifier, pShaderDebugInfo, shaderDebugInfoSize);
}

// Handler for GPU crash dump description callbacks
void GpuCrashTracker::onDescription(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription)
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

// Handler for app-managed marker resolve callback
void GpuCrashTracker::onResolveMarker(const void* pMarker, void** resolvedMarkerData, uint32_t* markerSize)
{
  // Important: the pointer passed back via resolvedMarkerData must remain valid after this function returns
  // using references for all of the m_markerMap accesses ensures that the pointers refer to the persistent data
  for(const auto& map : m_markerMap)
  {
    const auto& found_marker = map.find((uint64_t)pMarker);
    if(found_marker != map.end())
    {
      const std::string& marker_data = found_marker->second;
      // std::string::data() will return a valid pointer until the string is next modified
      // we don't modify the string after calling data() here, so the pointer should remain valid
      *resolvedMarkerData = (void*)marker_data.data();
      *markerSize         = static_cast<uint32_t>(marker_data.length());
      return;
    }
  }
}

// Helper for writing a GPU crash dump to a file
void GpuCrashTracker::writeGpuCrashDumpToFile(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize)
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
  const std::string base_file_name =
      std::string(applicationName.data()) + "-" + std::to_string(baseInfo.pid) + "-" + std::to_string(++count);

  // Write the crash dump data to a file using the .nv-gpudmp extension
  // registered with Nsight Graphics.
  const std::string crash_dump_file_name = base_file_name + ".nv-gpudmp";
  std::ofstream     dump_file(crash_dump_file_name, std::ios::out | std::ios::binary);
  if(dump_file)
  {
    dump_file.write(static_cast<const char*>(pGpuCrashDump), gpuCrashDumpSize);
    dump_file.close();
  }

  // Decode the crash dump to a JSON string.
  // Step 1: Generate the JSON and get the size.
  uint32_t jsonSize = 0;
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GenerateJSON(
      decoder, GFSDK_Aftermath_GpuCrashDumpDecoderFlags_ALL_INFO, GFSDK_Aftermath_GpuCrashDumpFormatterFlags_NONE,
      shaderDebugInfoLookupCallback, shaderLookupCallback, shaderSourceDebugInfoLookupCallback, this, &jsonSize));
  // Step 2: Allocate a buffer and fetch the generated JSON.
  std::vector<char> json(jsonSize);
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_GetJSON(decoder, uint32_t(json.size()), json.data()));

  // Write the crash dump data as JSON to a file.
  const std::string json_file_name = crash_dump_file_name + ".json";
  std::ofstream     json_file(json_file_name, std::ios::out | std::ios::binary);
  if(json_file)
  {
    // Write the JSON to the file (excluding string termination)
    json_file.write(json.data(), json.size() - 1);
    json_file.close();
  }

  // Destroy the GPU crash dump decoder object.
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(decoder));
}

// Helper for writing shader debug information to a file
void GpuCrashTracker::writeShaderDebugInformationToFile(GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier,
                                                        const void*                               pShaderDebugInfo,
                                                        const uint32_t                            shaderDebugInfoSize)
{
  // Create a unique file name.
  const std::string file_path = "shader-" + std::to_string(identifier) + ".nvdbg";

  std::ofstream f(file_path, std::ios::out | std::ios::binary);
  if(f)
  {
    f.write(static_cast<const char*>(pShaderDebugInfo), shaderDebugInfoSize);
  }
}

// Handler for shader debug information lookup callbacks.
// This is used by the JSON decoder for mapping shader instruction
// addresses to SPIR-V IL lines or GLSL source lines.
void GpuCrashTracker::onShaderDebugInfoLookup(const GFSDK_Aftermath_ShaderDebugInfoIdentifier& identifier,
                                              PFN_GFSDK_Aftermath_SetData                      setShaderDebugInfo) const
{
  // Search the list of shader debug information blobs received earlier.
  auto i_debug_info = m_shaderDebugInfo.find(identifier);
  if(i_debug_info == m_shaderDebugInfo.end())
  {
    // Early exit, nothing found. No need to call setShaderDebugInfo.
    return;
  }

  // Let the GPU crash dump decoder know about the shader debug information
  // that was found.
  setShaderDebugInfo(i_debug_info->second.data(), static_cast<uint32_t>(i_debug_info->second.size()));
}

// Handler for shader lookup callbacks.
// This is used by the JSON decoder for mapping shader instruction
// addresses to SPIR-V IL lines or GLSL source lines.
// NOTE: If the application loads stripped shader binaries (ie; --strip-all in spirv-remap),
// Aftermath will require access to both the stripped and the not stripped
// shader binaries.
void GpuCrashTracker::onShaderLookup(const GFSDK_Aftermath_ShaderBinaryHash& shaderHash, PFN_GFSDK_Aftermath_SetData setShaderBinary) const
{
  // Find shader binary data for the shader hash in the shader database.
  std::vector<uint32_t> shader_binary;
  if(!findShaderBinary(shaderHash, shader_binary))
  {
    // Early exit, nothing found. No need to call setShaderBinary.
    return;
  }

  // Let the GPU crash dump decoder know about the shader data
  // that was found.
  setShaderBinary(shader_binary.data(), sizeof(uint32_t) * static_cast<uint32_t>(shader_binary.size()));
}

// Handler for shader source debug info lookup callbacks.
// This is used by the JSON decoder for mapping shader instruction addresses to
// GLSL source lines, if the shaders used by the application were compiled with
// separate debug info data files.
void GpuCrashTracker::onShaderSourceDebugInfoLookup(const GFSDK_Aftermath_ShaderDebugName& shaderDebugName,
                                                    PFN_GFSDK_Aftermath_SetData            setShaderBinary) const
{
  // Find source debug info for the shader DebugName in the shader database.
  std::vector<uint32_t> shader_binary;
  if(!findShaderBinaryWithDebugData(shaderDebugName, shader_binary))
  {
    // Early exit, nothing found. No need to call setShaderBinary.
    return;
  }

  // Let the GPU crash dump decoder know about the shader debug data that was
  // found.
  setShaderBinary(shader_binary.data(), sizeof(uint32_t) * static_cast<uint32_t>(shader_binary.size()));
}

// Static callback wrapper for OnCrashDump
void GpuCrashTracker::gpuCrashDumpCallback(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize, void* pUserData)
{
  auto* p_gpu_crash_tracker = reinterpret_cast<GpuCrashTracker*>(pUserData);
  p_gpu_crash_tracker->onCrashDump(pGpuCrashDump, gpuCrashDumpSize);
}

// Static callback wrapper for OnShaderDebugInfo
void GpuCrashTracker::shaderDebugInfoCallback(const void* pShaderDebugInfo, const uint32_t shaderDebugInfoSize, void* pUserData)
{
  auto* p_gpu_crash_tracker = reinterpret_cast<GpuCrashTracker*>(pUserData);
  p_gpu_crash_tracker->onShaderDebugInfo(pShaderDebugInfo, shaderDebugInfoSize);
}

// Static callback wrapper for OnDescription
void GpuCrashTracker::crashDumpDescriptionCallback(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription, void* pUserData)
{
  auto* p_gpu_crash_tracker = reinterpret_cast<GpuCrashTracker*>(pUserData);
  p_gpu_crash_tracker->onDescription(addDescription);
}

// Static callback wrapper for OnResolveMarker
void GpuCrashTracker::resolveMarkerCallback(const void* pMarker, void* pUserData, void** resolvedMarkerData, uint32_t* markerSize)
{
  auto* p_gpu_crash_tracker = reinterpret_cast<GpuCrashTracker*>(pUserData);
  p_gpu_crash_tracker->onResolveMarker(pMarker, resolvedMarkerData, markerSize);
}

// Static callback wrapper for OnShaderDebugInfoLookup
void GpuCrashTracker::shaderDebugInfoLookupCallback(const GFSDK_Aftermath_ShaderDebugInfoIdentifier* pIdentifier,
                                                    PFN_GFSDK_Aftermath_SetData                      setShaderDebugInfo,
                                                    void*                                            pUserData)
{
  auto* p_gpu_crash_tracker = reinterpret_cast<GpuCrashTracker*>(pUserData);
  p_gpu_crash_tracker->onShaderDebugInfoLookup(*pIdentifier, setShaderDebugInfo);
}

// Static callback wrapper for OnShaderLookup
void GpuCrashTracker::shaderLookupCallback(const GFSDK_Aftermath_ShaderBinaryHash* pShaderHash,
                                           PFN_GFSDK_Aftermath_SetData             setShaderBinary,
                                           void*                                   pUserData)
{
  auto* p_gpu_crash_tracker = reinterpret_cast<GpuCrashTracker*>(pUserData);
  p_gpu_crash_tracker->onShaderLookup(*pShaderHash, setShaderBinary);
}

// Static callback wrapper for OnShaderSourceDebugInfoLookup
void GpuCrashTracker::shaderSourceDebugInfoLookupCallback(const GFSDK_Aftermath_ShaderDebugName* pShaderDebugName,
                                                          PFN_GFSDK_Aftermath_SetData            setShaderBinary,
                                                          void*                                  pUserData)
{
  auto* p_gpu_crash_tracker = reinterpret_cast<GpuCrashTracker*>(pUserData);
  p_gpu_crash_tracker->onShaderSourceDebugInfoLookup(*pShaderDebugName, setShaderBinary);
}


void GpuCrashTracker::addShaderBinary(std::vector<uint32_t>& data)
{

  // Create shader hash for the shader
  const GFSDK_Aftermath_SpirvCode  shader{data.data(), static_cast<uint32_t>(data.size())};
  GFSDK_Aftermath_ShaderBinaryHash shaderHash{};
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetShaderHashSpirv(GFSDK_Aftermath_Version_API, &shader, &shaderHash));

  // Store the data for shader mapping when decoding GPU crash dumps.
  // cf. FindShaderBinary()
  m_shaderBinaries[shaderHash] = data;
}

void GpuCrashTracker::addShaderBinaryWithDebugInfo(std::vector<uint32_t>& data, std::vector<uint32_t>& strippedData)
{
  // Generate shader debug name.
  GFSDK_Aftermath_ShaderDebugName debugName{};
  const GFSDK_Aftermath_SpirvCode shader{data.data(), static_cast<uint32_t>(data.size())};
  const GFSDK_Aftermath_SpirvCode strippedShader{strippedData.data(), static_cast<uint32_t>(strippedData.size())};
  AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetShaderDebugNameSpirv(GFSDK_Aftermath_Version_API, &shader, &strippedShader, &debugName));

  // Store the data for shader instruction address mapping when decoding GPU crash dumps.
  // cf. FindShaderBinaryWithDebugData()
  m_shaderBinariesWithDebugInfo[debugName] = data;
}

// Find a shader binary by shader hash.
bool GpuCrashTracker::findShaderBinary(const GFSDK_Aftermath_ShaderBinaryHash& shaderHash, std::vector<uint32_t>& shader) const
{
  // Find shader binary data for the shader hash
  auto i_shader = m_shaderBinaries.find(shaderHash);
  if(i_shader == m_shaderBinaries.end())
  {
    // Nothing found.
    return false;
  }

  shader = i_shader->second;
  return true;
}

// Find a shader binary with debug information by shader debug name.
bool GpuCrashTracker::findShaderBinaryWithDebugData(const GFSDK_Aftermath_ShaderDebugName& shaderDebugName,
                                                    std::vector<uint32_t>&                 shader) const
{
  // Find shader binary for the shader debug name.
  auto i_shader = m_shaderBinariesWithDebugInfo.find(shaderDebugName);
  if(i_shader == m_shaderBinariesWithDebugInfo.end())
  {
    // Nothing found.
    return false;
  }

  shader = i_shader->second;
  return true;
}


#endif  // USE_NSIGHT_AFTERMATH
