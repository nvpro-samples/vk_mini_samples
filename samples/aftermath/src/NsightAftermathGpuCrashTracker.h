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

#pragma once

#ifdef USE_NSIGHT_AFTERMATH

#include <map>
#include <mutex>

#include "NsightAftermathHelpers.h"

//*********************************************************
// Implements GPU crash dump tracking using the Nsight
// Aftermath API.
//
class GpuCrashTracker
{
public:
  // keep four frames worth of marker history
  const static unsigned int                                                 c_markerFrameHistory = 4;
  typedef std::array<std::map<uint64_t, std::string>, c_markerFrameHistory> MarkerMap;

  GpuCrashTracker(const MarkerMap& markerMap);
  ~GpuCrashTracker();

  // Initialize the GPU crash dump tracker.
  void initialize();

  // Track a shader compiled with -g
  void addShaderBinary(const std::vector<uint32_t>& data);

  // Track an optimized shader with additional debug information
  void addShaderBinaryWithDebugInfo(std::vector<uint32_t>& data, std::vector<uint32_t>& strippedData);


private:
  //*********************************************************
  // Callback handlers for GPU crash dumps and related data.
  //

  // Handler for GPU crash dump callbacks.
  void onCrashDump(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize);

  // Handler for shader debug information callbacks.
  void onShaderDebugInfo(const void* pShaderDebugInfo, const uint32_t shaderDebugInfoSize);

  // Handler for GPU crash dump description callbacks.
  static void onDescription(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription);

  // Handler for app-managed marker resolve callback
  void onResolveMarker(const void* pMarker, void** resolvedMarkerData, uint32_t* markerSize);

  //*********************************************************
  // Helpers for writing a GPU crash dump and debug information
  // data to files.
  //

  // Helper for writing a GPU crash dump to a file.
  void writeGpuCrashDumpToFile(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize);

  // Helper for writing shader debug information to a file
  static void writeShaderDebugInformationToFile(GFSDK_Aftermath_ShaderDebugInfoIdentifier identifier,
                                                const void*                               pShaderDebugInfo,
                                                const uint32_t                            shaderDebugInfoSize);

  //*********************************************************
  // Helpers for decoding GPU crash dump to JSON.
  //

  // Handler for shader debug info lookup callbacks.
  void onShaderDebugInfoLookup(const GFSDK_Aftermath_ShaderDebugInfoIdentifier& identifier,
                               PFN_GFSDK_Aftermath_SetData                      setShaderDebugInfo) const;

  // Handler for shader lookup callbacks.
  void onShaderLookup(const GFSDK_Aftermath_ShaderBinaryHash& shaderHash, PFN_GFSDK_Aftermath_SetData setShaderBinary) const;

  // Handler for shader source debug info lookup callbacks.
  void onShaderSourceDebugInfoLookup(const GFSDK_Aftermath_ShaderDebugName& shaderDebugName,
                                     PFN_GFSDK_Aftermath_SetData            setShaderBinary) const;

  //*********************************************************
  // Static callback wrappers.
  //

  // GPU crash dump callback.
  static void gpuCrashDumpCallback(const void* pGpuCrashDump, const uint32_t gpuCrashDumpSize, void* pUserData);

  // Shader debug information callback.
  static void shaderDebugInfoCallback(const void* pShaderDebugInfo, const uint32_t shaderDebugInfoSize, void* pUserData);

  // GPU crash dump description callback.
  static void crashDumpDescriptionCallback(PFN_GFSDK_Aftermath_AddGpuCrashDumpDescription addDescription, void* pUserData);

  // App-managed marker resolve callback
  static void resolveMarkerCallback(const void* pMarker, const uint32_t markerDataSize, void* pUserData, void** resolvedMarkerData, uint32_t* markerSize);

  // Shader debug information lookup callback.
  static void shaderDebugInfoLookupCallback(const GFSDK_Aftermath_ShaderDebugInfoIdentifier* pIdentifier,
                                            PFN_GFSDK_Aftermath_SetData                      setShaderDebugInfo,
                                            void*                                            pUserData);

  // Shader lookup callback.
  static void shaderLookupCallback(const GFSDK_Aftermath_ShaderBinaryHash* pShaderHash,
                                   PFN_GFSDK_Aftermath_SetData             setShaderBinary,
                                   void*                                   pUserData);

  // Shader source debug info lookup callback.
  static void shaderSourceDebugInfoLookupCallback(const GFSDK_Aftermath_ShaderDebugName* pShaderDebugName,
                                                  PFN_GFSDK_Aftermath_SetData            setShaderBinary,
                                                  void*                                  pUserData);

  //*********************************************************
  // GPU crash tracker state.
  //

  // Is the GPU crash dump tracker initialized?
  bool m_initialized;

  // For thread-safe access of GPU crash tracker state.
  mutable std::mutex m_mutex;

  // List of Shader Debug Information by ShaderDebugInfoIdentifier.
  std::map<GFSDK_Aftermath_ShaderDebugInfoIdentifier, std::vector<uint8_t>> m_shaderDebugInfo;

  // App-managed marker tracking
  const MarkerMap& m_markerMap;

  //*********************************************************
  // SAhder database .
  //

  // Find a shader bytecode binary by shader hash.
  bool findShaderBinary(const GFSDK_Aftermath_ShaderBinaryHash& shaderHash, std::vector<uint32_t>& shader) const;

  // Find a source shader debug info by shader debug name generated by the DXC compiler.
  bool findShaderBinaryWithDebugData(const GFSDK_Aftermath_ShaderDebugName& shaderDebugName, std::vector<uint32_t>& shader) const;


  // List of shader binaries by ShaderBinaryHash.
  std::map<GFSDK_Aftermath_ShaderBinaryHash, std::vector<uint32_t>> m_shaderBinaries;

  // List of available shader binaries with source debug information by ShaderDebugName.
  std::map<GFSDK_Aftermath_ShaderDebugName, std::vector<uint32_t>> m_shaderBinariesWithDebugInfo;
};


#endif  // USE_NSIGHT_AFTERMATH