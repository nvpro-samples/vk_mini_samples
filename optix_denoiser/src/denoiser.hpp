/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#pragma once

#include <array>
#include <iomanip>   // cerr
#include <iostream>  // setw

#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/memallocator_dedicated_vk.hpp"

#ifdef LINUX
#include <unistd.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include "imgui.h"

// for interop we use the dedicated allocator as well
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "optix_types.h"
#include <driver_types.h>


#define OPTIX_CHECK(call)                                                                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    OptixResult res = call;                                                                                            \
    if(res != OPTIX_SUCCESS)                                                                                           \
    {                                                                                                                  \
      std::stringstream ss;                                                                                            \
      ss << "Optix call (" << #call << " ) failed with code " << res << " (" __FILE__ << ":" << __LINE__ << ")\n";     \
      std::cerr << ss.str().c_str() << std::endl;                                                                      \
      throw std::runtime_error(ss.str().c_str());                                                                      \
    }                                                                                                                  \
  } while(false)

#define CUDA_CHECK(call)                                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    cudaError_t error = call;                                                                                          \
    if(error != cudaSuccess)                                                                                           \
    {                                                                                                                  \
      std::stringstream ss;                                                                                            \
      ss << "CUDA call (" << #call << " ) failed with code " << error << " (" __FILE__ << ":" << __LINE__ << ")\n";    \
      throw std::runtime_error(ss.str().c_str());                                                                      \
    }                                                                                                                  \
  } while(false)

#define OPTIX_CHECK_LOG(call)                                                                                           \
  do                                                                                                                    \
  {                                                                                                                     \
    OptixResult res = call;                                                                                             \
    if(res != OPTIX_SUCCESS)                                                                                            \
    {                                                                                                                   \
      std::stringstream ss;                                                                                             \
      ss << "Optix call (" << #call << " ) failed with code " << res << " (" __FILE__ << ":" << __LINE__ << ")\nLog:\n" \
         << log << "\n";                                                                                                \
      throw std::runtime_error(ss.str().c_str());                                                                       \
    }                                                                                                                   \
  } while(false)

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}


struct DenoiserOptix
{

  DenoiserOptix() = default;
  ~DenoiserOptix() { cuCtxDestroy(m_cudaContext); }

  void setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueIndex);
  bool initOptiX(const OptixDenoiserOptions& options, OptixPixelFormat pixelFormat, bool hdr);
  void denoiseImageBuffer(uint64_t& fenceValue);
  void createSemaphore();

  void destroy();
  void destroyBuffer();

  bool uiSetup();

  void allocateBuffers(const VkExtent2D& imgSize);
  void bufferToImage(const VkCommandBuffer& cmdBuf, nvvk::Texture* imgOut);
  void imageToBuffer(const VkCommandBuffer& cmdBuf, const std::vector<nvvk::Texture>& imgIn);
  void bufferToBuffer(const VkCommandBuffer& cmdBuf, const std::vector<nvvk::Buffer>& bufIn);

  void createCopyPipeline();
  void copyImageToBuffer(const VkCommandBuffer& cmd, const std::vector<nvvk::Texture>& imgIn);
  void copyBufferToImage(const VkCommandBuffer& cmd, const nvvk::Texture* imgIn);

  VkSemaphore getTLSemaphore() { return m_semaphore.vk; }

  // Ui
  int m_denoisedMode{1};
  int m_startDenoiserFrame{0};

private:
  // Holding the Buffer for Cuda interop
  struct BufferCuda
  {
    nvvk::Buffer bufVk;  // The Vulkan allocated buffer

    // Extra for Cuda
#ifdef WIN32
    HANDLE handle = nullptr;  // The Win32 handle
#else
    int handle = -1;
#endif
    void* cudaPtr = nullptr;

    void destroy(nvvk::ExportResourceAllocator& alloc)
    {
      alloc.destroy(bufVk);
#ifdef WIN32
      CloseHandle(handle);
      handle = 0;
#else
      if(handle != -1)
      {
        close(handle);
        handle = -1;
      }
#endif
    }
  };

  void createBufferCuda(BufferCuda& buf);


  // For synchronizing with Vulkan
  struct Semaphore
  {
    VkSemaphore             vk;  // Vulkan
    cudaExternalSemaphore_t cu;  // Cuda version
#ifdef WIN32
    HANDLE handle{INVALID_HANDLE_VALUE};
#else
    int handle{-1};
#endif
  } m_semaphore;


  OptixDenoiser        m_denoiser{nullptr};
  OptixDenoiserOptions m_dOptions{};
  OptixDenoiserSizes   m_dSizes{};
  CUdeviceptr          m_dStateBuffer{0};
  CUdeviceptr          m_dScratchBuffer{0};
  CUdeviceptr          m_dIntensity{0};
  CUdeviceptr          m_dMinRGB{0};
  CUcontext            m_cudaContext{nullptr};

  VkDevice         m_device;
  VkPhysicalDevice m_physicalDevice;
  uint32_t         m_queueIndex;

  nvvk::DedicatedMemoryAllocator m_memAlloc;  // Using dedicated allocations for simplicity
  nvvk::ExportResourceAllocator  m_allocEx;   // ResourceAllocator with export flag (interop)

  VkExtent2D                m_imageSize;
  std::array<BufferCuda, 3> m_pixelBufferIn;  // RGB, Albedo, normal
  BufferCuda                m_pixelBufferOut;

  OptixPixelFormat m_pixelFormat;
  uint32_t         m_sizeofPixel;
  int              m_denoiseAlpha{0};
  CUstream         m_cuStream;

  nvvk::DebugUtil m_debug;

  struct vkDescritors
  {
    VkDescriptorPool      pool;
    VkDescriptorSetLayout layout;
    VkDescriptorSet       set;
  };
  std::array<vkDescritors, 2> m_desc;

  struct vkPipelines
  {
    VkPipeline       p;
    VkPipelineLayout layout;
  };
  std::array<vkPipelines, 2> m_pipelines;
};
