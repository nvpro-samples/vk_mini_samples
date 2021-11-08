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


#include <sstream>

#include "vulkan/vulkan.h"

#include "optix.h"
#include "optix_function_table_definition.h"
#include "optix_stubs.h"

#include "denoiser.hpp"

#include "imgui/imgui_helper.h"
#include "nvvk/commands_vk.hpp"
#include "stb_image_write.h"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"


#include "_autogen/cpy_to_img.comp.h"
#include "_autogen/cpy_to_buffer.comp.h"

OptixDeviceContext m_optixDevice;
#define USE_COMPUTE 1

#define GRID_SIZE 16
inline VkExtent2D getGridSize(const VkExtent2D& size)
{
  return VkExtent2D{(size.width + (GRID_SIZE - 1)) / GRID_SIZE, (size.height + (GRID_SIZE - 1)) / GRID_SIZE};
}


//--------------------------------------------------------------------------------------------------
// The denoiser will take an image, will convert it to a buffer (compatible with Cuda)
// will denoise the buffer, and put back the buffer to an image.
//
// To make this working, it is important that the VkDeviceMemory associated with the buffer
// has the 'export' functionality.


void DenoiserOptix::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueIndex)
{
  m_queueIndex     = queueIndex;
  m_device         = device;
  m_physicalDevice = physicalDevice;
  m_memAlloc.init(device, physicalDevice);
  m_allocEx.init(device, physicalDevice, &m_memAlloc);
  m_debug.setup(device);
}

//--------------------------------------------------------------------------------------------------
// Initializing OptiX and creating the Denoiser instance
//
bool DenoiserOptix::initOptiX(const OptixDenoiserOptions& options, OptixPixelFormat pixelFormat, bool hdr)
{
  CUresult cuRes = cuInit(0);  // Initialize CUDA driver API.
  if(cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuInit() failed: " << cuRes << '\n';
    return false;
  }

  CUdevice device = 0;
  cuRes           = cuCtxCreate(&m_cudaContext, CU_CTX_SCHED_SPIN, device);
  if(cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuCtxCreate() failed: " << cuRes << '\n';
    return false;
  }

  // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
  cuRes = cuStreamCreate(&m_cuStream, CU_STREAM_DEFAULT);
  if(cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuStreamCreate() failed: " << cuRes << '\n';
    return false;
  }


  OPTIX_CHECK(optixInit());
  OPTIX_CHECK(optixDeviceContextCreate(m_cudaContext, nullptr, &m_optixDevice));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(m_optixDevice, context_log_cb, nullptr, 4));

  m_pixelFormat = pixelFormat;
  switch(pixelFormat)
  {

    case OPTIX_PIXEL_FORMAT_FLOAT3:
      m_sizeofPixel  = static_cast<uint32_t>(3 * sizeof(float));
      m_denoiseAlpha = 0;
      break;
    case OPTIX_PIXEL_FORMAT_FLOAT4:
      m_sizeofPixel  = static_cast<uint32_t>(4 * sizeof(float));
      m_denoiseAlpha = 1;
      break;
    case OPTIX_PIXEL_FORMAT_UCHAR3:
      m_sizeofPixel  = static_cast<uint32_t>(3 * sizeof(uint8_t));
      m_denoiseAlpha = 0;
      break;
    case OPTIX_PIXEL_FORMAT_UCHAR4:
      m_sizeofPixel  = static_cast<uint32_t>(4 * sizeof(uint8_t));
      m_denoiseAlpha = 1;
      break;
    case OPTIX_PIXEL_FORMAT_HALF3:
      m_sizeofPixel  = static_cast<uint32_t>(3 * sizeof(uint16_t));
      m_denoiseAlpha = 0;
      break;
    case OPTIX_PIXEL_FORMAT_HALF4:
      m_sizeofPixel  = static_cast<uint32_t>(4 * sizeof(uint16_t));
      m_denoiseAlpha = 1;
      break;
    default:
      assert(!"unsupported");
      break;
  }


  // This is to use RGB + Albedo + Normal
  m_dOptions                       = options;
  OptixDenoiserModelKind modelKind = hdr ? OPTIX_DENOISER_MODEL_KIND_HDR : OPTIX_DENOISER_MODEL_KIND_LDR;
  modelKind                        = OPTIX_DENOISER_MODEL_KIND_AOV;
  OPTIX_CHECK(optixDenoiserCreate(m_optixDevice, modelKind, &m_dOptions, &m_denoiser));


  return true;
}

//--------------------------------------------------------------------------------------------------
// Denoising the image in input and saving the denoised image in the output
//
void DenoiserOptix::denoiseImageBuffer(uint64_t& fenceValue)
{
  try
  {
    OptixPixelFormat pixelFormat      = m_pixelFormat;
    auto             sizeofPixel      = m_sizeofPixel;
    uint32_t         rowStrideInBytes = sizeofPixel * m_imageSize.width;

    //std::vector<OptixImage2D> inputLayer;  // Order: RGB, Albedo, Normal

    // Create and set our OptiX layers
    OptixDenoiserLayer layer = {};
    // Input
    layer.input.data               = (CUdeviceptr)m_pixelBufferIn[0].cudaPtr;
    layer.input.width              = m_imageSize.width;
    layer.input.height             = m_imageSize.height;
    layer.input.rowStrideInBytes   = rowStrideInBytes;
    layer.input.pixelStrideInBytes = m_sizeofPixel;
    layer.input.format             = pixelFormat;

    // Output
    layer.output.data               = (CUdeviceptr)m_pixelBufferOut.cudaPtr;
    layer.output.width              = m_imageSize.width;
    layer.output.height             = m_imageSize.height;
    layer.output.rowStrideInBytes   = rowStrideInBytes;
    layer.output.pixelStrideInBytes = sizeof(float) * 4;
    layer.output.format             = pixelFormat;


    OptixDenoiserGuideLayer guide_layer = {};
    // albedo
    if(m_dOptions.guideAlbedo)
    {
      guide_layer.albedo.data               = (CUdeviceptr)m_pixelBufferIn[1].cudaPtr;
      guide_layer.albedo.width              = m_imageSize.width;
      guide_layer.albedo.height             = m_imageSize.height;
      guide_layer.albedo.rowStrideInBytes   = rowStrideInBytes;
      guide_layer.albedo.pixelStrideInBytes = m_sizeofPixel;
      guide_layer.albedo.format             = pixelFormat;
    }

    // normal
    if(m_dOptions.guideNormal)
    {
      guide_layer.normal.data               = (CUdeviceptr)m_pixelBufferIn[2].cudaPtr;
      guide_layer.normal.width              = m_imageSize.width;
      guide_layer.normal.height             = m_imageSize.height;
      guide_layer.normal.rowStrideInBytes   = rowStrideInBytes;
      guide_layer.normal.pixelStrideInBytes = m_sizeofPixel;
      guide_layer.normal.format             = pixelFormat;
    }

    // Wait from Vulkan (Copy to Buffer)
    cudaExternalSemaphoreWaitParams waitParams{};
    waitParams.flags              = 0;
    waitParams.params.fence.value = fenceValue;
    cudaWaitExternalSemaphoresAsync(&m_semaphore.cu, &waitParams, 1, nullptr);

    if(m_dIntensity != 0)
    {
      OPTIX_CHECK(optixDenoiserComputeIntensity(m_denoiser, m_cuStream, &layer.input, m_dIntensity, m_dScratchBuffer,
                                                m_dSizes.withoutOverlapScratchSizeInBytes));
    }

    OptixDenoiserParams denoiserParams{};
    denoiserParams.denoiseAlpha = m_denoiseAlpha;
    denoiserParams.hdrIntensity = m_dIntensity;
    denoiserParams.blendFactor  = 0.0f;  // Fully denoised


    // Execute the denoiser
    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, m_cuStream, &denoiserParams, m_dStateBuffer, m_dSizes.stateSizeInBytes, &guide_layer,
                                    &layer, 1, 0, 0, m_dScratchBuffer, m_dSizes.withoutOverlapScratchSizeInBytes));


    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));  // Making sure the denoiser is done

    cudaExternalSemaphoreSignalParams sigParams{};
    sigParams.flags              = 0;
    sigParams.params.fence.value = ++fenceValue;
    cudaSignalExternalSemaphoresAsync(&m_semaphore.cu, &sigParams, 1, m_cuStream);
  }
  catch(const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
}

//--------------------------------------------------------------------------------------------------
// Converting the image to a buffer used by the denoiser
//
void DenoiserOptix::imageToBuffer(const VkCommandBuffer& cmdBuf, const std::vector<nvvk::Texture>& imgIn)
{
#if USE_COMPUTE
  copyImageToBuffer(cmdBuf, imgIn);
#else

  LABEL_SCOPE_VK(cmdBuf);
  for(int i = 0; i < static_cast<int>(imgIn.size()); i++)
  {
    const VkBuffer& pixelBufferIn = m_pixelBufferIn[i].bufVk.buffer;
    // Make the image layout eTransferSrcOptimal to copy to buffer
    VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    nvvk::cmdBarrierImageLayout(cmdBuf, imgIn[i].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresourceRange);

    // Copy the image to the buffer
    VkBufferImageCopy region           = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width           = m_imageSize.width;
    region.imageExtent.height          = m_imageSize.height;
    region.imageExtent.depth           = 1;
    vkCmdCopyImageToBuffer(cmdBuf, imgIn[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pixelBufferIn, 1, &region);

    // Put back the image as it was
    nvvk::cmdBarrierImageLayout(cmdBuf, imgIn[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
  }
#endif
}

//--------------------------------------------------------------------------------------------------
// Copying the image buffer to a buffer used by the denoiser
//
void DenoiserOptix::bufferToBuffer(const VkCommandBuffer& cmdBuf, const std::vector<nvvk::Buffer>& bufIn)
{
  LABEL_SCOPE_VK(cmdBuf);

  VkDeviceSize buf_size = static_cast<VkDeviceSize>(m_sizeofPixel * m_imageSize.width * m_imageSize.height);
  VkBufferCopy region{0, 0, buf_size};

  for(int i = 0; i < static_cast<int>(bufIn.size()); i++)
  {
    vkCmdCopyBuffer(cmdBuf, bufIn[i].buffer, m_pixelBufferIn[i].bufVk.buffer, 1, &region);
  }
}

//--------------------------------------------------------------------------------------------------
// Converting the output buffer to the image
//
void DenoiserOptix::bufferToImage(const VkCommandBuffer& cmdBuf, nvvk::Texture* imgOut)
{
#if USE_COMPUTE
  copyBufferToImage(cmdBuf, imgOut);
#else
  LABEL_SCOPE_VK(cmdBuf);
  const VkBuffer& pixelBufferOut = m_pixelBufferOut.bufVk.buffer;

  VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  nvvk::cmdBarrierImageLayout(cmdBuf, imgOut->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);

  // Copy the image to the buffer
  VkBufferImageCopy region           = {};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent.width           = m_imageSize.width;
  region.imageExtent.height          = m_imageSize.height;
  region.imageExtent.depth           = 1;
  vkCmdCopyBufferToImage(cmdBuf, pixelBufferOut, imgOut->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  // Put back the image as it was
  nvvk::cmdBarrierImageLayout(cmdBuf, imgOut->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);


#endif
}


//--------------------------------------------------------------------------------------------------
//
//
void DenoiserOptix::destroy()
{
  vkDestroySemaphore(m_device, m_semaphore.vk, nullptr);
  m_semaphore.vk = VK_NULL_HANDLE;

  destroyBuffer();
  for(auto& d : m_desc)
  {
    vkDestroyDescriptorPool(m_device, d.pool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, d.layout, nullptr);
    d.pool   = VK_NULL_HANDLE;
    d.layout = VK_NULL_HANDLE;
  }
  for(auto& p : m_pipelines)
  {
    vkDestroyPipeline(m_device, p.p, nullptr);
    vkDestroyPipelineLayout(m_device, p.layout, nullptr);
    p.p      = VK_NULL_HANDLE;
    p.layout = VK_NULL_HANDLE;
  }
}

//--------------------------------------------------------------------------------------------------
//
//
void DenoiserOptix::destroyBuffer()
{
  for(auto& p : m_pixelBufferIn)
    p.destroy(m_allocEx);
  m_pixelBufferOut.destroy(m_allocEx);

  if(m_dStateBuffer != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dStateBuffer));
    m_dStateBuffer = NULL;
  }
  if(m_dScratchBuffer != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dScratchBuffer));
    m_dScratchBuffer = NULL;
  }
  if(m_dIntensity != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dIntensity));
    m_dIntensity = NULL;
  }
  if(m_dMinRGB != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dMinRGB));
    m_dMinRGB = NULL;
  }
}

//--------------------------------------------------------------------------------------------------
// UI specific for the denoiser
//
bool DenoiserOptix::uiSetup()
{
  bool modified = false;
  if(ImGui::CollapsingHeader("Denoiser", ImGuiTreeNodeFlags_DefaultOpen))
  {
    modified |= ImGuiH::Control::Checkbox("Denoise", "", (bool*)&m_denoisedMode);
    modified |= ImGuiH::Control::Slider("Start Frame", "Frame at which the denoiser starts to be applied",
                                        &m_startDenoiserFrame, nullptr, ImGuiH::Control::Flags::Normal, 0, 99);
  }
  return modified;
}

//--------------------------------------------------------------------------------------------------
// Allocating all the buffers in which the images will be transfered.
// The buffers are shared with Cuda, therefore OptiX can denoised them
//
void DenoiserOptix::allocateBuffers(const VkExtent2D& imgSize)
{
  m_imageSize = imgSize;

  destroyBuffer();

  VkDeviceSize bufferSize = static_cast<unsigned long long>(m_imageSize.width) * m_imageSize.height * 4 * sizeof(float);
  VkBufferUsageFlags usage{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT};

  {  // Color
    m_pixelBufferIn[0].bufVk = m_allocEx.createBuffer(bufferSize, usage, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
    createBufferCuda(m_pixelBufferIn[0]);  // Exporting the buffer to Cuda handle and pointers
    NAME_VK(m_pixelBufferIn[0].bufVk.buffer);
  }

  // Albedo
  {
    m_pixelBufferIn[1].bufVk = m_allocEx.createBuffer(bufferSize, usage, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
    createBufferCuda(m_pixelBufferIn[1]);
    NAME_VK(m_pixelBufferIn[1].bufVk.buffer);
  }
  // Normal
  {
    m_pixelBufferIn[2].bufVk = m_allocEx.createBuffer(bufferSize, usage, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
    createBufferCuda(m_pixelBufferIn[2]);
    NAME_VK(m_pixelBufferIn[2].bufVk.buffer);
  }

  // Output image/buffer
  m_pixelBufferOut.bufVk =
      m_allocEx.createBuffer(bufferSize, usage | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
  createBufferCuda(m_pixelBufferOut);
  NAME_VK(m_pixelBufferOut.bufVk.buffer);


  // Computing the amount of memory needed to do the denoiser
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_imageSize.width, m_imageSize.height, &m_dSizes));

  CUDA_CHECK(cudaMalloc((void**)&m_dStateBuffer, m_dSizes.stateSizeInBytes));
  CUDA_CHECK(cudaMalloc((void**)&m_dScratchBuffer, m_dSizes.withoutOverlapScratchSizeInBytes));
  CUDA_CHECK(cudaMalloc((void**)&m_dMinRGB, 4 * sizeof(float)));
  if(m_pixelFormat == OPTIX_PIXEL_FORMAT_FLOAT3 || m_pixelFormat == OPTIX_PIXEL_FORMAT_FLOAT4)
    CUDA_CHECK(cudaMalloc((void**)&m_dIntensity, sizeof(float)));

  OPTIX_CHECK(optixDenoiserSetup(m_denoiser, m_cuStream, m_imageSize.width, m_imageSize.height, m_dStateBuffer,
                                 m_dSizes.stateSizeInBytes, m_dScratchBuffer, m_dSizes.withoutOverlapScratchSizeInBytes));
}


//--------------------------------------------------------------------------------------------------
// Get the Vulkan buffer and create the Cuda equivalent using the memory allocated in Vulkan
//
void DenoiserOptix::createBufferCuda(BufferCuda& buf)
{
  nvvk::MemAllocator::MemInfo memInfo = m_allocEx.getMemoryAllocator()->getMemoryInfo(buf.bufVk.memHandle);
#ifdef WIN32
  VkMemoryGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  info.memory     = memInfo.memory;
  info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  vkGetMemoryWin32HandleKHR(m_device, &info, &buf.handle);
#else
  VkMemoryGetFdInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  info.memory     = memInfo.memory;
  info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  vkGetMemoryFdKHR(m_device, &info, &buf.handle);

#endif

  VkMemoryRequirements memoryReq{};
  vkGetBufferMemoryRequirements(m_device, buf.bufVk.buffer, &memoryReq);

  cudaExternalMemoryHandleDesc cudaExtMemHandleDesc{};
  cudaExtMemHandleDesc.size = memoryReq.size;
#ifdef WIN32
  cudaExtMemHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
  cudaExtMemHandleDesc.handle.win32.handle = buf.handle;
#else
  cudaExtMemHandleDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
  cudaExtMemHandleDesc.handle.fd = buf.handle;
#endif

  cudaExternalMemory_t cudaExtMemVertexBuffer{};
  CUDA_CHECK(cudaImportExternalMemory(&cudaExtMemVertexBuffer, &cudaExtMemHandleDesc));

#ifndef WIN32
  // fd got consumed
  cudaExtMemHandleDesc.handle.fd = -1;
#endif

  cudaExternalMemoryBufferDesc cudaExtBufferDesc{};
  cudaExtBufferDesc.offset = 0;
  cudaExtBufferDesc.size   = memoryReq.size;
  cudaExtBufferDesc.flags  = 0;
  CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&buf.cudaPtr, cudaExtMemVertexBuffer, &cudaExtBufferDesc));
}

//--------------------------------------------------------------------------------------------------
// Creating the timeline semaphores for syncing with CUDA
//
void DenoiserOptix::createSemaphore()
{
#ifdef WIN32
  auto handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  auto handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

  VkSemaphoreTypeCreateInfo timelineCreateInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  timelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timelineCreateInfo.initialValue  = 0;

  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &timelineCreateInfo;

  VkExportSemaphoreCreateInfo esci{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR};
  esci.pNext       = &timelineCreateInfo;
  sci.pNext        = &esci;
  esci.handleTypes = handleType;

  vkCreateSemaphore(m_device, &sci, nullptr, &m_semaphore.vk);

#ifdef WIN32
  VkSemaphoreGetWin32HandleInfoKHR handleInfo{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
  handleInfo.handleType = handleType;
  handleInfo.semaphore  = m_semaphore.vk;
  vkGetSemaphoreWin32HandleKHR(m_device, &handleInfo, &m_semaphore.handle);
#else
  VkSemaphoreGetFdInfoKHR{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
  handleInfo.handleType = handleType;
  handleInfo.semaphore  = m_semaphore.vk;
  vkGetSemaphoreFdKHR(m_device, &handleInfo, &m_semaphore.handle);

#endif


  cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc{};
  std::memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
  externalSemaphoreHandleDesc.flags = 0;
#ifdef WIN32
  externalSemaphoreHandleDesc.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
  externalSemaphoreHandleDesc.handle.win32.handle = (void*)m_semaphore.handle;
#else
  externalSemaphoreHandleDesc.type      = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  externalSemaphoreHandleDesc.handle.fd = m_semaphore.handle;
#endif

  CUDA_CHECK(cudaImportExternalSemaphore(&m_semaphore.cu, &externalSemaphoreHandleDesc));
}

extern std::vector<std::string> defaultSearchPaths;

void DenoiserOptix::createCopyPipeline()
{
  {
    constexpr uint32_t SHD = 0;
    // Descriptor Set
    nvvk::DescriptorSetBindings bind;
    bind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    CREATE_NAMED_VK(m_desc[SHD].pool, bind.createPool(m_device, 1));
    CREATE_NAMED_VK(m_desc[SHD].layout, bind.createLayout(m_device));
    CREATE_NAMED_VK(m_desc[SHD].set, nvvk::allocateDescriptorSet(m_device, m_desc[SHD].pool, m_desc[SHD].layout));

    // Pipeline
    VkPipelineLayoutCreateInfo pipeInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeInfo.setLayoutCount = 1;
    pipeInfo.pSetLayouts    = &m_desc[SHD].layout;
    vkCreatePipelineLayout(m_device, &pipeInfo, nullptr, &m_pipelines[SHD].layout);
    NAME_VK(m_pipelines[SHD].layout);


    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = nvvk::createShaderModule(m_device, cpy_to_buffer_comp, sizeof(cpy_to_buffer_comp));
    stageInfo.pName  = "main";

    VkComputePipelineCreateInfo compInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compInfo.layout = m_pipelines[SHD].layout;
    compInfo.stage  = stageInfo;


    vkCreateComputePipelines(m_device, {}, 1, &compInfo, nullptr, &m_pipelines[SHD].p);
    NAME_VK(m_pipelines[SHD].p);

    vkDestroyShaderModule(m_device, compInfo.stage.module, nullptr);
  }

  {
    constexpr uint32_t SHD = 1;
    // Descriptor Set
    nvvk::DescriptorSetBindings bind;
    bind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    CREATE_NAMED_VK(m_desc[SHD].pool, bind.createPool(m_device, 1));
    CREATE_NAMED_VK(m_desc[SHD].layout, bind.createLayout(m_device));
    CREATE_NAMED_VK(m_desc[SHD].set, nvvk::allocateDescriptorSet(m_device, m_desc[SHD].pool, m_desc[SHD].layout));
    // Pipeline
    VkPipelineLayoutCreateInfo pipeInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeInfo.setLayoutCount = 1;
    pipeInfo.pSetLayouts    = &m_desc[SHD].layout;
    vkCreatePipelineLayout(m_device, &pipeInfo, nullptr, &m_pipelines[SHD].layout);
    NAME_VK(m_pipelines[SHD].layout);


    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = nvvk::createShaderModule(m_device, cpy_to_img_comp, sizeof(cpy_to_img_comp));
    stageInfo.pName  = "main";

    VkComputePipelineCreateInfo compInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compInfo.layout = m_pipelines[SHD].layout;
    compInfo.stage  = stageInfo;


    vkCreateComputePipelines(m_device, {}, 1, &compInfo, nullptr, &m_pipelines[SHD].p);
    NAME_VK(m_pipelines[SHD].p);

    vkDestroyShaderModule(m_device, compInfo.stage.module, nullptr);
  }
}

VkWriteDescriptorSet makeWrite(const VkDescriptorSet& set, uint32_t bind, const VkDescriptorImageInfo* img)
{
  VkWriteDescriptorSet wrt{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wrt.dstSet          = set;
  wrt.dstBinding      = bind;
  wrt.dstArrayElement = 0;
  wrt.descriptorCount = 1;
  wrt.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  wrt.pImageInfo      = img;
  return wrt;
};

VkWriteDescriptorSet makeWrite(const VkDescriptorSet& set, uint32_t bind, const VkDescriptorBufferInfo* buf)
{
  VkWriteDescriptorSet wrt{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wrt.dstSet          = set;
  wrt.dstBinding      = bind;
  wrt.dstArrayElement = 0;
  wrt.descriptorCount = 1;
  wrt.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wrt.pBufferInfo     = buf;
  return wrt;
};

//--------------------------------------------------------------------------------------------------
// Compute version of VkCmdCopyImageToBuffer
//
void DenoiserOptix::copyImageToBuffer(const VkCommandBuffer& cmd, const std::vector<nvvk::Texture>& imgIn)
{
  LABEL_SCOPE_VK(cmd);
  constexpr uint32_t SHD = 0;

  VkDescriptorImageInfo  img0{imgIn[0].descriptor};
  VkDescriptorImageInfo  img1{imgIn[1].descriptor};
  VkDescriptorImageInfo  img2{imgIn[2].descriptor};
  VkDescriptorBufferInfo buf0{m_pixelBufferIn[0].bufVk.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo buf1{m_pixelBufferIn[1].bufVk.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo buf2{m_pixelBufferIn[2].bufVk.buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(makeWrite(m_desc[SHD].set, 0, &img0));
  writes.emplace_back(makeWrite(m_desc[SHD].set, 1, &img1));
  writes.emplace_back(makeWrite(m_desc[SHD].set, 2, &img2));
  writes.emplace_back(makeWrite(m_desc[SHD].set, 3, &buf0));
  writes.emplace_back(makeWrite(m_desc[SHD].set, 4, &buf1));
  writes.emplace_back(makeWrite(m_desc[SHD].set, 5, &buf2));
  vkUpdateDescriptorSets(m_device, (uint32_t)writes.size(), writes.data(), 0, nullptr);


  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[SHD].layout, 0, 1, &m_desc[SHD].set, 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[SHD].p);
  auto grid = getGridSize(m_imageSize);
  vkCmdDispatch(cmd, grid.width, grid.height, 1);
}


//--------------------------------------------------------------------------------------------------
// Compute version of VkCmdCopyBufferToImage
//
void DenoiserOptix::copyBufferToImage(const VkCommandBuffer& cmd, const nvvk::Texture* imgIn)
{
  LABEL_SCOPE_VK(cmd);
  constexpr uint32_t SHD = 1;

  VkDescriptorImageInfo  img0{imgIn->descriptor};
  VkDescriptorBufferInfo buf0{m_pixelBufferOut.bufVk.buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(makeWrite(m_desc[SHD].set, 0, &img0));
  writes.emplace_back(makeWrite(m_desc[SHD].set, 1, &buf0));
  vkUpdateDescriptorSets(m_device, (uint32_t)writes.size(), writes.data(), 0, nullptr);


  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[SHD].layout, 0, 1, &m_desc[SHD].set, 0, nullptr);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[SHD].p);
  auto grid = getGridSize(m_imageSize);
  vkCmdDispatch(cmd, grid.width, grid.height, 1);
}
