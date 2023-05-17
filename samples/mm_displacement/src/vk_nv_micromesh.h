/*
* SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: LicenseRef-NvidiaProprietary
*
* NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
* property and proprietary rights in and to this material, related
* documentation and any modifications thereto. Any use, reproduction,
* disclosure or distribution of this material and related documentation
* without an express license agreement from NVIDIA CORPORATION or
* its affiliates is strictly prohibited.
*/

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//
// WARNING: VK_NV_displacement_micromap is in beta and subject to future changes.
//          Do not use these headers in production code.
//
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//
// This folder provides type definitions and function interfaces for the Vulkan
// micromap API, if they weren't provided by the headers in the Vulkan SDK and the
// loader in nvpro_core/nvvk/extensions_vk. In other words, this allows the APIs to
// be called even when the Vulkan SDK doesn't list them. Once NVVK requires a
// Vulkan SDK of at least 1.3.236.0, the opacity micromap definitions can be deleted
// from here.

#pragma once

#include <vulkan/vulkan_core.h>
#include <vulkan/vulkan_beta.h>

// for name, value in test:gmatch("(VK_[%w_]+) = ([%w_]+),") do print("#define "..name.."  ((VkStructureType)"..value..")") end


#ifdef __cplusplus
extern "C" {
#endif

// clang-format off
#ifndef VK_EXT_opacity_micromap
#define VULKAN_NV_DEFINED_EXT_opacity_micromap 1 // Defined if vulkan_core.h doesn't include VK_EXT_opacity_micromap

#define VK_EXT_opacity_micromap 1
#define VK_EXT_OPACITY_MICROMAP_SPEC_VERSION                         2
#define VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME                       "VK_EXT_opacity_micromap"
#define VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT                    ((VkStructureType)1000396000)
#define VK_STRUCTURE_TYPE_MICROMAP_VERSION_INFO_EXT                  ((VkStructureType)1000396001)
#define VK_STRUCTURE_TYPE_COPY_MICROMAP_INFO_EXT                     ((VkStructureType)1000396002)
#define VK_STRUCTURE_TYPE_COPY_MICROMAP_TO_MEMORY_INFO_EXT           ((VkStructureType)1000396003)
#define VK_STRUCTURE_TYPE_COPY_MEMORY_TO_MICROMAP_INFO_EXT           ((VkStructureType)1000396004)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT ((VkStructureType)1000396005)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT ((VkStructureType)1000396006)
#define VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT                   ((VkStructureType)1000396007)
#define VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT              ((VkStructureType)1000396008)
#define VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT ((VkStructureType)1000396009)
#define VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT                   ((VkPipelineStageFlagBits2)0x0000000040000000ULL)
#define VK_ACCESS_2_MICROMAP_READ_BIT_EXT                            ((VkAccessFlagBits2)0x0000100000000000ULL)
#define VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT                           ((VkAccessFlagBits2)0x0000200000000000ULL)
#define VK_QUERY_TYPE_MICROMAP_SERIALIZATION_SIZE_EXT                ((VkQueryType)1000396000)
#define VK_QUERY_TYPE_MICROMAP_COMPACTED_SIZE_EXT                    ((VkQueryType)1000396001)
#define VK_OBJECT_TYPE_MICROMAP_EXT                                  ((VkObjectType)1000396000)
#define VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT       ((VkBufferUsageFlagBits)0x00800000)
#define VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT                     ((VkBufferUsageFlagBits)0x01000000)
#define VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT      ((VkPipelineCreateFlagBits)0x01000000)
#define VK_GEOMETRY_INSTANCE_FORCE_OPACITY_MICROMAP_2_STATE_EXT      ((VkGeometryInstanceFlagBitsKHR)0x00000010)
#define VK_GEOMETRY_INSTANCE_DISABLE_OPACITY_MICROMAPS_EXT           ((VkGeometryInstanceFlagBitsKHR)0x00000020)
#define VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_UPDATE_EXT ((VkBuildAccelerationStructureFlagBitsKHR)0x00000040)
#define VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DISABLE_OPACITY_MICROMAPS_EXT ((VkBuildAccelerationStructureFlagBitsKHR)0x00000080)
#define VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_DATA_UPDATE_EXT ((VkBuildAccelerationStructureFlagBitsKHR)0x00000100)

  typedef enum VkMicromapTypeEXT {
    VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT = 0,
    VK_MICROMAP_TYPE_MAX_ENUM_EXT = 0x7FFFFFFF
  } VkMicromapTypeEXT;

  typedef struct VkMicromapUsageEXT {
    uint32_t                              count;
    uint32_t                              subdivisionLevel;
    uint32_t                              format;
  } VkMicromapUsageEXT;

  VK_DEFINE_NON_DISPATCHABLE_HANDLE(VkMicromapEXT)

    typedef VkFlags VkBuildMicromapFlagsEXT;

  typedef enum VkBuildMicromapModeEXT {
    VK_BUILD_MICROMAP_MODE_BUILD_EXT = 0,
    VK_BUILD_MICROMAP_MODE_MAX_ENUM_EXT = 0x7FFFFFFF
  } VkBuildMicromapModeEXT;

  typedef struct VkMicromapBuildInfoEXT {
    VkStructureType                       sType;
    void const*                           pNext;
    VkMicromapTypeEXT                     type;
    VkBuildMicromapFlagsEXT               flags;
    VkBuildMicromapModeEXT                mode;
    VkMicromapEXT                         dstMicromap;
    uint32_t                              usageCountsCount;
    VkMicromapUsageEXT const*             pUsageCounts;
    VkMicromapUsageEXT const* const*      ppUsageCounts;
    VkDeviceOrHostAddressConstKHR         data;
    VkDeviceOrHostAddressKHR              scratchData;
    VkDeviceOrHostAddressConstKHR         triangleArray;
    VkDeviceSize                          triangleArrayStride;
  } VkMicromapBuildInfoEXT;

  typedef VkFlags VkMicromapCreateFlagsEXT;

  typedef struct VkMicromapCreateInfoEXT {
    VkStructureType                       sType;
    void const*                           pNext;
    VkMicromapCreateFlagsEXT              createFlags;
    VkBuffer                              buffer;
    VkDeviceSize                          offset;
    VkDeviceSize                          size;
    VkMicromapTypeEXT                     type;
    VkDeviceAddress                       deviceAddress;
  } VkMicromapCreateInfoEXT;

  typedef enum VkBuildMicromapFlagBitsEXT {
    VK_BUILD_MICROMAP_PREFER_FAST_TRACE_BIT_EXT = 0x00000001,
    VK_BUILD_MICROMAP_PREFER_FAST_BUILD_BIT_EXT = 0x00000002,
    VK_BUILD_MICROMAP_ALLOW_COMPACTION_BIT_EXT = 0x00000004,
    VK_BUILD_MICROMAP_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
  } VkBuildMicromapFlagBitsEXT;

  typedef enum VkCopyMicromapModeEXT {
    VK_COPY_MICROMAP_MODE_CLONE_EXT = 0,
    VK_COPY_MICROMAP_MODE_SERIALIZE_EXT = 1,
    VK_COPY_MICROMAP_MODE_DESERIALIZE_EXT = 2,
    VK_COPY_MICROMAP_MODE_COMPACT_EXT = 3,
    VK_COPY_MICROMAP_MODE_MAX_ENUM_EXT = 0x7FFFFFFF
  } VkCopyMicromapModeEXT;

  typedef struct VkPhysicalDeviceOpacityMicromapFeaturesEXT {
    VkStructureType                       sType;
    void*                                 pNext;
    VkBool32                              micromap;
    VkBool32                              micromapCaptureReplay;
    VkBool32                              micromapHostCommands;
  } VkPhysicalDeviceOpacityMicromapFeaturesEXT;

  typedef struct VkPhysicalDeviceOpacityMicromapPropertiesEXT {
    VkStructureType                       sType;
    void*                                 pNext;
    uint32_t                              maxOpacity2StateSubdivisionLevel;
    uint32_t                              maxOpacity4StateSubdivisionLevel;
  } VkPhysicalDeviceOpacityMicromapPropertiesEXT;

  typedef struct VkMicromapVersionInfoEXT {
    VkStructureType                       sType;
    void const*                           pNext;
    uint8_t const*                        pVersionData;
  } VkMicromapVersionInfoEXT;

  typedef struct VkCopyMicromapToMemoryInfoEXT {
    VkStructureType                       sType;
    void const*                           pNext;
    VkMicromapEXT                         src;
    VkDeviceOrHostAddressKHR              dst;
    VkCopyMicromapModeEXT                 mode;
  } VkCopyMicromapToMemoryInfoEXT;

  typedef struct VkCopyMemoryToMicromapInfoEXT {
    VkStructureType                       sType;
    void const*                           pNext;
    VkDeviceOrHostAddressConstKHR         src;
    VkMicromapEXT                         dst;
    VkCopyMicromapModeEXT                 mode;
  } VkCopyMemoryToMicromapInfoEXT;

  typedef struct VkCopyMicromapInfoEXT {
    VkStructureType                       sType;
    void const*                           pNext;
    VkMicromapEXT                         src;
    VkMicromapEXT                         dst;
    VkCopyMicromapModeEXT                 mode;
  } VkCopyMicromapInfoEXT;

  typedef enum VkMicromapCreateFlagBitsEXT {
    VK_MICROMAP_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT_EXT = 0x00000001,
    VK_MICROMAP_CREATE_FLAG_BITS_MAX_ENUM_EXT = 0x7FFFFFFF
  } VkMicromapCreateFlagBitsEXT;

  typedef struct VkMicromapBuildSizesInfoEXT {
    VkStructureType                       sType;
    void const*                           pNext;
    VkDeviceSize                          micromapSize;
    VkDeviceSize                          buildScratchSize;
    VkBool32                              discardable;
  } VkMicromapBuildSizesInfoEXT;

  typedef enum VkOpacityMicromapFormatEXT {
    VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT = 1,
    VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT = 2,
    VK_OPACITY_MICROMAP_FORMAT_MAX_ENUM_EXT = 0x7FFFFFFF
  } VkOpacityMicromapFormatEXT;

  typedef struct VkAccelerationStructureTrianglesOpacityMicromapEXT {
    VkStructureType                       sType;
    void*                                 pNext;
    VkIndexType                           indexType;
    VkDeviceOrHostAddressConstKHR         indexBuffer;
    VkDeviceSize                          indexStride;
    uint32_t                              baseTriangle;
    uint32_t                              usageCountsCount;
    VkMicromapUsageEXT const*             pUsageCounts;
    VkMicromapUsageEXT const* const*      ppUsageCounts;
    VkMicromapEXT                         micromap;
  } VkAccelerationStructureTrianglesOpacityMicromapEXT;

  typedef struct VkMicromapTriangleEXT {
    uint32_t                              dataOffset;
    uint16_t                              subdivisionLevel;
    uint16_t                              format;
  } VkMicromapTriangleEXT;

  typedef enum VkOpacityMicromapSpecialIndexEXT {
    VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_TRANSPARENT_EXT = -1,
    VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT = -2,
    VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_TRANSPARENT_EXT = -3,
    VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_OPAQUE_EXT = -4,
    VK_OPACITY_MICROMAP_SPECIAL_INDEX_MAX_ENUM_EXT = 0x7FFFFFFF
  } VkOpacityMicromapSpecialIndexEXT;

  typedef VkResult (VKAPI_PTR *PFN_vkCreateMicromapEXT)(VkDevice device, const VkMicromapCreateInfoEXT *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkMicromapEXT *pMicromap);
  typedef void (VKAPI_PTR *PFN_vkDestroyMicromapEXT)(VkDevice device, VkMicromapEXT micromap, const VkAllocationCallbacks *pAllocator);
  typedef void (VKAPI_PTR *PFN_vkCmdBuildMicromapsEXT)(VkCommandBuffer commandBuffer, uint32_t infoCount, const VkMicromapBuildInfoEXT *pInfos);
  typedef VkResult (VKAPI_PTR *PFN_vkBuildMicromapsEXT)(VkDevice device, VkDeferredOperationKHR deferredOperation, uint32_t infoCount, const VkMicromapBuildInfoEXT *pInfos);
  typedef VkResult (VKAPI_PTR *PFN_vkCopyMicromapEXT)(VkDevice device, VkDeferredOperationKHR deferredOperation, const VkCopyMicromapInfoEXT *pInfo);
  typedef VkResult (VKAPI_PTR *PFN_vkCopyMicromapToMemoryEXT)(VkDevice device, VkDeferredOperationKHR deferredOperation, const VkCopyMicromapToMemoryInfoEXT *pInfo);
  typedef VkResult (VKAPI_PTR *PFN_vkCopyMemoryToMicromapEXT)(VkDevice device, VkDeferredOperationKHR deferredOperation, const VkCopyMemoryToMicromapInfoEXT *pInfo);
  typedef VkResult (VKAPI_PTR *PFN_vkWriteMicromapsPropertiesEXT)(VkDevice device, uint32_t micromapCount, const VkMicromapEXT *pMicromaps, VkQueryType queryType, size_t dataSize, void *pData, size_t stride);
  typedef void (VKAPI_PTR *PFN_vkCmdCopyMicromapEXT)(VkCommandBuffer commandBuffer, const VkCopyMicromapInfoEXT *pInfo);
  typedef void (VKAPI_PTR *PFN_vkCmdCopyMicromapToMemoryEXT)(VkCommandBuffer commandBuffer, const VkCopyMicromapToMemoryInfoEXT *pInfo);
  typedef void (VKAPI_PTR *PFN_vkCmdCopyMemoryToMicromapEXT)(VkCommandBuffer commandBuffer, const VkCopyMemoryToMicromapInfoEXT *pInfo);
  typedef void (VKAPI_PTR *PFN_vkCmdWriteMicromapsPropertiesEXT)(VkCommandBuffer commandBuffer, uint32_t micromapCount, const VkMicromapEXT *pMicromaps, VkQueryType queryType, VkQueryPool queryPool, uint32_t firstQuery);
  typedef void (VKAPI_PTR *PFN_vkGetDeviceMicromapCompatibilityEXT)(VkDevice device, const VkMicromapVersionInfoEXT *pVersionInfo, VkAccelerationStructureCompatibilityKHR *pCompatibility);
  typedef void (VKAPI_PTR *PFN_vkGetMicromapBuildSizesEXT)(VkDevice device, VkAccelerationStructureBuildTypeKHR buildType, const VkMicromapBuildInfoEXT *pBuildInfo, VkMicromapBuildSizesInfoEXT *pSizeInfo);

#ifndef VK_NO_PROTOTYPES
  VKAPI_ATTR VkResult VKAPI_CALL vkCreateMicromapEXT(
    VkDevice                                    device,
    VkMicromapCreateInfoEXT const*              pCreateInfo,
    VkAllocationCallbacks const*                pAllocator,
    VkMicromapEXT*                              pMicromap);

  VKAPI_ATTR void VKAPI_CALL vkDestroyMicromapEXT(
    VkDevice                                    device,
    VkMicromapEXT                               micromap,
    VkAllocationCallbacks const*                pAllocator);

  VKAPI_ATTR void VKAPI_CALL vkCmdBuildMicromapsEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    infoCount,
    VkMicromapBuildInfoEXT const*               pInfos);

  VKAPI_ATTR VkResult VKAPI_CALL vkBuildMicromapsEXT(
    VkDevice                                    device,
    VkDeferredOperationKHR                      deferredOperation,
    uint32_t                                    infoCount,
    VkMicromapBuildInfoEXT const*               pInfos);

  VKAPI_ATTR VkResult VKAPI_CALL vkCopyMicromapEXT(
    VkDevice                                    device,
    VkDeferredOperationKHR                      deferredOperation,
    VkCopyMicromapInfoEXT const*                pInfo);

  VKAPI_ATTR VkResult VKAPI_CALL vkCopyMicromapToMemoryEXT(
    VkDevice                                    device,
    VkDeferredOperationKHR                      deferredOperation,
    VkCopyMicromapToMemoryInfoEXT const*        pInfo);

  VKAPI_ATTR VkResult VKAPI_CALL vkCopyMemoryToMicromapEXT(
    VkDevice                                    device,
    VkDeferredOperationKHR                      deferredOperation,
    VkCopyMemoryToMicromapInfoEXT const*        pInfo);

  VKAPI_ATTR VkResult VKAPI_CALL vkWriteMicromapsPropertiesEXT(
    VkDevice                                    device,
    uint32_t                                    micromapCount,
    VkMicromapEXT const*                        pMicromaps,
    VkQueryType                                 queryType,
    size_t                                      dataSize,
    void*                                       pData,
    size_t                                      stride);

  VKAPI_ATTR void VKAPI_CALL vkCmdCopyMicromapEXT(
    VkCommandBuffer                             commandBuffer,
    VkCopyMicromapInfoEXT const*                pInfo);

  VKAPI_ATTR void VKAPI_CALL vkCmdCopyMicromapToMemoryEXT(
    VkCommandBuffer                             commandBuffer,
    VkCopyMicromapToMemoryInfoEXT const*        pInfo);

  VKAPI_ATTR void VKAPI_CALL vkCmdCopyMemoryToMicromapEXT(
    VkCommandBuffer                             commandBuffer,
    VkCopyMemoryToMicromapInfoEXT const*        pInfo);

  VKAPI_ATTR void VKAPI_CALL vkCmdWriteMicromapsPropertiesEXT(
    VkCommandBuffer                             commandBuffer,
    uint32_t                                    micromapCount,
    VkMicromapEXT const*                        pMicromaps,
    VkQueryType                                 queryType,
    VkQueryPool                                 queryPool,
    uint32_t                                    firstQuery);

  VKAPI_ATTR void VKAPI_CALL vkGetDeviceMicromapCompatibilityEXT(
    VkDevice                                    device,
    VkMicromapVersionInfoEXT const*             pVersionInfo,
    VkAccelerationStructureCompatibilityKHR*    pCompatibility);

  VKAPI_ATTR void VKAPI_CALL vkGetMicromapBuildSizesEXT(
    VkDevice                                    device,
    VkAccelerationStructureBuildTypeKHR         buildType,
    VkMicromapBuildInfoEXT const*               pBuildInfo,
    VkMicromapBuildSizesInfoEXT*                pSizeInfo);
#endif
#endif

#ifndef VK_NV_displacement_micromap
#define VULKAN_NV_DEFINED_NV_displacement_micromap 1 // Defined if vulkan_core.h doesn't include VK_EXT_opacity_micromap

#define VK_NV_displacement_micromap 1
#define VK_NV_DISPLACEMENT_MICROMAP_SPEC_VERSION                     1
#define VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME                   "VK_NV_displacement_micromap"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES_NV ((VkStructureType)1000397000)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_PROPERTIES_NV ((VkStructureType)1000397001)
#define VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_DISPLACEMENT_MICROMAP_NV ((VkStructureType)1000397002)
#define VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV  ((VkPipelineCreateFlagBits)0x10000000)
#define VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DISPLACEMENT_MICROMAP_UPDATE_NV ((VkBuildAccelerationStructureFlagBitsKHR)0x00000200)
#define VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DISPLACEMENT_MICROMAP_INSTANCE_NV ((VkBuildAccelerationStructureFlagBitsKHR)0x00000400)
#define VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV                    ((VkMicromapTypeEXT)1000397000)

  typedef struct VkPhysicalDeviceDisplacementMicromapFeaturesNV {
    VkStructureType                       sType;
    void*                                 pNext;
    VkBool32                              displacementMicromap;
  } VkPhysicalDeviceDisplacementMicromapFeaturesNV;

  typedef struct VkPhysicalDeviceDisplacementMicromapPropertiesNV {
    VkStructureType                       sType;
    void*                                 pNext;
    uint32_t                              maxDisplacementMicromapSubdivisionLevel;
  } VkPhysicalDeviceDisplacementMicromapPropertiesNV;

  typedef struct VkAccelerationStructureTrianglesDisplacementMicromapNV {
    VkStructureType                       sType;
    void*                                 pNext;
    VkFormat                              displacementBiasAndScaleFormat;
    VkFormat                              displacementVectorFormat;
    VkDeviceOrHostAddressConstKHR         displacementBiasAndScaleBuffer;
    VkDeviceSize                          displacementBiasAndScaleStride;
    VkDeviceOrHostAddressConstKHR         displacementVectorBuffer;
    VkDeviceSize                          displacementVectorStride;
    VkDeviceOrHostAddressConstKHR         displacedMicromapPrimitiveFlags;
    VkDeviceSize                          displacedMicromapPrimitiveFlagsStride;
    VkIndexType                           indexType;
    VkDeviceOrHostAddressConstKHR         indexBuffer;
    VkDeviceSize                          indexStride;
    uint32_t                              baseTriangle;
    uint32_t                              usageCountsCount;
    VkMicromapUsageEXT const*             pUsageCounts;
    VkMicromapUsageEXT const* const*      ppUsageCounts;
    VkMicromapEXT                         micromap;
  } VkAccelerationStructureTrianglesDisplacementMicromapNV;

  typedef enum VkDisplacementMicromapFormatNV {
    VK_DISPLACEMENT_MICROMAP_FORMAT_64_TRIANGLES_64_BYTES_NV = 1,
    VK_DISPLACEMENT_MICROMAP_FORMAT_256_TRIANGLES_128_BYTES_NV = 2,
    VK_DISPLACEMENT_MICROMAP_FORMAT_1024_TRIANGLES_128_BYTES_NV = 3,
    VK_DISPLACEMENT_MICROMAP_FORMAT_MAX_ENUM_NV = 0x7FFFFFFF
  } VkDisplacementMicromapFormatNV;
#endif

// clang-format on

#ifdef __cplusplus
}
#endif