/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


// Shader header for inspection shader variables
// Prior to including this header the following macros need to be defined
// Either INSPECTOR_MODE_COMPUTE or INSPECTOR_MODE_FRAGMENT
// If INSPECTOR_MODE_COMPUTE is defined the shader must expose invocation information (e.g. gl_LocalInvocationID).
// This typically applies to compute, task and mesh shaders
// If INSPECTOR_MODE_FRAGMENT is defined the shader must expose fragment information (e.g. gl_FragCoord).
// This applies to fragment shaders
//
// INSPECTOR_DESCRIPTOR_SET: the index of the descriptor set containing the Inspector buffers
// INSPECTOR_INSPECTION_DATA_BINDING: the binding index of the buffer containing the inspection information, as provided by ElementInspector::getComputeInspectionBuffer()
// INSPECTOR_METADATA_BINDING: the binding index of the buffer containing the inspection metadata, as provided by ElementInspector::getComputeMetadataBuffer()

#ifndef DH_INSPECTOR_H
#define DH_INSPECTOR_H

#define WARP_SIZE 32

#define WARP_2D_SIZE_X 8
#define WARP_2D_SIZE_Y 4
#define WARP_2D_SIZE_Z 1

struct InspectorComputeMetadata
{
  uint3 minBlock;
  uint u32PerThread;
  uint3 maxBlock;
  uint minWarpInBlock;
  uint maxWarpInBlock;
};

struct InspectorFragmentMetadata
{
  uint2 minFragment;
  uint2 maxFragment;
  uint2 renderSize;
  uint u32PerThread;
};

struct InspectorCustomMetadata
{
  uint3 minCoord;
  uint pad0;
  uint3 maxCoord;
  uint pad1;
  uint3 extent;
  uint u32PerThread;
};


#if !(defined INSPECTOR_MODE_COMPUTE) && !(defined INSPECTOR_MODE_FRAGMENT) && !(defined INSPECTOR_MODE_CUSTOM)
#error At least one inspector mode (INSPECTOR_MODE_COMPUTE, INSPECTOR_MODE_FRAGMENT, INSPECTOR_MODE_CUSTOM) must be defined before including this file
#endif

#if(defined INSPECTOR_MODE_COMPUTE) && (defined INSPECTOR_MODE_FRAGMENT)
#error Only one of inspector modes INSPECTOR_MODE_COMPUTE, INSPECTOR_MODE_FRAGMENT can be chosen
#endif

#ifndef INSPECTOR_DESCRIPTOR_SET
#error The descriptor set containing thread inspection data must be provided before including this file
#endif


#ifdef INSPECTOR_MODE_CUSTOM
#ifndef INSPECTOR_CUSTOM_INSPECTION_DATA_BINDING
#error The descriptor binding containing custom thread inspection data must be provided before including this file
#endif
#ifndef INSPECTOR_CUSTOM_METADATA_BINDING
#error The descriptor binding containing custom thread inspection metadata must be provided before including this file
#endif
#endif

#if(defined INSPECTOR_MODE_COMPUTE) || (defined INSPECTOR_MODE_FRAGMENT)
#ifndef INSPECTOR_INSPECTION_DATA_BINDING
#error The descriptor binding containing thread inspection data must be provided before including this file
#endif

#ifndef INSPECTOR_METADATA_BINDING
#error The descriptor binding containing thread inspection metadata must be provided before including this file
#endif
#endif

#ifdef INSPECTOR_MODE_COMPUTE
[[vk::binding(INSPECTOR_INSPECTION_DATA_BINDING, INSPECTOR_DESCRIPTOR_SET)]]  RWStructuredBuffer<uint> inspectorInspectionData;
[[vk::binding(INSPECTOR_METADATA_BINDING, INSPECTOR_DESCRIPTOR_SET)]]  StructuredBuffer<InspectorComputeMetadata> inspectorMetadata;

void inspect32BitValue(uint index, uint v)
{
  if (any(clamp(gl_WorkGroupID, inspectorMetadata[0].minBlock, inspectorMetadata[0].maxBlock) != gl_WorkGroupID))
  {
    return;
  }

  uint warpIndex = gl_SubgroupID;

  if (warpIndex < inspectorMetadata[0].minWarpInBlock || warpIndex > inspectorMetadata[0].maxWarpInBlock)
    return;
  uint inspectedThreadsPerBlock = (inspectorMetadata[0].maxWarpInBlock - inspectorMetadata[0].minWarpInBlock + 1) * gl_SubgroupSize;
  
  uint blockIndex = gl_WorkGroupID.x + gl_NumWorkGroups.x * (gl_WorkGroupID.y + gl_NumWorkGroups.y * gl_WorkGroupID.z);
  uint minBlockIndex =
      inspectorMetadata[0].minBlock.x
      + gl_NumWorkGroups.x * (inspectorMetadata[0].minBlock.y + gl_NumWorkGroups.y * inspectorMetadata[0].minBlock.z);

  uint blockStart = inspectedThreadsPerBlock * (blockIndex - minBlockIndex) * inspectorMetadata[0].u32PerThread;
  uint warpStart = (warpIndex - inspectorMetadata[0].minWarpInBlock) * inspectorMetadata[0].u32PerThread * gl_SubgroupSize;
  uint threadInWarpStart = gl_SubgroupInvocationID * inspectorMetadata[0].u32PerThread;

  inspectorInspectionData[blockStart + warpStart + threadInWarpStart + index] = v;
}

#endif

#ifdef INSPECTOR_MODE_FRAGMENT

[[vk::binding(INSPECTOR_INSPECTION_DATA_BINDING, INSPECTOR_DESCRIPTOR_SET)]]  RWByteAddressBuffer inspectorInspectionData;
[[vk::binding(INSPECTOR_METADATA_BINDING, INSPECTOR_DESCRIPTOR_SET)]]  StructuredBuffer<InspectorFragmentMetadata> inspectorMetadata;

void inspect32BitValue(uint index, uint v)
{
  uint2 fragment = uint2(floor(gl_FragCoord.xy));

  if(any(clamp(fragment, inspectorMetadata[0].minFragment, inspectorMetadata[0].maxFragment) != fragment))
  {
    return;
  }
  
  uint2 localFragment = fragment - inspectorMetadata[0].minFragment;
  uint inspectionWidth = inspectorMetadata[0].maxFragment.x - inspectorMetadata[0].minFragment.x + 1;
  uint fragmentIndex = localFragment.x + inspectionWidth * localFragment.y;

  // Atomically store the fragment depth along with the value so we always keep the fragment value
  // with the lowest depth
  float z = 1.f - clamp(gl_FragCoord.z, 0.f, 1.f);
  uint64_t zUint = asuint(z);
  uint64_t value = (zUint << 32) | uint64_t(v);
  uint dataIndex = fragmentIndex * inspectorMetadata[0].u32PerThread / 2 + index;
  uint byteAddress = dataIndex * 8;
  inspectorInspectionData.InterlockedMaxU64(byteAddress, value);
}

#endif

#ifdef INSPECTOR_MODE_CUSTOM

layout(set = INSPECTOR_DESCRIPTOR_SET, binding = INSPECTOR_CUSTOM_INSPECTION_DATA_BINDING) buffer InspectorCustomInspections
{
  uint32_t inspectorCustomInspection[];
};

layout(set = INSPECTOR_DESCRIPTOR_SET, binding = INSPECTOR_CUSTOM_METADATA_BINDING) readonly buffer InspectorCustomInspectionMetadata
{
  InspectorCustomMetadata inspectorCustomMetadata;
};


void inspectCustom32BitValue(uint32_t index, uvec3 location, uint32_t v)
{
  if(clamp(location, inspectorCustomMetadata.minCoord, inspectorCustomMetadata.maxCoord) != location)
  {
    return;
  }


  uvec3    localCoord       = location - inspectorCustomMetadata.minCoord;
  uint32_t inspectionWidth  = inspectorCustomMetadata.maxCoord.x - inspectorCustomMetadata.minCoord.x + 1;
  uint32_t inspectionHeight = inspectorCustomMetadata.maxCoord.y - inspectorCustomMetadata.minCoord.y + 1;
  uint32_t coordIndex       = localCoord.x + inspectionWidth * (localCoord.y + inspectionHeight * localCoord.z);

  inspectorCustomInspection[coordIndex * inspectorCustomMetadata.u32PerThread + index] = v;
}


#endif


#ifdef __cplusplus
}  // namespace nvvkhl_shaders
#endif

#endif  // DH_INSPECTOR_H