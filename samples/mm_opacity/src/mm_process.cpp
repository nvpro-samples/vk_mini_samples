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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#define _USE_MATH_DEFINES
#include <map>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp>  // Perlin noise

#include "nvmath/nvmath.h"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "nvh/parallel_work.hpp"

#include "mm_process.hpp"
#include "bird_curve_helper.hpp"
#include "bit_packer.hpp"
#include "nesting_scoped_timer.hpp"
#include "nvh/alignment.hpp"
#include <array>

MicromapProcess::MicromapProcess(nvvk::Context* ctx, nvvk::ResourceAllocator* allocator)
    : m_alloc(allocator)
    , m_device(ctx->m_device)
{
  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &m_oppacityProps;
  vkGetPhysicalDeviceProperties2(ctx->m_physicalDevice, &prop2);
}

MicromapProcess::~MicromapProcess()
{
  m_alloc->destroy(m_inputData);
  m_alloc->destroy(m_microData);
  m_alloc->destroy(m_trianglesBuffer);
  m_alloc->destroy(m_scratchBuffer);
  m_alloc->destroy(m_indexBuffer);
  vkDestroyMicromapEXT(m_device, m_micromap, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Create the data for displacement
// - Get a vector of displacement values per triangle
// - Pack the data to 11 bit (64_TRIANGLES_64_BYTES format)
// - Get the usage
// - Create the vector of VkMicromapTriangleEXT
bool MicromapProcess::createMicromapData(VkCommandBuffer cmd, const nvh::PrimitiveMesh& mesh, uint16_t subdivLevel, float radius, uint16_t micromapFormat)
{
  NestingScopedTimer stimer("Create Micromap Data");

  vkDestroyMicromapEXT(m_device, m_micromap, nullptr);
  m_alloc->destroy(m_scratchBuffer);
  m_alloc->destroy(m_inputData);
  m_alloc->destroy(m_microData);
  m_alloc->destroy(m_trianglesBuffer);
  m_alloc->destroy(m_indexBuffer);

  // Get an array of displacement per triangle
  MicroOpacity micro_dist = createOpacity(mesh, subdivLevel, radius);

  // Number of triangles in the mesh and number of micro-triangles in a triangle
  const auto num_tri       = static_cast<uint32_t>(micro_dist.rawTriangles.size());
  const auto num_micro_tri = BirdCurveHelper::getNumMicroTriangles(subdivLevel);

  // Micromesh Usage
  {
    // The usage is like an histogram; how many triangles, using a `format` and a `subdivisionLevel`.
    // Since all our triangles have the same subdivision level, and the same storage format, there is only
    // one usage.
    m_usages.resize(1);
    m_usages[0].count            = num_tri;
    m_usages[0].format           = micromapFormat;
    m_usages[0].subdivisionLevel = subdivLevel;
  }

  // Can store 8 triangle info per byte for VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT
  uint32_t storage_byte = (num_micro_tri + 7) / 8;
  if(micromapFormat == VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT)
  {
    storage_byte *= 2;  // Need twice as much for the 4 state
  }

  // Micromesh Input Values
  {
    // Allocate the array to push on the GPU.
    std::vector<uint8_t> packed_data(storage_byte * num_tri);
    memset(packed_data.data(), 0U, storage_byte * num_tri * sizeof(uint8_t));

    // Loop over all triangles of the mesh
    for(uint32_t tri_index = 0U; tri_index < num_tri; tri_index++)
    {
      // The offset from the start of packed_data, must be a multiple of 64 bit
      uint32_t offset = storage_byte * tri_index;

      // Access to all displacement values
      const std::vector<int>& values = micro_dist.rawTriangles[tri_index].values;

      // The BitPacker will store contiguously unorm11 (float normalized on 11 bit), from the beginning of the
      // triangle (offset), plus each extra block
      BitPacker packer(&packed_data[offset]);

      // Loop for all block of 64 triangles
      for(const auto& value : values)
      {
        if(micromapFormat == VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT)
        {
          if(value == VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_TRANSPARENT_EXT)
          {
            packer.push(0, 1);
          }
          else
          {
            packer.push(1, 1);
          }
        }
        else
        {
          if(value == VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_TRANSPARENT_EXT)
          {
            packer.push(0, 2);
          }
          else if(value == VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT)
          {
            packer.push(1, 2);
          }
          else
          {
            packer.push(3, 2);
          }
        }
      }
    }

    m_inputData = m_alloc->createBuffer(
        cmd, packed_data, VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  }

  // Micromap Triangle
  {
    std::vector<VkMicromapTriangleEXT> micromap_triangles;
    micromap_triangles.reserve(num_tri);
    for(uint32_t tri_index = 0; tri_index < num_tri; tri_index++)
    {
      uint32_t offset = storage_byte * tri_index;  // Same offset as when storing the data
      micromap_triangles.push_back({offset, subdivLevel, micromapFormat});
    }
    m_trianglesBuffer = m_alloc->createBuffer(cmd, micromap_triangles,
                                              VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT
                                                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  }

  // Index buffer: referencing the Micromap Triangle buffer
  {
    std::vector<uint32_t> index(num_tri);
    int                   cnt{0};
    for(auto& i : index)
    {
      i = cnt++;
    }
    m_indexBuffer = m_alloc->createBuffer(
        cmd, index, VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  }


  barrier(cmd);

  buildMicromap(cmd, VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT);

  return true;
}

//--------------------------------------------------------------------------------------------------
// Building the micromap using: triangle data, input data (values), usage
//
bool MicromapProcess::buildMicromap(VkCommandBuffer cmd, VkMicromapTypeEXT micromapType)
{
  NestingScopedTimer stimer("Build Micromap");

  // Find the size required
  VkMicromapBuildSizesInfoEXT size_info{VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT};
  VkMicromapBuildInfoEXT      build_info{VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
  build_info.mode             = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
  build_info.usageCountsCount = static_cast<uint32_t>(m_usages.size());
  build_info.pUsageCounts     = m_usages.data();
  build_info.type             = micromapType;  // Opacity
  vkGetMicromapBuildSizesEXT(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &size_info);
  assert(size_info.micromapSize && "sizeInfo.micromeshSize was zero");

  // create micromeshData buffer
  m_microData = m_alloc->createBuffer(size_info.micromapSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                  | VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT);

  uint64_t scratch_size = std::max(size_info.buildScratchSize, static_cast<VkDeviceSize>(4));
  m_scratchBuffer =
      m_alloc->createBuffer(scratch_size, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT);

  // Create micromap
  VkMicromapCreateInfoEXT mm_create_info{VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT};
  mm_create_info.buffer = m_microData.buffer;
  mm_create_info.size   = size_info.micromapSize;
  mm_create_info.type   = micromapType;
  NVVK_CHECK(vkCreateMicromapEXT(m_device, &mm_create_info, nullptr, &m_micromap));

  {
    // Fill in the pointers we didn't have at size query
    build_info.dstMicromap                 = m_micromap;
    build_info.scratchData.deviceAddress   = nvvk::getBufferDeviceAddress(m_device, m_scratchBuffer.buffer);
    build_info.data.deviceAddress          = nvvk::getBufferDeviceAddress(m_device, m_inputData.buffer);
    build_info.triangleArray.deviceAddress = nvvk::getBufferDeviceAddress(m_device, m_trianglesBuffer.buffer);
    build_info.triangleArrayStride         = sizeof(VkMicromapTriangleEXT);
    vkCmdBuildMicromapsEXT(cmd, 1, &build_info);
  }
  barrier(cmd);

  return true;
}

//--------------------------------------------------------------------------------------------------
// This can be called when the Micromap has been build
//
void MicromapProcess::cleanBuildData()
{
  m_alloc->destroy(m_scratchBuffer);
  m_alloc->destroy(m_inputData);
  m_alloc->destroy(m_trianglesBuffer);
}


//--------------------------------------------------------------------------------------------------
// Make sure all the data are ready before building the micromap
void MicromapProcess::barrier(VkCommandBuffer cmd)
{
  // barrier for upload finish
  VkMemoryBarrier2 mem_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,         nullptr,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,           VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT, VK_ACCESS_2_MICROMAP_READ_BIT_EXT};
  VkDependencyInfo dep_info{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep_info.memoryBarrierCount = 1;
  dep_info.pMemoryBarriers    = &mem_barrier;
  vkCmdPipelineBarrier2(cmd, &dep_info);
}

// Intersecting of a triangle and a circle
// Return 2 when triangle is within the circle
// Return 1 when triangle intersect, point, edge or surface
// Return 0 when it is totally outside
uint32_t triangleCircleItersection(const std::array<nvmath::vec3f, 3>& p, const nvmath::vec3f& center, float radius)
{
  float radiusSqr = radius * radius;

  // Vertices within circle
  int           hit = 0;
  nvmath::vec3f c[3];
  float         csqr[3];
  for(int i = 0; i < 3; i++)
  {
    c[i]    = center - p[i];
    csqr[i] = nvmath::dot(c[i], c[i]) - radiusSqr;
    hit += csqr[i] <= 0 ? 1 : 0;
  }

  if(hit == 3)
    return 2;  // Completely inside the circle

  if(hit > 0)
    return 1;  // Circle crossing the triangle

  nvmath::vec3f edges[3];
  edges[0] = p[1] - p[0];
  edges[1] = p[2] - p[1];
  edges[2] = p[0] - p[2];

  // Circle is within triangle? (not happening, discard)
  //hit = 0;
  //nvmath::vec3f n[3];
  //for(int i = 0; i < 3; i++)
  //{
  //  n[i] = nvmath::cross(edges[i], c[i]);
  //  hit += n[i].y > 0 ? 1 : 0;
  //}
  //if(hit == 3)
  //{
  //  return 1;  // Circle cover partly the triangle
  //}


  // Circle intersects edges
  float k[3];
  for(int i = 0; i < 3; i++)
  {
    k[i] = nvmath::dot(edges[i], c[i]);
    if(k[i] > 0)
    {
      float len = nvmath::dot(edges[i], edges[i]);  // squared len

      if(k[i] < len)
      {
        if(csqr[i] * len <= k[i] * k[i])
          return 1;
      }
    }
  }

  // Triangle outside circle
  return 0;
}


//--------------------------------------------------------------------------------------------------
// Set the visibility information per micro-triangle.
// - A micro triangle will be fully opaque if all its position are within the `radius`.
//   fully transparent when all its position are outside and unknown if one position crosses
//   the radius boundary.
MicromapProcess::MicroOpacity MicromapProcess::createOpacity(const nvh::PrimitiveMesh& mesh, uint16_t subdivLevel, float radius)
{
  NestingScopedTimer stimer("Create Displacements");

  MicroOpacity displacements;  // Return of displacement values for all triangles

  const auto num_micro_tri = BirdCurveHelper::getNumMicroTriangles(subdivLevel);

  auto num_tri = static_cast<uint32_t>(mesh.indices.size() / 3);
  displacements.rawTriangles.resize(num_tri);

  const nvmath::vec3f center{0.0F, 0.0F, 0.0F};

  // Find the distances in parallel
  // Faster than : for(size_t tri_index = 0; tri_index < num_tri; tri_index++)
  nvh::parallel_batches<32>(
      num_tri,
      [&](uint64_t tri_index) {
        // Retrieve the positions of the triangle
        nvmath::vec3f t0 = mesh.vertices[mesh.indices[tri_index * 3 + 0]].p;
        nvmath::vec3f t1 = mesh.vertices[mesh.indices[tri_index * 3 + 1]].p;
        nvmath::vec3f t2 = mesh.vertices[mesh.indices[tri_index * 3 + 2]].p;

        // Working on this triangle
        RawTriangle& triangle = displacements.rawTriangles[tri_index];
        triangle.values.resize(num_micro_tri);
        triangle.subdivLevel = subdivLevel;

        // TODO: check if the triangle is completely in or out to avoid subdividing it
        uint32_t hit = triangleCircleItersection({t0, t1, t2}, center, radius);

        for(uint32_t index = 0; index < num_micro_tri; index++)
        {
          // Utility to get the barycentric values
          nvmath::vec3f uv0, uv1, uv2;
          BirdCurveHelper::micro2bary(index, subdivLevel, uv0, uv1, uv2);

          // The sub-triangle position
          nvmath::vec3f p0 = getInterpolated(t0, t1, t2, uv0);
          nvmath::vec3f p1 = getInterpolated(t0, t1, t2, uv1);
          nvmath::vec3f p2 = getInterpolated(t0, t1, t2, uv2);

          // Check how many sub-triangle vertex are within the radius
          uint32_t hit = triangleCircleItersection({p0, p1, p2}, center, radius);

          // Determining the visibility of the triangle
          switch(hit)
          {
            case 2:
              triangle.values[index] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_OPAQUE_EXT;
              break;
            case 0:
              triangle.values[index] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_TRANSPARENT_EXT;
              break;
            default:
              triangle.values[index] = VK_OPACITY_MICROMAP_SPECIAL_INDEX_FULLY_UNKNOWN_TRANSPARENT_EXT;
              break;
          }
        }
      },
      std::thread::hardware_concurrency());

  return displacements;
}
