/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define _USE_MATH_DEFINES
#include <map>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp>  // Perlin noise
#include "dmm_process.hpp"
#include "bird_curve_helper.hpp"
#include "bit_packer.hpp"
#include "nvh/parallel_work.hpp"
#include "nvh/timesampler.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "vk_nv_micromesh.h"

MicromapProcess::MicromapProcess(nvvk::Context* ctx, nvvk::ResourceAllocator* allocator)
    : m_device(ctx->m_device)
    , m_alloc(allocator)
{
}

MicromapProcess::~MicromapProcess()
{
  m_alloc->destroy(m_inputData);
  m_alloc->destroy(m_microData);
  m_alloc->destroy(m_trianglesBuffer);
  m_alloc->destroy(m_primitiveFlags);
  m_alloc->destroy(m_displacementDirections);
  m_alloc->destroy(m_displacementBiasAndScale);
  m_alloc->destroy(m_scratchBuffer);
  vkDestroyMicromapEXT(m_device, m_micromap, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Create the data for displacement
// - Get a vector of displacement values per triangle
// - Pack the data to 11 bit (64_TRIANGLES_64_BYTES format)
// - Get the usage
// - Create the vector of VkMicromapTriangleEXT
bool MicromapProcess::createMicromapData(VkCommandBuffer cmd, const nvh::PrimitiveMesh& mesh, uint16_t subdivLevel, const Terrain& terrain)
{
  nvh::ScopedTimer stimer("Create Micromap Data");

  vkDestroyMicromapEXT(m_device, m_micromap, nullptr);
  m_alloc->destroy(m_scratchBuffer);
  m_alloc->destroy(m_inputData);
  m_alloc->destroy(m_microData);
  m_alloc->destroy(m_trianglesBuffer);

  // Get an array of displacement per triangle
  MicroDistances micro_dist = createDisplacements(mesh, subdivLevel, terrain);

  const auto num_tri = static_cast<uint32_t>(micro_dist.rawTriangles.size());


  // This is for VK_DISPLACEMENT_MICROMAP_FORMAT_64_TRIANGLES_64_BYTES_NV: uncompressed data but packed.
  // In this sample, it is the only supported format, as it other format requires compression and is out
  // of scope for this simple version. The micro-mesh SDK have functions that helps compressing and
  // optimize the data.
  if(subdivLevel <= 5)
  {

    // Find the displacement blocks for the subdivision. There is only one displacement block for level0..3,
    // as up to 64 triangles can fit. For higher levels, the triangle need to be subdivided in blocks of 64
    // triangles, therefore level-4 have 4 blocks and level-5 have 16 blocks. Each block contains the
    // indices the Bird-Curve subdivided sub-triangle. The indices refer to the indices of the subdivided
    // triangle.
    BirdCurveHelper                     barycentrics(subdivLevel);
    BirdCurveHelper::DisplacementBlocks blocks     = barycentrics.createDisplacementBlocks(subdivLevel);
    uint32_t                            num_blocks = static_cast<uint32_t>(blocks.size());

    {
      // Allocate the array to push on the GPU. This correspond to 64 bytes per triangle * number of displacement blocks
      std::vector<uint8_t> packed_data(64ULL * num_tri * num_blocks);

      // Loop over all triangles of the mesh
      for(uint32_t tri_index = 0U; tri_index < num_tri; tri_index++)
      {
        // The offset from the start of packed_data, must be a multiple of 64 bit
        uint32_t offset = 64U * tri_index * num_blocks;

        // Access to all displacement values
        const std::vector<float>& values = micro_dist.rawTriangles[tri_index].values;

        // Loop for all block of 64 triangles
        for(uint32_t block_idx = 0U; block_idx < num_blocks; block_idx++)
        {
          // The BitPacker will store contiguously unorm11 (float normalized on 11 bit), from the beginning of the
          // triangle (offset), plus each extra block
          BitPacker11 packer11(&packed_data[offset + 64U * block_idx]);

          // Get the number of indices in the Block. Subdivision Level 3 and up will always have
          // 45 sub-triangle indices, and less for lower subdivision levels
          uint32_t num_tri_idx = static_cast<uint32_t>(blocks[block_idx].size());

          // Each block stores displacements for up to 45 micro-vertices. Find the value index within
          // the base triangle that corresponds to the barycentric location within the current block.
          for(uint32_t block_tri_idx = 0U; block_tri_idx < num_tri_idx; block_tri_idx++)
          {
            uint32_t value_idx = blocks[block_idx][block_tri_idx];
            packer11.push(values[value_idx]);
          }
        }
      }

      m_inputData = m_alloc->createBuffer(
          cmd, packed_data, VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    }


    // Micromap Triangle
    {
      // Each triangle is stored every 64 bytes * number of displacement blocks, see above, and all are using the same subdivision level
      std::vector<VkMicromapTriangleEXT> micromap_triangles;
      micromap_triangles.reserve(num_tri);
      for(uint32_t tri_index = 0; tri_index < num_tri; tri_index++)
      {
        uint32_t offset = 64U * tri_index * num_blocks;  // Same offset as when storing the data
        micromap_triangles.push_back({offset, subdivLevel, VK_DISPLACEMENT_MICROMAP_FORMAT_64_TRIANGLES_64_BYTES_NV});
      }
      m_trianglesBuffer = m_alloc->createBuffer(cmd, micromap_triangles,
                                                VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT
                                                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    }


    // Micromesh Usage
    {
      // The usage is like an histogram; how many triangles, using a `format` and a `subdivisionLevel`.
      // Since all our triangles have the same subdivision level, and the same storage format, there is only
      // one usage.
      m_usages.resize(1);
      m_usages[0].count            = num_tri;
      m_usages[0].format           = VK_DISPLACEMENT_MICROMAP_FORMAT_64_TRIANGLES_64_BYTES_NV;
      m_usages[0].subdivisionLevel = subdivLevel;
    }
  }

  barrier(cmd);

  buildMicromap(cmd);

  return true;
}

//--------------------------------------------------------------------------------------------------
// Building the micromap using: triangle data, input data (values), usage
//
bool MicromapProcess::buildMicromap(VkCommandBuffer cmd)
{
  nvh::ScopedTimer stimer("Build Micromap");

  // Find the size required
  VkMicromapBuildSizesInfoEXT size_info{VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT};
  VkMicromapBuildInfoEXT      build_info{VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
  build_info.mode             = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
  build_info.usageCountsCount = static_cast<uint32_t>(m_usages.size());
  build_info.pUsageCounts     = m_usages.data();
  build_info.type             = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV;  // Displacement
  vkGetMicromapBuildSizesEXT(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &size_info);
  assert(size_info.micromapSize && "sizeInfo.micromeshSize was zero");

  // create micromeshData buffer
  m_microData = m_alloc->createBuffer(size_info.micromapSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                  | VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT);

  uint64_t scratch_size = std::max(size_info.buildScratchSize, static_cast<VkDeviceSize>(4));
  m_scratchBuffer =
      m_alloc->createBuffer(scratch_size, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT);

  // Create micromesh
  VkMicromapCreateInfoEXT mm_create_info{VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT};
  mm_create_info.buffer = m_microData.buffer;
  mm_create_info.size   = size_info.micromapSize;
  mm_create_info.type   = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV;
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
//
//
void MicromapProcess::createMicromapBuffers(VkCommandBuffer cmd, const nvh::PrimitiveMesh& mesh, const nvmath::vec2f& biasScale)
{
  m_alloc->destroy(m_primitiveFlags);
  m_alloc->destroy(m_displacementDirections);
  m_alloc->destroy(m_displacementBiasAndScale);

  auto num_tri = static_cast<uint32_t>(mesh.triangles.size());

  // Direction vectors
  {
    // We are taking the normal of the triangle as direction vectors
    using f16vec4 = glm::vec<4, glm::detail::hdata, glm::defaultp>;
    std::vector<f16vec4> temp;
    temp.reserve(mesh.vertices.size());
    for(const nvh::PrimitiveVertex& v : mesh.vertices)
    {
      temp.emplace_back(glm::detail::toFloat16(v.n.x), glm::detail::toFloat16(v.n.y), glm::detail::toFloat16(v.n.z),
                        glm::detail::toFloat16(0.0F));  // convert to a vector of half float
    }
    m_displacementDirections =
        m_alloc->createBuffer(cmd, temp, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  }


  // Direction Bounds
  {
    // Making the bias-scale uniform across all triangle vertices
    std::vector<nvmath::vec2f> bias_scale;
    bias_scale.reserve(num_tri * 3ULL);
    for(uint32_t i = 0; i < num_tri; i++)
    {
      bias_scale.emplace_back(biasScale);
      bias_scale.emplace_back(biasScale);
      bias_scale.emplace_back(biasScale);
    }
    m_displacementBiasAndScale =
        m_alloc->createBuffer(cmd, bias_scale, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  }


  // // Primitive flags [unused for now, since all triangles have the same subdivision level]
  {
    std::vector<uint8_t> primitive_flags;
    if(!primitive_flags.empty())
    {
      m_primitiveFlags = m_alloc->createBuffer(cmd, primitive_flags,
                                               VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT
                                                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    }
  }


  barrier(cmd);
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


//--------------------------------------------------------------------------------------------------
// Get the displacement values per triangle; UV distance from [0.5,0.5]
// -
MicromapProcess::MicroDistances MicromapProcess::createDisplacements(const nvh::PrimitiveMesh& mesh, uint16_t subdivLevel, const Terrain& terrain)
{
  nvh::ScopedTimer stimer("Create Displacements");

  MicroDistances displacements;  // Return of displacement values for all triangles

  // Utility to get the barycentric values
  BirdCurveHelper                         barycentrics(subdivLevel);
  const BirdCurveHelper::BaryCoordinates& bvalues = barycentrics.getVertexCoord(subdivLevel);

  auto num_tri = static_cast<uint32_t>(mesh.triangles.size());
  displacements.rawTriangles.resize(num_tri);

  // Find the distances in parallel
  // Faster than : for(size_t tri_index = 0; tri_index < num_tri; tri_index++)
  nvh::parallel_batches<32>(
      num_tri,
      [&](uint64_t tri_index) {
        // Retrieve the UV of the triangle
        nvmath::vec2f t0 = mesh.vertices[mesh.triangles[tri_index].v[0]].t;
        nvmath::vec2f t1 = mesh.vertices[mesh.triangles[tri_index].v[1]].t;
        nvmath::vec2f t2 = mesh.vertices[mesh.triangles[tri_index].v[2]].t;

        // Working on this triangle
        RawTriangle& triangle = displacements.rawTriangles[tri_index];
        triangle.values.resize(bvalues.size());
        triangle.subdivLevel = subdivLevel;

        for(size_t index = 0; index < bvalues.size(); index++)
        {
          nvmath::vec2f uv = getInterpolated(t0, t1, t2, bvalues[index]);

          // Simple perlin noise
          float v     = 0.0F;
          float scale = terrain.power;
          float freq  = terrain.freq;
          for(int oct = 0; oct < terrain.octave; oct++)
          {
            v += glm::perlin(glm::vec3(uv.x, uv.y, terrain.seed) * freq) / scale;
            freq *= 2.0F;            // Double the frequency
            scale *= terrain.power;  // Next power of b
          }

          // Adjusting the value
          triangle.values[index] = nvmath::clamp((1.0F + v) * 0.5F, 0.0F, 1.0F);
        }
      },
      std::thread::hardware_concurrency());

  return displacements;
}
