/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*

  Cluster Acceleration Structure (CLAS) sample with a simple level-of-detail (LOD).

  A sphere is generated at several levels of detail. Each level is split into small
  contiguous triangle patches (clusters). At startup we build ALL the clusters of ALL
  levels into CLAS (VK_NV_cluster_acceleration_structure) exactly once, plus one
  "cluster" bottom-level AS per level (each referencing that level's CLAS).

  Every frame we pick a level from the camera distance (or a manual override) and point
  the top-level AS at that level's cluster BLAS. Because CLAS are built once, switching
  detail is only a cheap TLAS rebuild. As you move closer the level rises and more, finer
  clusters appear.

  On a hit, the closest-hit shader reads the per-hit cluster ID exposed by the extension
  and colorizes the surface, so the cluster decomposition is directly visible.

*/
//////////////////////////////////////////////////////////////////////////

#define USE_SLANG true
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }

#include <array>
#include <vector>

#include <imgui/imgui.h>
#include <vulkan/vulkan_core.h>

#include "shaders/shaderio.h"

#include "_autogen/raytrace_clusters.slang.h"
#include "_autogen/raytrace_clusters.rgen.glsl.h"
#include "_autogen/raytrace_clusters.rchit.glsl.h"
#include "_autogen/raytrace_clusters.rmiss.glsl.h"

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/imgui_texture.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/property_editor.hpp>
#include <nvslang/slang.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/barriers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/staging.hpp>

#include "common/utils.hpp"

std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};

//////////////////////////////////////////////////////////////////////////
/// Ray tracing a clustered sphere with cluster-based LOD
class RayTraceClusters : public nvapp::IAppElement
{
  // Description of one cluster inside the concatenated vertex/index buffers
  struct ClusterInfo
  {
    uint32_t vertexByteOffset{};  // byte offset of the cluster's first vertex in m_vertexBuffer
    uint32_t indexByteOffset{};   // byte offset of the cluster's first index in m_indexBuffer (8-bit indices)
    uint32_t vertexCount{};       // number of local vertices
    uint32_t triangleCount{};     // number of triangles
  };

  // One level of detail: a contiguous range of clusters and the BLAS built from them
  struct Lod
  {
    uint32_t        clasOffset{};    // index of the level's first cluster in the global CLAS array
    uint32_t        clusterCount{};  // number of clusters at this level
    VkDeviceAddress blasAddress{};   // device address of the cluster BLAS for this level
  };

  // Sphere resolution per level (latitude x longitude subdivisions). Cluster patch is 4x4 cells,
  // so the number of clusters is (rings/4) * (sectors/4): 8, 32, 128, 512.
  static constexpr int        kPatch                = 4;
  static constexpr int        kNumLevels            = 4;
  static constexpr glm::ivec2 kLevelRes[kNumLevels] = {{8, 16}, {16, 32}, {32, 64}, {64, 128}};

public:
  RayTraceClusters()           = default;
  ~RayTraceClusters() override = default;

  void onAttach(nvapp::Application* app) override
  {
    SCOPED_TIMER(__FUNCTION__);
    m_app    = app;
    m_device = app->getDevice();

    m_alloc.init({
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });

    m_slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
    m_slangCompiler.defaultTarget();
    m_slangCompiler.defaultOptions();
    m_slangCompiler.addOption({slang::CompilerOptionName::DebugInformation, {slang::CompilerOptionValueKind::Int, 1}});

    g_cameraManip->setClipPlanes({0.1F, 100.0F});
    g_cameraManip->setLookat({2.5F, 2.0F, 3.5F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});

    NVVK_CHECK(m_renderTarget.init({.alloc = &m_alloc, .colorFormats = {VK_FORMAT_R8G8B8A8_UNORM}, .debugName = "RayTraceClusters"}));

    // Ray tracing + cluster properties (alignments used by the CLAS build)
    m_rtProp.pNext = &m_asProp;
    m_asProp.pNext = &m_clusterProp;
    VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProp};
    vkGetPhysicalDeviceProperties2(app->getPhysicalDevice(), &prop2);

    createFrameInfoBuffer();
    createScene();
    buildClusterAccelerationStructures();
    createRtxPipeline();

    m_pushConst.colorByCluster = 1;
    m_lightDir                 = glm::normalize(glm::vec3(0.5F, 1.0F, 0.7F));

    // Build the initial TLAS for the level matching the starting camera distance
    setLevel(selectLod());
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);

    m_alloc.destroyBuffer(m_frameInfo);
    m_alloc.destroyBuffer(m_vertexBuffer);
    m_alloc.destroyBuffer(m_indexBuffer);
    m_alloc.destroyBuffer(m_clasBuffer);
    m_alloc.destroyBuffer(m_clasAddressBuffer);
    m_alloc.destroyBuffer(m_clusterBlasBuffer);
    m_alloc.destroyBuffer(m_instancesBuffer);
    m_alloc.destroyBuffer(m_tlasScratch);
    m_alloc.destroyBuffer(m_sbtBuffer);
    m_alloc.destroyAcceleration(m_tlas);

    vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);

    m_viewportImage.deinit();
    m_renderTarget.deinit();
    m_alloc.deinit();
    m_app = nullptr;
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    NVVK_CHECK(m_renderTarget.update(cmd, size));
    m_viewportImage.update(m_renderTarget.getUiImageView(0));
  }

  void onUIRender() override
  {
    namespace PE = nvgui::PropertyEditor;

    if(ImGui::Begin("Settings"))
    {
      nvgui::CameraWidget(g_cameraManip);

      ImGui::SeparatorText("Level of Detail");
      ImGui::Text("Camera distance: %.2f", g_cameraManip->getDistanceToCenter());
      ImGui::Text("Active level: %d  (%u clusters)", m_currentLevel, m_numClusters);

      const char* lodItems[1 + kNumLevels] = {"Auto", "Level 0", "Level 1", "Level 2", "Level 3"};
      int         comboIndex               = m_lodOverride + 1;  // -1 (auto) -> 0
      PE::begin();
      if(PE::Combo("LOD selection", &comboIndex, lodItems, 1 + kNumLevels))
        m_lodOverride = comboIndex - 1;
      bool colorByCluster = m_pushConst.colorByCluster != 0;
      if(PE::Checkbox("Color by cluster", &colorByCluster))
        m_pushConst.colorByCluster = colorByCluster ? 1 : 0;
      PE::DragFloat3("Light direction", &m_lightDir.x, 0.01F);
      PE::end();
    }
    ImGui::End();

    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    if(ImGui::Begin("Viewport"))
    {
      ImGui::Image(m_viewportImage, ImGui::GetContentRegionAvail());
    }
    ImGui::End();
    ImGui::PopStyleVar();
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);

    // If the ray tracing pipeline failed to build (e.g. a shader compile error), skip rendering
    // instead of binding a null pipeline / empty SBT.
    if(m_rtPipeline == VK_NULL_HANDLE)
      return;

    // Switch level of detail when needed (cheap: rebuild the 1-instance TLAS only)
    const int level = selectLod();
    if(level != m_currentLevel)
      setLevel(level);

    shaderio::FrameInfo frameInfo{.projInv = glm::inverse(g_cameraManip->getPerspectiveMatrix()),
                                  .viewInv = glm::inverse(g_cameraManip->getViewMatrix())};
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &frameInfo);
    nvvk::cmdBufferMemoryBarrier(cmd, {m_frameInfo.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR});

    m_pushConst.lightDir = m_lightDir;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
    pushDescriptorSet(cmd);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);

    const VkExtent2D& size = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                      size.width, size.height, 1);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
  }

private:
  //--------------------------------------------------------------------------------------------------
  // Choose a level of detail: manual override if set, otherwise from the camera distance.
  //
  int selectLod() const
  {
    if(m_lodOverride >= 0)
      return std::min(m_lodOverride, kNumLevels - 1);

    const double d = g_cameraManip->getDistanceToCenter();
    if(d > 6.0)
      return 0;
    if(d > 4.0)
      return 1;
    if(d > 2.5)
      return 2;
    return 3;
  }

  //--------------------------------------------------------------------------------------------------
  // Append one sphere LOD level to the global vertex/index/cluster arrays. Returns the number of
  // clusters generated. Each 4x4-cell patch becomes one cluster with its own local vertices and
  // 8-bit indices; degenerate triangles at the poles are skipped.
  //
  uint32_t appendSphereLevel(int rings, int sectors)
  {
    const float radius = 1.0F;

    auto spherePos = [&](int i, int j) -> glm::vec3 {
      const float theta = glm::pi<float>() * float(i) / float(rings);           // 0..pi
      const float phi   = 2.0F * glm::pi<float>() * float(j) / float(sectors);  // 0..2pi
      return radius * glm::vec3(std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi));
    };

    const uint32_t clustersBefore = uint32_t(m_clusters.size());

    for(int pi = 0; pi < rings; pi += kPatch)
    {
      for(int pj = 0; pj < sectors; pj += kPatch)
      {
        const int i0 = pi, i1 = std::min(pi + kPatch, rings);
        const int j0 = pj, j1 = std::min(pj + kPatch, sectors);
        const int nCols = (j1 - j0) + 1;

        std::vector<glm::vec3> localVerts;
        for(int i = i0; i <= i1; i++)
          for(int j = j0; j <= j1; j++)
            localVerts.push_back(spherePos(i, j));

        auto localIndex = [&](int i, int j) -> uint8_t { return uint8_t((i - i0) * nCols + (j - j0)); };

        std::vector<uint8_t> localIndices;
        auto                 addTriangle = [&](uint8_t a, uint8_t b, uint8_t c) {
          const float eps = 1e-6F;
          if(glm::distance(localVerts[a], localVerts[b]) < eps || glm::distance(localVerts[b], localVerts[c]) < eps
             || glm::distance(localVerts[a], localVerts[c]) < eps)
            return;
          localIndices.insert(localIndices.end(), {a, b, c});
        };

        for(int i = i0; i < i1; i++)
        {
          for(int j = j0; j < j1; j++)
          {
            const uint8_t a = localIndex(i, j);
            const uint8_t b = localIndex(i + 1, j);
            const uint8_t c = localIndex(i + 1, j + 1);
            const uint8_t d = localIndex(i, j + 1);
            addTriangle(a, b, c);
            addTriangle(a, c, d);
          }
        }

        ClusterInfo cluster{
            .vertexByteOffset = uint32_t(m_vertices.size() * sizeof(glm::vec3)),
            .indexByteOffset  = uint32_t(m_indices.size()),
            .vertexCount      = uint32_t(localVerts.size()),
            .triangleCount    = uint32_t(localIndices.size() / 3),
        };
        m_clusters.push_back(cluster);
        m_maxClusterVertices  = std::max(m_maxClusterVertices, cluster.vertexCount);
        m_maxClusterTriangles = std::max(m_maxClusterTriangles, cluster.triangleCount);

        m_vertices.insert(m_vertices.end(), localVerts.begin(), localVerts.end());
        m_indices.insert(m_indices.end(), localIndices.begin(), localIndices.end());
      }
    }

    return uint32_t(m_clusters.size()) - clustersBefore;
  }

  //--------------------------------------------------------------------------------------------------
  // Generate every LOD level and upload the concatenated geometry.
  //
  void createScene()
  {
    SCOPED_TIMER(__FUNCTION__);

    for(int level = 0; level < kNumLevels; level++)
    {
      Lod lod;
      lod.clasOffset   = uint32_t(m_clusters.size());
      lod.clusterCount = appendSphereLevel(kLevelRes[level].x, kLevelRes[level].y);
      m_lods.push_back(lod);
      LOGI("LOD %d: %u clusters\n", level, lod.clusterCount);
    }

    // Per-cluster hardware limits (8-bit indices => <= 256 vertices)
    assert(m_maxClusterVertices <= m_clusterProp.maxVerticesPerCluster);
    assert(m_maxClusterTriangles <= m_clusterProp.maxTrianglesPerCluster);

    const VkBufferUsageFlags2 asInputUsage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    NVVK_CHECK(m_alloc.createBuffer(m_vertexBuffer, std::span(m_vertices).size_bytes(), asInputUsage));
    NVVK_CHECK(m_alloc.createBuffer(m_indexBuffer, std::span(m_indices).size_bytes(), asInputUsage));
    NVVK_DBG_NAME(m_vertexBuffer.buffer);
    NVVK_DBG_NAME(m_indexBuffer.buffer);
    NVVK_CHECK(uploader.appendBuffer(m_vertexBuffer, 0, std::span(m_vertices)));
    NVVK_CHECK(uploader.appendBuffer(m_indexBuffer, 0, std::span(m_indices)));
    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();
  }

  //--------------------------------------------------------------------------------------------------
  // Build one CLAS per cluster (all levels at once), then one cluster bottom-level AS per level.
  //
  // There are two acceleration structures below the TLAS: a CLAS is a BVH over the *triangles* of a
  // cluster (BUILD_TRIANGLE_CLUSTER); the cluster BLAS is a real BLAS whose *leaves are references to
  // CLAS* (BUILD_CLUSTERS_BOTTOM_LEVEL), i.e. its build input is a list of CLAS addresses, not
  // vertices. Traversal is TLAS -> cluster BLAS -> CLAS -> triangles. The CLAS are built once; a BLAS
  // just gathers pointers, so it is cheap to (re)assemble from any subset of the CLAS pool - which is
  // how per-cluster LOD works. See README.md ("How CLAS and the BLAS fit together").
  //
  // Both builds use IMPLICIT_DESTINATIONS: the driver sub-allocates the output from one blob and
  // writes the resulting device addresses into an array we provide. The per-CLAS addresses feed
  // straight into the per-level BLAS build, entirely on the GPU.
  //
  void buildClusterAccelerationStructures()
  {
    SCOPED_TIMER(__FUNCTION__);

    const uint32_t numClusters   = uint32_t(m_clusters.size());
    const uint32_t numLevels     = uint32_t(m_lods.size());
    uint32_t       maxLevelCount = 0;
    for(const Lod& lod : m_lods)
      maxLevelCount = std::max(maxLevelCount, lod.clusterCount);

    // ---------- CLAS (triangle-cluster) sizing ----------
    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput{
        .sType                         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV,
        .vertexFormat                  = VK_FORMAT_R32G32B32_SFLOAT,
        .maxGeometryIndexValue         = 0,
        .maxClusterUniqueGeometryCount = 1,
        .maxClusterTriangleCount       = m_maxClusterTriangles,
        .maxClusterVertexCount         = m_maxClusterVertices,
        .maxTotalTriangleCount         = uint32_t(m_indices.size() / 3),
        .maxTotalVertexCount           = uint32_t(m_vertices.size()),
        .minPositionTruncateBitCount   = 0,
    };
    VkClusterAccelerationStructureInputInfoNV clasInput{
        .sType                         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = numClusters,
        .flags                         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV,
        .opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
        .opInput                       = {.pTriangleClusters = &triangleInput},
    };
    VkAccelerationStructureBuildSizesInfoKHR clasSizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(m_device, &clasInput, &clasSizes);

    // ---------- Cluster BLAS sizing (one BLAS per level) ----------
    VkClusterAccelerationStructureClustersBottomLevelInputNV blasInput{
        .sType                = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV,
        .maxTotalClusterCount = numClusters,
        .maxClusterCountPerAccelerationStructure = maxLevelCount,
    };
    VkClusterAccelerationStructureInputInfoNV blasInputInfo{
        .sType                         = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV,
        .maxAccelerationStructureCount = numLevels,
        .flags                         = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .opType                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV,
        .opMode                        = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV,
        .opInput                       = {.pClustersBottomLevel = &blasInput},
    };
    VkAccelerationStructureBuildSizesInfoKHR blasSizes{.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetClusterAccelerationStructureBuildSizesNV(m_device, &blasInputInfo, &blasSizes);

    // ---------- Allocate output storage, scratch and argument buffers ----------
    const VkBufferUsageFlags2 storageUsage =
        VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;

    nvvk::Buffer scratch;
    NVVK_CHECK(m_alloc.createBuffer(scratch, std::max(clasSizes.buildScratchSize, blasSizes.buildScratchSize),
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                        | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
                                    VMA_MEMORY_USAGE_AUTO, {}, m_clusterProp.clusterScratchByteAlignment));

    NVVK_CHECK(m_alloc.createBuffer(m_clasBuffer, clasSizes.accelerationStructureSize, storageUsage,
                                    VMA_MEMORY_USAGE_AUTO, {}, m_clusterProp.clusterByteAlignment));
    NVVK_CHECK(m_alloc.createBuffer(m_clusterBlasBuffer, blasSizes.accelerationStructureSize, storageUsage,
                                    VMA_MEMORY_USAGE_AUTO, {}, m_clusterProp.clusterBottomLevelByteAlignment));
    NVVK_DBG_NAME(m_clasBuffer.buffer);
    NVVK_DBG_NAME(m_clusterBlasBuffer.buffer);

    NVVK_CHECK(m_alloc.createBuffer(m_clasAddressBuffer, numClusters * sizeof(uint64_t),
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_DBG_NAME(m_clasAddressBuffer.buffer);

    const VkBufferUsageFlags2 argUsage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                         | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    const VmaAllocationCreateFlags hostFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    nvvk::Buffer clasBuildInfos;  // per-cluster VkClusterAccelerationStructureBuildTriangleClusterInfoNV
    nvvk::Buffer clasSizesOut;    // per-CLAS sizes (output, unused)
    nvvk::Buffer blasBuildInfos;  // per-level VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV
    nvvk::Buffer blasAddressOut;  // per-level BLAS device addresses (output)
    nvvk::Buffer blasSizesOut;    // per-level BLAS sizes (output, unused)
    NVVK_CHECK(m_alloc.createBuffer(clasBuildInfos, numClusters * sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
                                    argUsage, VMA_MEMORY_USAGE_AUTO, hostFlags));
    NVVK_CHECK(m_alloc.createBuffer(clasSizesOut, numClusters * sizeof(uint32_t),
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                    VMA_MEMORY_USAGE_AUTO, hostFlags));
    NVVK_CHECK(m_alloc.createBuffer(blasBuildInfos, numLevels * sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
                                    argUsage, VMA_MEMORY_USAGE_AUTO, hostFlags));
    NVVK_CHECK(m_alloc.createBuffer(blasAddressOut, numLevels * sizeof(uint64_t),
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                    VMA_MEMORY_USAGE_AUTO, hostFlags));
    NVVK_CHECK(m_alloc.createBuffer(blasSizesOut, numLevels * sizeof(uint32_t),
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                    VMA_MEMORY_USAGE_AUTO, hostFlags));

    // ---------- Fill the per-cluster CLAS build infos ----------
    auto* buildInfos = reinterpret_cast<VkClusterAccelerationStructureBuildTriangleClusterInfoNV*>(clasBuildInfos.mapping);
    for(uint32_t i = 0; i < numClusters; i++)
    {
      const ClusterInfo& c = m_clusters[i];

      VkClusterAccelerationStructureBuildTriangleClusterInfoNV info{};
      info.clusterID                                       = i;  // <-- surfaced to the shader on a hit
      info.triangleCount                                   = c.triangleCount;
      info.vertexCount                                     = c.vertexCount;
      info.indexType                                       = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
      info.indexBufferStride                               = 1;
      info.vertexBufferStride                              = uint16_t(sizeof(glm::vec3));
      info.positionTruncateBitCount                        = 0;
      info.baseGeometryIndexAndGeometryFlags.geometryIndex = 0;
      info.baseGeometryIndexAndGeometryFlags.geometryFlags = VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV;
      info.indexBuffer                                     = m_indexBuffer.address + c.indexByteOffset;
      info.vertexBuffer                                    = m_vertexBuffer.address + c.vertexByteOffset;
      buildInfos[i]                                        = info;
    }

    // ---------- Fill the per-level cluster-BLAS build infos ----------
    auto* blasInfos = reinterpret_cast<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV*>(blasBuildInfos.mapping);
    for(uint32_t l = 0; l < numLevels; l++)
    {
      blasInfos[l].clusterReferencesCount  = m_lods[l].clusterCount;
      blasInfos[l].clusterReferencesStride = sizeof(uint64_t);
      blasInfos[l].clusterReferences       = m_clasAddressBuffer.address + m_lods[l].clasOffset * sizeof(uint64_t);
    }

    // Make the host-written build infos visible to the GPU (no-op on coherent memory)
    m_alloc.autoFlushBuffer(clasBuildInfos);
    m_alloc.autoFlushBuffer(blasBuildInfos);

    // ---------- Record both builds ----------
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    VkClusterAccelerationStructureCommandsInfoNV clasCmd{
        .sType             = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
        .input             = clasInput,
        .dstImplicitData   = m_clasBuffer.address,
        .scratchData       = scratch.address,
        .dstAddressesArray = {.deviceAddress = m_clasAddressBuffer.address,
                              .stride        = sizeof(uint64_t),
                              .size          = m_clasAddressBuffer.bufferSize},
        .dstSizesArray = {.deviceAddress = clasSizesOut.address, .stride = sizeof(uint32_t), .size = clasSizesOut.bufferSize},
        .srcInfosArray = {.deviceAddress = clasBuildInfos.address,
                          .stride        = sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
                          .size          = clasBuildInfos.bufferSize},
    };
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &clasCmd);

    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    VkClusterAccelerationStructureCommandsInfoNV blasCmd{
        .sType             = VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV,
        .input             = blasInputInfo,
        .dstImplicitData   = m_clusterBlasBuffer.address,
        .scratchData       = scratch.address,
        .dstAddressesArray = {.deviceAddress = blasAddressOut.address,
                              .stride        = sizeof(uint64_t),
                              .size          = blasAddressOut.bufferSize},
        .dstSizesArray = {.deviceAddress = blasSizesOut.address, .stride = sizeof(uint32_t), .size = blasSizesOut.bufferSize},
        .srcInfosArray = {.deviceAddress = blasBuildInfos.address,
                          .stride        = sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
                          .size          = blasBuildInfos.bufferSize},
    };
    vkCmdBuildClusterAccelerationStructureIndirectNV(cmd, &blasCmd);

    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Make the GPU-written BLAS addresses visible to the host (no-op on coherent memory)
    m_alloc.autoInvalidateBuffer(blasAddressOut);

    // Store each level's cluster-BLAS device address (referenced by the TLAS)
    const uint64_t* blasAddresses = reinterpret_cast<uint64_t*>(blasAddressOut.mapping);
    for(uint32_t l = 0; l < numLevels; l++)
      m_lods[l].blasAddress = blasAddresses[l];

    m_alloc.destroyBuffer(scratch);
    m_alloc.destroyBuffer(clasBuildInfos);
    m_alloc.destroyBuffer(clasSizesOut);
    m_alloc.destroyBuffer(blasBuildInfos);
    m_alloc.destroyBuffer(blasAddressOut);
    m_alloc.destroyBuffer(blasSizesOut);
  }

  //--------------------------------------------------------------------------------------------------
  // A single TLAS instance referencing the given cluster BLAS (identity transform).
  //
  VkAccelerationStructureInstanceKHR makeInstance(VkDeviceAddress blasAddress) const
  {
    return VkAccelerationStructureInstanceKHR{
        .transform                              = nvvk::toTransformMatrixKHR(glm::mat4(1)),
        .instanceCustomIndex                    = 0,
        .mask                                   = 0xFF,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
        .accelerationStructureReference         = blasAddress,
    };
  }

  //--------------------------------------------------------------------------------------------------
  // Build the top-level AS once, with ALLOW_UPDATE so that later LOD switches only refit it. The
  // single instance lives in a persistent buffer that we rewrite in place on each switch.
  //
  void buildTlas(VkDeviceAddress blasAddress)
  {
    NVVK_CHECK(m_alloc.createBuffer(m_instancesBuffer, sizeof(VkAccelerationStructureInstanceKHR),
                                    VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                                        | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
    NVVK_DBG_NAME(m_instancesBuffer.buffer);

    m_tlasData.addGeometry(m_tlasData.makeInstanceGeometry(1, m_instancesBuffer.address));
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
        m_tlasData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                  | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);

    // Scratch must fit both the initial build and later updates
    NVVK_CHECK(m_alloc.createBuffer(m_tlasScratch, std::max(sizeInfo.buildScratchSize, sizeInfo.updateScratchSize),
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                    VMA_MEMORY_USAGE_AUTO, {}, m_asProp.minAccelerationStructureScratchOffsetAlignment));
    NVVK_CHECK(m_alloc.createAcceleration(m_tlas, m_tlasData.makeCreateInfo()));
    NVVK_DBG_NAME(m_tlas.accel);

    VkCommandBuffer                    cmd      = m_app->createTempCmdBuffer();
    VkAccelerationStructureInstanceKHR instance = makeInstance(blasAddress);
    vkCmdUpdateBuffer(cmd, m_instancesBuffer.buffer, 0, sizeof(instance), &instance);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                                       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);
    m_tlasData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, m_tlasScratch.address);
    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  //--------------------------------------------------------------------------------------------------
  // Refit (update) the existing TLAS to reference a different level's cluster BLAS. Only the instance
  // data changes (still one instance), so an in-place update is enough - no destroy/rebuild.
  //
  void refitTlas(VkDeviceAddress blasAddress)
  {
    VkCommandBuffer                    cmd      = m_app->createTempCmdBuffer();
    VkAccelerationStructureInstanceKHR instance = makeInstance(blasAddress);
    vkCmdUpdateBuffer(cmd, m_instancesBuffer.buffer, 0, sizeof(instance), &instance);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT,
                                       VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR | VK_ACCESS_2_SHADER_READ_BIT);
    m_tlasData.cmdUpdateAccelerationStructure(cmd, m_tlas.accel, m_tlasScratch.address);
    m_app->submitAndWaitTempCmdBuffer(cmd);
  }

  //--------------------------------------------------------------------------------------------------
  // Point the TLAS at a given LOD level: build it the first time (ALLOW_UPDATE), refit afterwards.
  // Level changes are rare, so a wait + refit is fine.
  //
  void setLevel(int level)
  {
    vkDeviceWaitIdle(m_device);
    if(m_tlas.accel == VK_NULL_HANDLE)
      buildTlas(m_lods[level].blasAddress);
    else
      refitTlas(m_lods[level].blasAddress);
    m_currentLevel = level;
    m_numClusters  = m_lods[level].clusterCount;
  }

  //--------------------------------------------------------------------------------------------------
  // Ray tracing pipeline: rgen / rmiss / rchit + SBT. The cluster-ID built-in is only valid when the
  // pipeline is created with the cluster pNext.
  //
  void createRtxPipeline()
  {
    SCOPED_TIMER(__FUNCTION__);

#if USE_SLANG
    if(!m_slangCompiler.compileFile("raytrace_clusters.slang"))
    {
      LOGE("Error compiling raytrace_clusters.slang\n");
      return;
    }
#endif

    vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
    m_rtPipeline = VK_NULL_HANDLE;

    if(m_descriptorSetLayout == VK_NULL_HANDLE)
    {
      m_descriptorBindings.addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
      m_descriptorBindings.addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
      m_descriptorBindings.addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
      NVVK_CHECK(m_descriptorBindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                                &m_descriptorSetLayout));
      NVVK_DBG_NAME(m_descriptorSetLayout);

      const VkPushConstantRange pushRange{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant)};
      NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorSetLayout}, {pushRange}));
      NVVK_DBG_NAME(m_pipelineLayout);
    }

    enum StageIndices
    {
      eRaygen,
      eMiss,
      eClosestHit,
      eStageCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eStageCount> stages{};

#if USE_SLANG
    const VkShaderModuleCreateInfo slangModule{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                               .codeSize = m_slangCompiler.getSpirvSize(),
                                               .pCode    = m_slangCompiler.getSpirv()};
    stages[eRaygen]     = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                           .pNext = &slangModule,
                           .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                           .pName = "rgenMain"};
    stages[eMiss]       = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                           .pNext = &slangModule,
                           .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
                           .pName = "rmissMain"};
    stages[eClosestHit] = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                           .pNext = &slangModule,
                           .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                           .pName = "rchitMain"};
#else
    const VkShaderModuleCreateInfo rgenModule  = nvsamples::getShaderModuleCreateInfo(raytrace_clusters_rgen_glsl);
    const VkShaderModuleCreateInfo rmissModule = nvsamples::getShaderModuleCreateInfo(raytrace_clusters_rmiss_glsl);
    const VkShaderModuleCreateInfo rchitModule = nvsamples::getShaderModuleCreateInfo(raytrace_clusters_rchit_glsl);
    stages[eRaygen]                            = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                  .pNext = &rgenModule,
                                                  .stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                                                  .pName = "main"};
    stages[eMiss]                              = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                  .pNext = &rmissModule,
                                                  .stage = VK_SHADER_STAGE_MISS_BIT_KHR,
                                                  .pName = "main"};
    stages[eClosestHit]                        = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                  .pNext = &rchitModule,
                                                  .stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                                                  .pName = "main"};
#endif

    VkRayTracingShaderGroupCreateInfoKHR group{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                               .generalShader      = VK_SHADER_UNUSED_KHR,
                                               .closestHitShader   = VK_SHADER_UNUSED_KHR,
                                               .anyHitShader       = VK_SHADER_UNUSED_KHR,
                                               .intersectionShader = VK_SHADER_UNUSED_KHR};
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shaderGroups.push_back(group);
    group.generalShader = eMiss;
    shaderGroups.push_back(group);
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    shaderGroups.push_back(group);

    // Enable cluster acceleration structures for this pipeline (required for the cluster-ID built-in)
    VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV clusterPipeInfo{
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV,
        .allowClusterAccelerationStructure = VK_TRUE,
    };

    VkRayTracingPipelineCreateInfoKHR pipelineInfo{
        .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .pNext                        = &clusterPipeInfo,
        .stageCount                   = uint32_t(stages.size()),
        .pStages                      = stages.data(),
        .groupCount                   = uint32_t(shaderGroups.size()),
        .pGroups                      = shaderGroups.data(),
        .maxPipelineRayRecursionDepth = 1,
        .layout                       = m_pipelineLayout,
    };
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &pipelineInfo, nullptr, &m_rtPipeline));
    NVVK_DBG_NAME(m_rtPipeline);

    m_alloc.destroyBuffer(m_sbtBuffer);
    nvvk::SBTGenerator sbtGenerator;
    sbtGenerator.init(m_device, m_rtProp);
    const size_t bufferSize = sbtGenerator.calculateSBTBufferSize(m_rtPipeline, pipelineInfo);
    NVVK_CHECK(m_alloc.createBuffer(m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                    sbtGenerator.getBufferAlignment()));
    NVVK_DBG_NAME(m_sbtBuffer.buffer);
    NVVK_CHECK(sbtGenerator.populateSBTBuffer(m_sbtBuffer.address, bufferSize, m_sbtBuffer.mapping));
    m_sbtRegions = sbtGenerator.getSBTRegions();
    sbtGenerator.deinit();
  }

  void createFrameInfoBuffer()
  {
    NVVK_CHECK(m_alloc.createBuffer(m_frameInfo, sizeof(shaderio::FrameInfo),
                                    VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
    NVVK_DBG_NAME(m_frameInfo.buffer);
  }

  void pushDescriptorSet(VkCommandBuffer cmd)
  {
    nvvk::WriteSetContainer writes{};
    writes.append(m_descriptorBindings.getWriteSet(B_tlas), m_tlas);
    writes.append(m_descriptorBindings.getWriteSet(B_outImage), m_renderTarget.getColorAttachmentView(0), VK_IMAGE_LAYOUT_GENERAL);
    writes.append(m_descriptorBindings.getWriteSet(B_frameInfo), m_frameInfo);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0, uint32_t(writes.size()),
                              writes.data());
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_renderTarget.getColorImage(0), m_renderTarget.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  nvapp::Application*     m_app{};
  VkDevice                m_device{};
  nvvk::ResourceAllocator m_alloc{};
  nvvk::RenderTarget      m_renderTarget{};
  nvapp::ImTexture        m_viewportImage{};
  nvslang::SlangCompiler  m_slangCompiler{};

  // Geometry (all LOD levels concatenated)
  std::vector<glm::vec3>   m_vertices;
  std::vector<uint8_t>     m_indices;
  std::vector<ClusterInfo> m_clusters;
  std::vector<Lod>         m_lods;
  uint32_t                 m_maxClusterVertices{};
  uint32_t                 m_maxClusterTriangles{};
  nvvk::Buffer             m_vertexBuffer{};
  nvvk::Buffer             m_indexBuffer{};

  // Cluster acceleration structures
  nvvk::Buffer m_clasBuffer{};         // storage for all CLAS (all levels)
  nvvk::Buffer m_clasAddressBuffer{};  // per-CLAS device addresses
  nvvk::Buffer m_clusterBlasBuffer{};  // storage for the per-level cluster BLAS

  nvvk::AccelerationStructure m_tlas{};
  nvvk::AccelerationStructureBuildData m_tlasData{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};  // kept alive for refits
  nvvk::Buffer m_instancesBuffer{};  // the single TLAS instance (rewritten on LOD change)
  nvvk::Buffer m_tlasScratch{};      // persistent TLAS build/update scratch
  int          m_currentLevel{-1};   // level currently referenced by the TLAS
  int          m_lodOverride{-1};    // -1 = automatic (distance-based)
  uint32_t     m_numClusters{};      // cluster count of the active level (UI)

  // Pipeline
  nvvk::DescriptorBindings    m_descriptorBindings{};
  VkDescriptorSetLayout       m_descriptorSetLayout{};
  VkPipelineLayout            m_pipelineLayout{};
  VkPipeline                  m_rtPipeline{};
  nvvk::Buffer                m_sbtBuffer{};
  nvvk::SBTGenerator::Regions m_sbtRegions{};

  nvvk::Buffer           m_frameInfo{};
  shaderio::PushConstant m_pushConst{};
  glm::vec3              m_lightDir{0.5F, 1.0F, 0.7F};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_asProp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceClusterAccelerationStructurePropertiesNV m_clusterProp{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_PROPERTIES_NV};
};

//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  bool                       verbose = false;
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.addVector({"size", "Size of the window to be created", "s"}, &appInfo.windowSize);
  reg.add({"headless"}, &appInfo.headless, true);
  reg.add({"frames", "Number of frames to run in headless mode"}, &appInfo.headlessFrameCount);
  reg.add({"verbose", "Verbose output of the Vulkan context"}, &verbose);
  cli.add(reg);
  cli.parse(argc, argv);

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceClusterAccelerationStructureFeaturesNV clusterFeature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CLUSTER_ACCELERATION_STRUCTURE_FEATURES_NV};

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions =
          {
              {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
              {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
              {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},
              {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},
              {VK_NV_CLUSTER_ACCELERATION_STRUCTURE_EXTENSION_NAME, &clusterFeature, false, 2},  // request spec version 2
          },
      .queues = {VK_QUEUE_GRAPHICS_BIT},
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  vkSetup.verbose = verbose;

  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  appInfo.name           = fmt::format("{} ({})", nvutils::getExecutablePath().stem().string(), SHADER_LANGUAGE_STR);
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  nvapp::Application app;
  app.init(appInfo);

  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(g_cameraManip);

  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", appInfo.name));
  app.addElement(std::make_shared<RayTraceClusters>());

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
