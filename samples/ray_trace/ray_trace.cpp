/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*

  This sample raytraces a scene made of multiple primitives. 
  - The scene is created in createScene()
  - Then Vulkan buffers holding the scene are created in createVkBuffers()
  - Bottom and Top level acceleration structures are using the Vulkan buffers 
    and scene description in createBottomLevelAS() and createTopLevelAS()
  - The raytracing pipeline, composed of RayGen, Miss, ClosestHit shaders
    and the creation of the shading binding table, is done in createRtxPipeline()
  - Rendering is done in onRender()


*/
//////////////////////////////////////////////////////////////////////////


#define IMGUI_DEFINE_MATH_OPERATORS  // ImGUI ImVec maths

#include "common/vk_context.hpp"                    // Vulkan context creation
#include "imgui/imgui_axis.hpp"                     // Display of axis
#include "imgui/imgui_camera_widget.h"              // Camera UI
#include "imgui/imgui_helper.h"                     // Property editor
#include "nvh/primitives.hpp"                       // Various primitives
#include "nvvk/acceleration_structures.hpp"         // BLAS & TLAS creation helper
#include "nvvk/descriptorsets_vk.hpp"               // Descriptor set creation helper
#include "nvvk/extensions_vk.hpp"                   // Vulkan extension declaration
#include "nvvk/sbtwrapper_vk.hpp"                   // Shading binding table creation helper
#include "nvvk/shaders_vk.hpp"                      // Shader module creation wrapper
#include "nvvkhl/element_benchmark_parameters.hpp"  // For benchmark and tests
#include "nvvkhl/element_camera.hpp"                // To manipulate the camera
#include "nvvkhl/element_gui.hpp"                   // Application Menu / titlebar
#include "nvvkhl/gbuffer.hpp"                       // G-Buffer creation helper
#include "nvvkhl/pipeline_container.hpp"            // Container to hold pipelines
#include "nvvkhl/sky.hpp"                           // Sun & Sky
#include "nvvk/renderpasses_vk.hpp"


namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
#include "shaders/dh_bindings.h"  // Local device/host shared structures
}  // namespace DH

// Local shaders
#if USE_HLSL
#include "_autogen/raytrace_rgenMain.spirv.h"
#include "_autogen/raytrace_rchitMain.spirv.h"
#include "_autogen/raytrace_rmissMain.spirv.h"
const auto& rgen_shd  = std::vector<char>{std::begin(raytrace_rgenMain), std::end(raytrace_rgenMain)};
const auto& rchit_shd = std::vector<char>{std::begin(raytrace_rchitMain), std::end(raytrace_rchitMain)};
const auto& rmiss_shd = std::vector<char>{std::begin(raytrace_rmissMain), std::end(raytrace_rmissMain)};
#elif USE_SLANG
#include "_autogen/raytrace_slang.h"
#else
#include "_autogen/raytrace.rchit.glsl.h"
#include "_autogen/raytrace.rgen.glsl.h"
#include "_autogen/raytrace.rmiss.glsl.h"
const auto& rgen_shd  = std::vector<uint32_t>{std::begin(raytrace_rgen_glsl), std::end(raytrace_rgen_glsl)};
const auto& rchit_shd = std::vector<uint32_t>{std::begin(raytrace_rchit_glsl), std::end(raytrace_rchit_glsl)};
const auto& rmiss_shd = std::vector<uint32_t>{std::begin(raytrace_rmiss_glsl), std::end(raytrace_rmiss_glsl)};
#endif

// The maximum depth recursion for the ray tracer
uint32_t MAXRAYRECURSIONDEPTH = 10;

//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class Raytracing : public nvvkhl::IAppElement
{
public:
  Raytracing()           = default;
  ~Raytracing() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    nvh::ScopedTimer st(__FUNCTION__);

    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
    m_alloc = std::make_unique<nvvk::ResourceAllocatorDma>(m_device, m_app->getPhysicalDevice());
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProperties};
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);
    MAXRAYRECURSIONDEPTH = std::min(m_rtProperties.maxRayRecursionDepth, MAXRAYRECURSIONDEPTH);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    const uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt.setup(m_device, gct_queue_index, m_alloc.get(), m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    nvh::ScopedTimer st(__FUNCTION__);
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), VkExtent2D{width, height}, m_colorFormat);
    writeRtDesc();
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();

      using namespace ImGuiH;
      PropertyEditor::begin();
      PropertyEditor::entry("Metallic", [&] { return ImGui::SliderFloat("#1", &m_pushConst.metallic, 0.0F, 1.0F); });
      PropertyEditor::entry("Roughness", [&] { return ImGui::SliderFloat("#1", &m_pushConst.roughness, 0.0F, 1.0F); });
      PropertyEditor::entry("Intensity", [&] { return ImGui::SliderFloat("#1", &m_pushConst.intensity, 0.0F, 10.0F); });
      PropertyEditor::entry("Depth",
                            [&] { return ImGui::SliderInt("#1", &m_pushConst.maxDepth, 0, MAXRAYRECURSIONDEPTH); });
      PropertyEditor::end();
      ImGui::Separator();
      ImGui::Text("Sun Orientation");
      PropertyEditor::begin();
      nvvkhl::skyParametersUI(m_skyParams);
      PropertyEditor::end();
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());

      {  // Display orientation axis at the bottom left corner of the window
        float  axisSize = 25.F;
        ImVec2 pos      = ImGui::GetWindowPos();
        pos.y += ImGui::GetWindowSize().y;
        pos += ImVec2(axisSize * 1.1F, -axisSize * 1.1F) * ImGui::GetWindowDpiScale();  // Offset
        ImGuiH::Axis(pos, CameraManip.getMatrix(), axisSize);
      }

      ImGui::End();
    }
  }

  void memoryBarrier(VkCommandBuffer cmd)
  {
    VkMemoryBarrier mb{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    VkPipelineStageFlags srcDstStage{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    vkCmdPipelineBarrier(cmd, srcDstStage, srcDstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
  }

  void onRender(VkCommandBuffer cmd) override
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    // Camera matrices
    glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), CameraManip.getAspectRatio(),
                                           CameraManip.getClipPlanes().x, CameraManip.getClipPlanes().y);
    proj[1][1] *= -1;  // Vulkan has it inverted

    // Update uniform buffers
    DH::FrameInfo finfo{.projInv = glm::inverse(proj), .viewInv = glm::inverse(CameraManip.getMatrix())};
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);  // Update FrameInfo
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(nvvkhl_shaders::SimpleSkyParameters), &m_skyParams);  // Update the sky
    memoryBarrier(cmd);  // Make sure the data has moved to device before rendering

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtSet->getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0, (uint32_t)desc_sets.size(),
                            desc_sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &m_pushConst);

    const std::array<VkStridedDeviceAddressRegionKHR, 4>& bindingTables = m_sbt.getRegions();
    const VkExtent2D&                                     size          = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &bindingTables[0], &bindingTables[1], &bindingTables[2], &bindingTables[3], size.width, size.height, 1);
  }

private:
  glm::vec4 nextColor()
  {
    static float    index = 1.0F;
    const glm::vec3 freq  = glm::vec3(1.33333F, 2.33333F, 3.33333F) * index;
    const glm::vec3 v     = static_cast<glm::vec3>(sin(freq) * 0.5F + 0.5F);
    index += 0.707F;
    return {v, 1};
  }
  void createScene()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    // Meshes
    m_meshes.emplace_back(nvh::createSphereUv(0.5f, 200, 200));
    m_meshes.emplace_back(nvh::createCube(0.7f, 0.7f, 0.7f));
    m_meshes.emplace_back(nvh::createTetrahedron());
    m_meshes.emplace_back(nvh::createOctahedron());
    m_meshes.emplace_back(nvh::createIcosahedron());
    m_meshes.emplace_back(nvh::createConeMesh(.5f, 1.0f, 100));
    m_meshes.emplace_back(nvh::createTorusMesh(.25f, .15f, 100, 100));
    const int numMeshes = static_cast<int>(m_meshes.size());

    // Materials (colorful)
    for(int i = 0; i < numMeshes; i++)
    {
      m_materials.push_back({nextColor()});
    }

    // Instances
    for(int i = 0; i < numMeshes; i++)
    {
      nvh::Node& n  = m_nodes.emplace_back();
      n.mesh        = i;
      n.material    = i;
      n.translation = glm::vec3(-(static_cast<float>(numMeshes) / 2.F) + static_cast<float>(i), 0.F, 0.F);
    }

    // Adding a plane & material
    m_materials.push_back({glm::vec4(.7F, .7F, .7F, 1.0F)});
    m_meshes.emplace_back(nvh::createPlane(10, 100, 100));
    nvh::Node& n  = m_nodes.emplace_back();
    n.mesh        = static_cast<int>(m_meshes.size()) - 1;
    n.material    = static_cast<int>(m_materials.size()) - 1;
    n.translation = {0, -1, 0};

    // Setting camera to see the scene
    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({-0.5F, 0.0F, 5.0F}, {-0.5F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});

    // Default parameters for overall material
    m_pushConst.intensity = 5.0F;
    m_pushConst.maxDepth  = 5;
    m_pushConst.roughness = 0.2F;
    m_pushConst.metallic  = 0.3F;

    // Default Sky values
    m_skyParams = nvvkhl_shaders::initSimpleSkyParameters();
  }


  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_bMeshes.resize(m_meshes.size());

    const VkBufferUsageFlags rt_usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_bMeshes[i];
      m.vertices         = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rt_usage_flag);
      m.indices          = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rt_usage_flag);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);
    }

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    // Create the buffer of sky parameters, updated at each frame
    m_bSkyParams = m_alloc->createBuffer(sizeof(nvvkhl_shaders::SimpleSkyParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bSkyParams.buffer);

    // Primitive instance information
    std::vector<DH::InstanceInfo> inst_info;
    inst_info.reserve(m_nodes.size());
    for(const nvh::Node& node : m_nodes)
    {
      DH::InstanceInfo info{.transform = node.localMatrix(), .materialID = node.material};
      inst_info.push_back(info);
    }
    m_bInstInfoBuffer =
        m_alloc->createBuffer(cmd, inst_info, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bInstInfoBuffer.buffer);

    m_bMaterials = m_alloc->createBuffer(cmd, m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bMaterials.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const nvh::PrimitiveMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    nvvk::AccelerationStructureGeometryInfo result;
    const auto                              triangleCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,  // vec3 vertex position data
        .vertexData   = {.deviceAddress = vertexAddress},
        .vertexStride = sizeof(nvh::PrimitiveVertex),
        .maxVertex    = static_cast<uint32_t>(prim.vertices.size()) - 1,
        .indexType    = VK_INDEX_TYPE_UINT32,
        .indexData    = {.deviceAddress = indexAddress},
    };

    // Identify the above data as containing opaque triangles.
    result.geometry = VkAccelerationStructureGeometryKHR{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry     = {.triangles = triangles},
        .flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR,
    };

    result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{
        .primitiveCount  = triangleCount,
        .primitiveOffset = 0,
        .firstVertex     = 0,
        .transformOffset = 0,
    };

    return result;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    size_t numMeshes = m_meshes.size();

    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::AccelerationStructureBuildData> blasBuildData;
    blasBuildData.reserve(numMeshes);
    m_blas.resize(numMeshes);  // All BLAS

    // Get the build information for all the BLAS
    for(uint32_t p_idx = 0; p_idx < numMeshes; p_idx++)
    {
      nvvk::AccelerationStructureBuildData buildData{VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};

      const VkDeviceAddress vertexBufferAddress = m_bMeshes[p_idx].vertices.address;
      const VkDeviceAddress indexBufferAddress  = m_bMeshes[p_idx].indices.address;

      auto geo = primitiveToGeometry(m_meshes[p_idx], vertexBufferAddress, indexBufferAddress);
      buildData.addGeometry(geo);

      VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
          buildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                   | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);

      blasBuildData.emplace_back(buildData);
    }

    // Find the most optimal size for our scratch buffer, and get the addresses of the scratch buffers
    // to allow a maximum of BLAS to be built in parallel, within the budget
    nvvk::BlasBuilder blasBuilder(m_alloc.get(), m_device);
    VkDeviceSize      hintScratchBudget = 2'000'000;  // Limiting the size of the scratch buffer to 2MB
    VkDeviceSize      scratchSize       = blasBuilder.getScratchSize(hintScratchBudget, blasBuildData);
    nvvk::Buffer      scratchBuffer =
        m_alloc->createBuffer(scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    std::vector<VkDeviceAddress> scratchAddresses;
    blasBuilder.getScratchAddresses(hintScratchBudget, blasBuildData, scratchBuffer.address, scratchAddresses);

    // Start the build and compaction of the BLAS
    VkDeviceSize hintBuildBudget = 2'000'000;  // Limiting the size of the scratch buffer to 2MB
    bool         finished        = false;
    LOGI("\n");
    do
    {
      {
        // Create, build and query the size of the BLAS, up to the 2MBi
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();
        finished = blasBuilder.cmdCreateParallelBlas(cmd, blasBuildData, m_blas, scratchAddresses, hintBuildBudget);
        m_app->submitAndWaitTempCmdBuffer(cmd);
      }
      {
        // Compacting the BLAS, and destroy the previous ones
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();
        blasBuilder.cmdCompactBlas(cmd, blasBuildData, m_blas);
        m_app->submitAndWaitTempCmdBuffer(cmd);
        blasBuilder.destroyNonCompactedBlas();
      }
    } while(!finished);
    LOGI("%s%s\n", nvh::ScopedTimer::indent().c_str(), blasBuilder.getStatistics().toString().c_str());

    // Cleanup
    m_alloc->destroy(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(m_nodes.size());
    for(const nvh::Node& node : m_nodes)
    {
      VkAccelerationStructureInstanceKHR ray_inst{
          .transform           = nvvk::toTransformMatrixKHR(node.localMatrix()),  // Position of the instance
          .instanceCustomIndex = static_cast<uint32_t>(node.mesh),                // gl_InstanceCustomIndexEX
          .mask                = 0xFF,                                            // All objects
          .instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all object
          .flags                                  = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR,
          .accelerationStructureReference         = m_blas[node.mesh].address,
      };
      tlasInstances.emplace_back(ray_inst);
    }

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    // Create the instances buffer, add a barrier to ensure the data is copied before the TLAS build
    nvvk::Buffer instancesBuffer = m_alloc->createBuffer(cmd, tlasInstances,
                                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                                             | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);


    nvvk::AccelerationStructureBuildData    tlasBuildData{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
    nvvk::AccelerationStructureGeometryInfo geometryInfo =
        tlasBuildData.makeInstanceGeometry(tlasInstances.size(), instancesBuffer.address);
    tlasBuildData.addGeometry(geometryInfo);
    // Get the size of the TLAS
    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    // Create the scratch buffer
    nvvk::Buffer scratchBuffer = m_alloc->createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Create the TLAS
    m_tlas = m_alloc->createAcceleration(tlasBuildData.makeCreateInfo());
    tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, scratchBuffer.address);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_alloc->destroy(scratchBuffer);
    m_alloc->destroy(instancesBuffer);
    m_alloc->finalizeAndReleaseStaging();
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    nvvkhl::PipelineContainer& p = m_rtPipe;
    p.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set
    m_rtSet->addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_skyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_vertex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_rtSet->initLayout();
    m_rtSet->initPool(1);

    m_dutil->DBG_NAME(m_rtSet->getLayout());
    m_dutil->DBG_NAME(m_rtSet->getSet(0));

    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eClosestHit,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    for(auto& s : stages)
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
#if USE_SLANG
    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &raytraceSlang[0], sizeof(raytraceSlang));
    stages[eRaygen].module      = shaderModule;
    stages[eRaygen].pName       = "rgenMain";
    stages[eRaygen].stage       = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].module        = shaderModule;
    stages[eMiss].pName         = "rmissMain";
    stages[eMiss].stage         = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].module  = shaderModule;
    stages[eClosestHit].pName   = "rchitMain";
    stages[eClosestHit].stage   = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
#else
    stages[eRaygen].module     = nvvk::createShaderModule(m_device, rgen_shd);
    stages[eRaygen].pName      = USE_HLSL ? "rgenMain" : "main";
    stages[eRaygen].stage      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].module       = nvvk::createShaderModule(m_device, rmiss_shd);
    stages[eMiss].pName        = USE_HLSL ? "rmissMain" : "main";
    stages[eMiss].stage        = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].module = nvvk::createShaderModule(m_device, rchit_shd);
    stages[eClosestHit].pName  = USE_HLSL ? "rchitMain" : "main";
    stages[eClosestHit].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
#endif

    m_dutil->setObjectName(stages[eRaygen].module, "Raygen");
    m_dutil->setObjectName(stages[eMiss].module, "Miss");
    m_dutil->setObjectName(stages[eClosestHit].module, "Closest Hit");


    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                                               .generalShader      = VK_SHADER_UNUSED_KHR,
                                               .closestHitShader   = VK_SHADER_UNUSED_KHR,
                                               .anyHitShader       = VK_SHADER_UNUSED_KHR,
                                               .intersectionShader = VK_SHADER_UNUSED_KHR};

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shader_groups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    shader_groups.push_back(group);

    // closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    shader_groups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant)};

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {m_rtSet->getLayout()};  // , m_pContainer[eGraphic].dstLayout};
    VkPipelineLayoutCreateInfo pipeline_layout_create_info{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = static_cast<uint32_t>(rt_desc_set_layouts.size()),
        .pSetLayouts            = rt_desc_set_layouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant,
    };
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout));
    m_dutil->DBG_NAME(p.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{
        .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount                   = static_cast<uint32_t>(stages.size()),  // Stages are shader
        .pStages                      = stages.data(),
        .groupCount                   = static_cast<uint32_t>(shader_groups.size()),
        .pGroups                      = shader_groups.data(),
        .maxPipelineRayRecursionDepth = MAXRAYRECURSIONDEPTH,  // Ray dept
        .layout                       = p.layout,
    };
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, &p.plines[0]));
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt.create(p.plines[0], ray_pipeline_info);

    // Removing temp modules
#if USE_SLANG
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#else
    for(const VkPipelineShaderStageCreateInfo& s : stages)
      vkDestroyShaderModule(m_device, s.module, nullptr);
#endif
  }

  void writeRtDesc()
  {
    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_tlas.accel;
    VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                                                              .accelerationStructureCount = 1,
                                                              .pAccelerationStructures    = &tlas};
    const VkDescriptorImageInfo  image_info{{}, m_gBuffers->getColorImageView(), VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo dbi_sky{m_bSkyParams.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo mat_desc{m_bMaterials.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo inst_desc{m_bInstInfoBuffer.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkDescriptorBufferInfo> vertex_desc;
    std::vector<VkDescriptorBufferInfo> index_desc;
    vertex_desc.reserve(m_bMeshes.size());
    index_desc.reserve(m_bMeshes.size());
    for(auto& m : m_bMeshes)
    {
      vertex_desc.push_back({m.vertices.buffer, 0, VK_WHOLE_SIZE});
      index_desc.push_back({m.indices.buffer, 0, VK_WHOLE_SIZE});
    }

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &desc_as_info));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, &image_info));
    writes.emplace_back(m_rtSet->makeWrite(0, B_frameInfo, &dbi_unif));
    writes.emplace_back(m_rtSet->makeWrite(0, B_skyParam, &dbi_sky));
    writes.emplace_back(m_rtSet->makeWrite(0, B_materials, &mat_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_instances, &inst_desc));
    writes.emplace_back(m_rtSet->makeWriteArray(0, B_vertex, vertex_desc.data()));
    writes.emplace_back(m_rtSet->makeWriteArray(0, B_index, index_desc.data()));

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void destroyResources()
  {
    for(PrimitiveMeshVk& m : m_bMeshes)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bInstInfoBuffer);
    m_alloc->destroy(m_bMaterials);
    m_alloc->destroy(m_bSkyParams);

    m_rtSet->deinit();
    m_gBuffers.reset();

    m_rtPipe.destroy(m_device);
    m_sbt.destroy();

    for(auto& b : m_blas)
      m_alloc->destroy(b);
    m_alloc->destroy(m_tlas);
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers->getColorImage(), m_gBuffers->getSize(),
                           nvh::getExecutablePath().replace_extension(".jpg").string(), 95);
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                          m_app = nullptr;
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvk::ResourceAllocatorDma>   m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtSet;  // Descriptor set

  VkFormat                            m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Color format of the image
  VkDevice                            m_device      = VK_NULL_HANDLE;            // Convenient
  std::unique_ptr<nvvkhl::GBuffer>    m_gBuffers;                                // G-Buffers: color + depth
  nvvkhl_shaders::SimpleSkyParameters m_skyParams{};

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;
    nvvk::Buffer indices;
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;  // Each primitive holds a buffer of vertices and indices
  nvvk::Buffer                 m_bFrameInfo;
  nvvk::Buffer                 m_bInstInfoBuffer;
  nvvk::Buffer                 m_bMaterials;
  nvvk::Buffer                 m_bSkyParams;

  std::vector<nvvk::AccelKHR> m_blas;  // Bottom-level AS
  nvvk::AccelKHR              m_tlas;  // Top-level AS

  // Data and setting
  struct Material
  {
    glm::vec4 color{1.F};
  };
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;
  std::vector<Material>           m_materials;

  // Pipeline
  DH::PushConstant m_pushConst{};  // Information sent to the shader

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper          m_sbt;     // Shading binding table wrapper
  nvvkhl::PipelineContainer m_rtPipe;  // Hold pipelines and layout
};


//////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo appInfo;

  nvh::CommandLineParser cli(PROJECT_NAME);
  cli.addArgument({"--headless"}, &appInfo.headless, "Run in headless mode");
  cli.addArgument({"--frames"}, &appInfo.headlessFrameCount, "Number of frames to render in headless mode");
  cli.parse(argc, argv);


  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};

  // Configure Vulkan context creation
  VkContextSettings vkSetup;
  if(!appInfo.headless)
  {
    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accel_feature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rt_pipeline_feature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME});  // Require for Undockable Viewport

#if USE_HLSL || USE_SLANG  // DXC is automatically adding the extension
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});
#endif  // USE_HLSL

#if(VK_HEADER_VERSION >= 283)
  // To enable ray tracing validation, set the NV_ALLOW_RAYTRACING_VALIDATION=1 environment variable
  // https://developer.nvidia.com/blog/ray-tracing-validation-at-the-driver-level/
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_ray_tracing_validation.html
  VkPhysicalDeviceRayTracingValidationFeaturesNV validationFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES_NV};
  vkSetup.deviceExtensions.push_back({VK_NV_RAY_TRACING_VALIDATION_EXTENSION_NAME, &validationFeatures, false});
#endif

  // Create Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  // Loading the Vulkan extension pointers
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);

  // Configure application creation
  appInfo.name                  = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync                 = true;
  appInfo.instance              = vkContext->getInstance();
  appInfo.device                = vkContext->getDevice();
  appInfo.physicalDevice        = vkContext->getPhysicalDevice();
  appInfo.queues                = vkContext->getQueueInfos();
  appInfo.hasUndockableViewport = true;

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appInfo);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

#if(VK_HEADER_VERSION >= 283)
  // Check if ray tracing validation is supported
  if(validationFeatures.rayTracingValidation == VK_TRUE)
  {
    LOGI("Ray tracing validation supported");
  }
#endif

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());       // Camera manipulation
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<Raytracing>());  // Sample

  app->run();
  app.reset();
  vkContext.reset();

  return test->errorCode();
}
