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

//////////////////////////////////////////////////////////////////////////
/*

  This sample raytrace a scene made of multiple primitives. 
  - The scene is created in createScene()
  - Then Vulkan buffers holding the scene are created in createVkBuffers()
  - Bottom and Top level acceleration structures are using the Vulkan buffers 
    and scene description in createBottomLevelAS() and createTopLevelAS()
  - The raytracing pipeline, composed of RayGen, Miss, ClosestHit shaders
    and the creation of the shading binding table, is done increateRtxPipeline()
  - Rendering is done in onRender()


*/
//////////////////////////////////////////////////////////////////////////


#define IMGUI_DEFINE_MATH_OPERATORS  // ImGUI ImVec maths

#define VMA_IMPLEMENTATION
#include "imgui/imgui_axis.hpp"
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/shaders/dh_sky.h"

#include "shaders/dh_bindings.h"

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

//#undef USE_HLSL

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


#define MAXRAYRECURSIONDEPTH 5

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

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);                    // Debug utility
    m_alloc = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());  // Allocator
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProperties};
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    const uint32_t gct_queue_index = m_app->getContext()->m_queueGCT.familyIndex;
    m_rtBuilder.setup(m_device, m_alloc.get(), gct_queue_index);
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
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), VkExtent2D{width, height}, m_colorFormat, m_depthFormat);
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
      glm::vec3 dir = m_skyParams.directionToLight;
      ImGuiH::azimuthElevationSliders(dir, false);
      m_skyParams.directionToLight = dir;
      PropertyEditor::end();
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
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
      ImGui::PopStyleVar();
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
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(nvvkhl_shaders::ProceduralSkyShaderParameters), &m_skyParams);  // Update the sky
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
  void createScene()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    // Meshes
    m_meshes.emplace_back(nvh::createSphereUv());
    m_meshes.emplace_back(nvh::createCube());
    m_meshes.emplace_back(nvh::createTetrahedron());
    m_meshes.emplace_back(nvh::createOctahedron());
    m_meshes.emplace_back(nvh::createIcosahedron());
    m_meshes.emplace_back(nvh::createConeMesh());
    const int num_meshes = static_cast<int>(m_meshes.size());

    // Materials (colorful)
    for(int i = 0; i < num_meshes; i++)
    {
      const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
      const glm::vec3 v    = static_cast<glm::vec3>(sin(freq) * 0.5F + 0.5F);
      m_materials.push_back({glm::vec4(v, 1)});
    }

    // Instances
    for(int i = 0; i < num_meshes; i++)
    {
      nvh::Node& n  = m_nodes.emplace_back();
      n.mesh        = i;
      n.material    = i;
      n.translation = glm::vec3(-(static_cast<float>(num_meshes) / 2.F) + static_cast<float>(i), 0.F, 0.F);
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
    m_pushConst.roughness = 1.0F;
    m_pushConst.metallic  = 0.5F;

    // Default Sky values
    m_skyParams = nvvkhl_shaders::initSkyShaderParameters();
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
    m_bSkyParams = m_alloc->createBuffer(sizeof(nvvkhl_shaders::ProceduralSkyShaderParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
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
  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const nvh::PrimitiveMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    const auto max_primitive_count = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32A32_SFLOAT,  // vec3 vertex position data
        .vertexData   = {.deviceAddress = vertexAddress},
        .vertexStride = sizeof(nvh::PrimitiveVertex),
        .maxVertex    = static_cast<uint32_t>(prim.vertices.size()) - 1,
        .indexType    = VK_INDEX_TYPE_UINT32,
        .indexData    = {.deviceAddress = indexAddress},
    };

    // Identify the above data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR as_geom{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry     = {.triangles = triangles},
        .flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR,
    };

    VkAccelerationStructureBuildRangeInfoKHR offset{
        .primitiveCount  = max_primitive_count,
        .primitiveOffset = 0,
        .firstVertex     = 0,
        .transformOffset = 0,
    };

    // Our BLAS is made from only one geometry, but could be made of many geometries
    nvvk::RaytracingBuilderKHR::BlasInput input;
    input.asGeometry.emplace_back(as_geom);
    input.asBuildOffsetInfo.emplace_back(offset);

    return input;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::RaytracingBuilderKHR::BlasInput> all_blas;
    all_blas.reserve(m_meshes.size());

    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      const VkDeviceAddress vertex_address = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].vertices.buffer);
      const VkDeviceAddress index_address  = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].indices.buffer);

      const nvvk::RaytracingBuilderKHR::BlasInput geo = primitiveToGeometry(m_meshes[p_idx], vertex_address, index_address);
      all_blas.push_back({geo});
    }
    m_rtBuilder.buildBlas(all_blas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    std::vector<VkAccelerationStructureInstanceKHR> tlas;
    tlas.reserve(m_nodes.size());
    for(const nvh::Node& node : m_nodes)
    {
      VkAccelerationStructureInstanceKHR ray_inst{
          .transform           = nvvk::toTransformMatrixKHR(node.localMatrix()),  // Position of the instance
          .instanceCustomIndex = static_cast<uint32_t>(node.mesh),                // gl_InstanceCustomIndexEX
          .mask                = 0xFF,                                            // All objects
          .instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all object
          .flags                                  = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR,
          .accelerationStructureReference         = m_rtBuilder.getBlasDeviceAddress(node.mesh),
      };
      tlas.emplace_back(ray_inst);
    }
    m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
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
    vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout);
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
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, &p.plines[0]);
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
    VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
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
    m_rtBuilder.destroy();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                          m_app = nullptr;
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>             m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtSet;  // Descriptor set

  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  nvvkhl_shaders::ProceduralSkyShaderParameters m_skyParams{};

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
  nvvk::SBTWrapper           m_sbt;        // Shading binding table wrapper
  nvvk::RaytracingBuilderKHR m_rtBuilder;  // Helper for building Top and bottom level acceleration structures
  nvvkhl::PipelineContainer  m_rtPipe;     // Hold pipelines and layout
};


//////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  spec.vSync            = true;
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  spec.vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accel_feature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rt_pipeline_feature);  // To use vkCmdTraceRaysKHR
  spec.vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
#if USE_HLSL  // DXC is automatically adding the extension
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayqueryFeature);
#endif  // USE_HLSL

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());       // Camera manipulation
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<Raytracing>());  // Sample

  app->run();
  app.reset();

  return test->errorCode();
}
