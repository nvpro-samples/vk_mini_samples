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

//////////////////////////////////////////////////////////////////////////
/*

    This shows the use of Shading Execution Reorder (SER) through a simple
    path tracer. The reorder code is in the rayGen shader, and adds an 
    indication if the ray has hit something or not.

*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"

#include "shaders/dh_bindings.h"
#include "shaders/device_host.h"
#include "nvvkhl/shaders/dh_sky.h"

#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rgen.h"
#include "_autogen/pathtrace.rmiss.h"
#include "nvvk/specialization.hpp"
#include "nvvk/images_vk.hpp"


/// </summary> Ray trace multiple primitives using SER
class SerPathtrace : public nvvkhl::IAppElement
{
  enum
  {
    eImgTonemapped,
    eImgRendered
  };

public:
  SerPathtrace()           = default;
  ~SerPathtrace() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil      = std::make_unique<nvvk::DebugUtil>(m_device);                    // Debug utility
    m_alloc      = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());  // Allocator
    m_rtSet      = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(m_app->getContext().get(), m_alloc.get());

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    int32_t gctQueueIndex = m_app->getContext()->m_queueGCT.familyIndex;
    m_rtBuilder.setup(m_device, m_alloc.get(), gctQueueIndex);
    m_sbt.setup(m_device, gctQueueIndex, m_alloc.get(), m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();
    m_tonemapper->createComputePipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    createGbuffers({width, height});
    m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(eImgRendered),
                                              m_gBuffers->getDescriptorImageInfo(eImgTonemapped));
    resetFrame();
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");

      static float col[3] = {0, 0, 0};
      ImGui::ColorEdit3("Color Clear", col);

      ImGuiH::CameraWidget();

      using namespace ImGuiH;
      bool changed{false};
      if(ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();
        {
          changed |= PropertyEditor::entry("Heatmap", [&] { return ImGui::Checkbox("", (bool*)&m_pushConst.heatmap); });
          if(PropertyEditor::entry("Use SER", [&] { return ImGui::Checkbox("", (bool*)&m_useSER); }))
          {
            changed = true;
            vkDeviceWaitIdle(m_device);
            createRtxPipeline();
          }
        }
        PropertyEditor::end();

        PropertyEditor::begin();
        if(PropertyEditor::treeNode("Material"))
        {
          changed |= PropertyEditor::entry("Metallic",
                                           [&] { return ImGui::SliderFloat("#1", &m_pushConst.metallic, 0.0F, 1.0F); });
          changed |= PropertyEditor::entry("Roughness",
                                           [&] { return ImGui::SliderFloat("#1", &m_pushConst.roughness, 0.001F, 1.0F); });
          PropertyEditor::treePop();
        }
        if(PropertyEditor::treeNode("Sun"))
        {
          changed |= PropertyEditor::entry("Intensity",
                                           [&] { return ImGui::SliderFloat("#1", &m_pushConst.intensity, 0.0F, 10.0F); });
          nvmath::vec3f dir = m_skyParams.directionToLight;
          changed |= ImGuiH::azimuthElevationSliders(dir, false);
          m_skyParams.directionToLight = dir;
          PropertyEditor::treePop();
        }
        if(PropertyEditor::treeNode("Ray Tracer"))
        {
          changed |= PropertyEditor::entry("Depth", [&] { return ImGui::SliderInt("#1", &m_pushConst.maxDepth, 0, 20); });
          changed |=
              PropertyEditor::entry("Samples", [&] { return ImGui::SliderInt("#1", &m_pushConst.maxSamples, 1, 10); });
          PropertyEditor::treePop();
        }
        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        m_tonemapper->onUI();
      }


      ImGui::End();
      if(changed)
        resetFrame();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(eImgTonemapped), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    auto sdbg = m_dutil->DBG_SCOPE(cmd);

    if(!updateFrame())
    {
      return;
    }

    float         view_aspect_ratio = m_viewSize.x / m_viewSize.y;
    nvmath::vec3f eye;
    nvmath::vec3f center;
    nvmath::vec3f up;
    CameraManip.getLookat(eye, center, up);

    // Update Frame buffer uniform buffer
    FrameInfo   finfo{};
    const auto& clip = CameraManip.getClipPlanes();
    finfo.view       = CameraManip.getMatrix();
    finfo.proj       = nvmath::perspectiveVK(CameraManip.getFov(), view_aspect_ratio, clip.x, clip.y);
    finfo.projInv    = nvmath::inverse(finfo.proj);
    finfo.viewInv    = nvmath::inverse(finfo.view);
    finfo.camPos     = eye;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &finfo);

    // Update the sky
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(ProceduralSkyShaderParameters), &m_skyParams);

    // Reset maximum for current frame
    vkCmdFillBuffer(cmd, m_bHeatStats.buffer, (uint32_t(m_frame) & 1) * sizeof(uint32_t), sizeof(uint32_t), 1);

    m_pushConst.frame = m_frame;

    VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    memBarrier.srcAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
    memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1,
                         &memBarrier, 0, nullptr, 0, nullptr);

    // Ray trace
    std::vector<VkDescriptorSet> descSets{m_rtSet->getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    pushDescriptorSet(cmd);
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), &m_pushConst);

    const auto& regions = m_sbt.getRegions();
    const auto& size    = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &regions[0], &regions[1], &regions[2], &regions[3], size.width, size.height, 1);

    // Making sure the rendered image is ready to be used
    auto image_memory_barrier =
        nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(eImgRendered), VK_ACCESS_SHADER_READ_BIT,
                                     VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &image_memory_barrier);

    m_tonemapper->runCompute(cmd, size);
  }

private:
  void createScene()
  {
    constexpr int   num_obj       = 20;
    constexpr float obj_size      = 1.0F;
    constexpr float obj_spacing   = 2.0f;
    constexpr int   num_materials = 128;

    srand(12312412);

    // Meshes
    //m_meshes.emplace_back(nvh::cube(obj_size, obj_size, obj_size));
    m_meshes.emplace_back(nvh::createSphereMesh(obj_size));
    //m_meshes.emplace_back(nvh::tetrahedron());
    //m_meshes.emplace_back(nvh::octahedron());
    //m_meshes.emplace_back(nvh::icosahedron());
    //m_meshes.emplace_back(nvh::cone());
    // int num_meshes = static_cast<int>(m_meshes.size());

    // Materials (colorful)
    m_materials.reserve(num_materials);
    for(int i = 0; i < num_materials; i++)
    {
      const vec3 freq = vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
      vec3       v    = static_cast<vec3>(sin(freq) * 0.5F + 0.5F);
      m_materials.push_back({vec4(v, 1)});
    }

    auto inRange = [](int a, int b, int v0 = 10 - 3, int v1 = 10 + 2) {
      return (a >= v0 && a <= v1) && (b >= v0 && b <= v1);
    };

    // Instances
    m_nodes.reserve(static_cast<size_t>(num_obj * num_obj) * num_obj);
    bool skip{false};
    for(int k = 0; k < num_obj; k++)
    {
      for(int j = 0; j < num_obj; j++)
      {
        for(int i = 0; i < num_obj; i++)
        {
          bool center = inRange(i, j);
          center |= inRange(i, k);
          center |= inRange(k, j);
          if(!skip && !center)
          {
            auto& n       = m_nodes.emplace_back();
            n.mesh        = 0;
            n.material    = rand() % num_materials;  // (i * num_obj * num_obj) + (j * num_obj) + (k);
            n.translation = vec3(-(static_cast<float>(num_obj) / 2.F) + static_cast<float>(i),
                                 -(static_cast<float>(num_obj) / 2.F) + static_cast<float>(j),
                                 -(static_cast<float>(num_obj) / 2.F) + static_cast<float>(k));

            n.translation *= obj_spacing;
          }
          skip = !skip;
        }
        skip = !skip;
      }
      skip = !skip;
    }
    m_nodes.shrink_to_fit();

    // Adding a plane & material
    m_materials.push_back({vec4(.7F, .7F, .7F, 1.0F)});
    m_meshes.emplace_back(nvh::createPlane(10, 100, 100));
    auto& n       = m_nodes.emplace_back();
    n.mesh        = static_cast<int>(m_meshes.size()) - 1;
    n.material    = static_cast<int>(m_materials.size()) - 1;
    n.translation = {0.0f, static_cast<float>(-num_obj / 2 - 1) * obj_spacing, 0.0f};

    // Setting camera to see the scene
    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({0.0F, 2.0F, static_cast<float>(num_obj) * obj_spacing}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});

    // Default parameters for overall material
    m_pushConst.intensity             = 1.0F;
    m_pushConst.maxDepth              = 5;
    m_pushConst.roughness             = 0.05F;
    m_pushConst.metallic              = 0.5F;
    m_pushConst.frame                 = 0;
    m_pushConst.fireflyClampThreshold = 10;
    m_pushConst.maxSamples            = 2;
    m_pushConst.heatmap               = 0;


    // Default Sky values
    m_skyParams = initSkyShaderParameters();
  }


  void createGbuffers(const nvmath::vec2f& size)
  {
    // Rendering image targets
    m_viewSize                          = size;
    std::vector<VkFormat> color_buffers = {m_colorFormat, m_colorFormat};  // tonemapped, original
    m_gBuffers                          = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                   VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                   color_buffers, m_depthFormat);
  }

  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    auto* cmd = m_app->createTempCmdBuffer();
    m_bMeshes.resize(m_meshes.size());

    auto rtUsageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    std::vector<PrimMeshInfo> primInfo;
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      auto& m    = m_bMeshes[i];
      m.vertices = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rtUsageFlag);
      m.indices  = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rtUsageFlag);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);

      // To find the buffers of the mesh (buffer reference)
      PrimMeshInfo info{};
      info.vertexAddress = nvvk::getBufferDeviceAddress(m_device, m.vertices.buffer);
      info.indexAddress  = nvvk::getBufferDeviceAddress(m_device, m.indices.buffer);
      primInfo.emplace_back(info);
    }

    // Creating the buffer of all primitive information
    m_bPrimInfo = m_alloc->createBuffer(cmd, primInfo, rtUsageFlag);
    m_dutil->DBG_NAME(m_bPrimInfo.buffer);

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    // Create the buffer of sky parameters, updated at each frame
    m_bSkyParams = m_alloc->createBuffer(sizeof(ProceduralSkyShaderParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bSkyParams.buffer);

    // Create the buffer for the heatmap statistics
    m_bHeatStats = m_alloc->createBuffer(sizeof(ProceduralSkyShaderParameters), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_dutil->DBG_NAME(m_bHeatStats.buffer);

    // Primitive instance information
    std::vector<InstanceInfo> instInfo;
    for(auto& node : m_nodes)
    {
      InstanceInfo info{};
      info.transform  = node.localMatrix();
      info.materialID = node.material;
      instInfo.emplace_back(info);
    }
    m_bInstInfoBuffer =
        m_alloc->createBuffer(cmd, instInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bInstInfoBuffer.buffer);

    m_bMaterials = m_alloc->createBuffer(cmd, m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bMaterials.buffer);

    // Buffer references of all scene elements
    SceneDescription sceneDesc{};
    sceneDesc.materialAddress = nvvk::getBufferDeviceAddress(m_device, m_bMaterials.buffer);
    sceneDesc.primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_bPrimInfo.buffer);
    sceneDesc.instInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_bInstInfoBuffer.buffer);
    m_bSceneDesc              = m_alloc->createBuffer(cmd, sizeof(SceneDescription), &sceneDesc,
                                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bSceneDesc.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const nvh::PrimitiveMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    uint32_t maxPrimitiveCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat             = VK_FORMAT_R32G32B32A32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride             = sizeof(nvh::PrimitiveVertex);
    triangles.indexType                = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = indexAddress;
    triangles.maxVertex                = static_cast<uint32_t>(prim.vertices.size());
    //triangles.transformData; // Identity

    // Identify the above data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    asGeom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    asGeom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    asGeom.geometry.triangles = triangles;

    VkAccelerationStructureBuildRangeInfoKHR offset{};
    offset.firstVertex     = 0;
    offset.primitiveCount  = maxPrimitiveCount;
    offset.primitiveOffset = 0;
    offset.transformOffset = 0;

    // Our BLAS is made from only one geometry, but could be made of many geometries
    nvvk::RaytracingBuilderKHR::BlasInput input;
    input.asGeometry.emplace_back(asGeom);
    input.asBuildOffsetInfo.emplace_back(offset);

    return input;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
    allBlas.reserve(m_meshes.size());

    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto vertexAddress = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].vertices.buffer);
      auto indexAddress  = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].indices.buffer);

      auto geo = primitiveToGeometry(m_meshes[p_idx], vertexAddress, indexAddress);
      allBlas.push_back({geo});
    }
    m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    std::vector<VkAccelerationStructureInstanceKHR> tlas;
    tlas.reserve(m_nodes.size());
    for(auto& node : m_nodes)
    {
      VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};

      VkAccelerationStructureInstanceKHR rayInst{};
      rayInst.transform           = nvvk::toTransformMatrixKHR(node.localMatrix());  // Position of the instance
      rayInst.instanceCustomIndex = node.mesh;                                       // gl_InstanceCustomIndexEXT
      rayInst.accelerationStructureReference         = m_rtBuilder.getBlasDeviceAddress(node.mesh);
      rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
      rayInst.flags                                  = flags;
      rayInst.mask                                   = 0xFF;
      tlas.emplace_back(rayInst);
    }
    m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    m_rtPipe.destroy(m_device);
    m_rtSet->deinit();
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    auto& p = m_rtPipe;
    auto& d = m_rtSet;
    p.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set
    d->addBinding(BRtTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(BRtOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(BRtFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(BRtSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(BRtSkyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(BRtHeatStats, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    nvvk::Specialization specialization;
    specialization.add(0, m_useSER ? 1 : 0);

    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eClosestHit,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.pName = "main";  // All the same entry point
    // Raygen
    stage.module              = nvvk::createShaderModule(m_device, pathtrace_rgen, sizeof(pathtrace_rgen));
    stage.stage               = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stage.pSpecializationInfo = specialization.getSpecialization();
    stages[eRaygen]           = stage;

    m_dutil->setObjectName(stage.module, "Raygen");
    // Miss
    stage.module  = nvvk::createShaderModule(m_device, pathtrace_rmiss, sizeof(pathtrace_rmiss));
    stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss] = stage;
    m_dutil->setObjectName(stage.module, "Miss");
    // Hit Group - Closest Hit
    stage.module              = nvvk::createShaderModule(m_device, pathtrace_rchit, sizeof(pathtrace_rchit));
    stage.stage               = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stage.pSpecializationInfo = specialization.getSpecialization();
    stages[eClosestHit]       = stage;
    m_dutil->setObjectName(stage.module, "Closest Hit");


    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader       = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shaderGroups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    shaderGroups.push_back(group);

    // closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    shaderGroups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant)};

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {d->getLayout()};  // , m_pContainer[eGraphic].dstLayout};
    pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
    pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();
    vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &p.layout);
    m_dutil->DBG_NAME(p.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    rayPipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    rayPipelineInfo.pStages                      = stages.data();
    rayPipelineInfo.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
    rayPipelineInfo.pGroups                      = shaderGroups.data();
    rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
    rayPipelineInfo.layout                       = p.layout;
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &p.plines[0]);
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt.create(p.plines[0], rayPipelineInfo);

    // Removing temp modules
    for(auto& s : stages)
      vkDestroyShaderModule(m_device, s.module, nullptr);
  }


  void pushDescriptorSet(VkCommandBuffer cmd)
  {
    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    descASInfo.accelerationStructureCount = 1;
    descASInfo.pAccelerationStructures    = &tlas;
    VkDescriptorImageInfo  imageInfo{{}, m_gBuffers->getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbi_sky{m_bSkyParams.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo sceneDesc{m_bSceneDesc.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbi_heatstats{m_bHeatStats.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtSet->makeWrite(0, BRtTlas, &descASInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, BRtOutImage, &imageInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, BRtFrameInfo, &dbi_unif));
    writes.emplace_back(m_rtSet->makeWrite(0, BRtSceneDesc, &sceneDesc));
    writes.emplace_back(m_rtSet->makeWrite(0, BRtSkyParam, &dbi_sky));
    writes.emplace_back(m_rtSet->makeWrite(0, BRtHeatStats, &dbi_heatstats));
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());
  }


  //--------------------------------------------------------------------------------------------------
  // To be call when renderer need to re-start
  //
  void resetFrame() { m_frame = -1; }

  //--------------------------------------------------------------------------------------------------
  // If the camera matrix has changed, resets the frame.
  // otherwise, increments frame.
  //
  bool updateFrame()
  {
    static nvmath::mat4f ref_cam_matrix;
    static float         ref_fov{CameraManip.getFov()};

    const auto& m   = CameraManip.getMatrix();
    const auto  fov = CameraManip.getFov();

    if(memcmp(&ref_cam_matrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || ref_fov != fov)
    {
      resetFrame();
      ref_cam_matrix = m;
      ref_fov        = fov;
    }

    if(m_frame >= m_maxFrames)
    {
      return false;
    }
    m_frame++;
    return true;
  }


  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    for(auto& m : m_bMeshes)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bPrimInfo);
    m_alloc->destroy(m_bSceneDesc);
    m_alloc->destroy(m_bInstInfoBuffer);
    m_alloc->destroy(m_bMaterials);
    m_alloc->destroy(m_bSkyParams);
    m_alloc->destroy(m_bHeatStats);

    m_rtSet->deinit();
    m_gBuffers.reset();

    m_rtPipe.destroy(m_device);

    m_sbt.destroy();
    m_rtBuilder.destroy();
    m_tonemapper.reset();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                           m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>               m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>              m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer>  m_rtSet;  // Descriptor set
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;

  nvmath::vec2f                    m_viewSize    = {1, 1};
  VkFormat                         m_colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  ProceduralSkyShaderParameters    m_skyParams{};

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;
  nvvk::Buffer                 m_bFrameInfo;
  nvvk::Buffer                 m_bPrimInfo;
  nvvk::Buffer                 m_bSceneDesc;  // SceneDescription
  nvvk::Buffer                 m_bInstInfoBuffer;
  nvvk::Buffer                 m_bMaterials;
  nvvk::Buffer                 m_bSkyParams;
  nvvk::Buffer                 m_bHeatStats;

  bool m_useSER{false};

  // Data and setting
  struct Material
  {
    vec4 color{1.F};
  };
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;
  std::vector<Material>           m_materials;

  // Pipeline
  PushConstant     m_pushConst{};                        // Information sent to the shader
  VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  int              m_frame{0};
  int              m_maxFrames{10000};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper           m_sbt;  // Shading binding table wrapper
  nvvk::RaytracingBuilderKHR m_rtBuilder;
  nvvkhl::PipelineContainer  m_rtPipe;
};

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = PROJECT_NAME " Example";
  spec.vSync            = false;
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  spec.vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);  // To use vkCmdTraceRaysKHR
  spec.vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  VkPhysicalDeviceShaderClockFeaturesKHR clockFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME, false, &clockFeature);
  spec.vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  spec.vkSetup.addDeviceExtension("VK_NV_ray_tracing_invocation_reorder");

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());         // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());  // Window title info
  app->addElement(std::make_shared<SerPathtrace>());


  app->run();
  app.reset();

  return test->errorCode();
}
