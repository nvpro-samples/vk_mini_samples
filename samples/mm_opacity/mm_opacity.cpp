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

  This sample raytraces a plane made of 6x6 triangles with Micro-Mesh displacement
  - The scene is created in createScene()
  - Micro-mesh creation uses the MicromapProcess class
  - Vulkan buffers holding the scene are created in createVkBuffers()
  - Bottom and Top level acceleration structures are using the Vulkan buffers 
    and scene description in createBottomLevelAS() and createTopLevelAS()
  - The raytracing pipeline, composed of RayGen, Miss, ClosestHit shaders
    and the creation of the shader binding table, is done increateRtxPipeline()
  - Rendering is done in onRender()

  Note: search for #MICROMAP for specific changes for Micro-Mesh

*/
//////////////////////////////////////////////////////////////////////////

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#include <imgui/imgui.h>

#include <array>
#include <glm/detail/type_half.hpp>  // for half float
#include <glm/glm.hpp>

#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION


#include "shaders/dh_bindings.h"
#include "nvshaders/sky_io.h.slang"
#include "shaders/shaderio.h"  // Shared between host and device

#include "bird_curve_helper.hpp"
#include "common/utils.hpp"
#include "mm_process.hpp"

//#undef USE_HLSL


#include "_autogen/mm_opacity.rahit.glsl.h"
#include "_autogen/mm_opacity.rchit.glsl.h"
#include "_autogen/mm_opacity.rgen.glsl.h"
#include "_autogen/mm_opacity.rmiss.glsl.h"
#include "_autogen/mm_opacity.slang.h"

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/azimuth_sliders.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/property_editor.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/file_mapping.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/validation_settings.hpp>


// The camera for the scene
std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};

//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class MicomapOpacity : public nvapp::IAppElement
{
  struct Settings
  {
    float intensity{5.0F};
    float metallic{0.5F};
    float roughness{1.0F};
    int   maxDepth{5};
    // #MICROMAP
    bool     enableOpacity{true};
    int      subdivlevel{3};
    bool     showWireframe{true};
    float    radius{0.5F};
    bool     useAnyHit{true};
    uint16_t micromapFormat{VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT};
  } m_settings;


public:
  MicomapOpacity()           = default;
  ~MicomapOpacity() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    m_alloc.init(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    });  // Allocator

    m_micromap = std::make_unique<MicromapProcess>(&m_alloc);

    // Acquiring the sampler which will be used for displaying the GBuffer
    m_samplerPool.init(app->getDevice());
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    m_gBuffers.init({
        .allocator      = &m_alloc,
        .colorFormats   = {m_colorFormat},  // Only one GBuffer color attachment
        .depthFormat    = m_depthFormat,
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    });


    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    m_mmProperties.pNext = &m_asProperties;
    m_rtProperties.pNext = &m_mmProperties;
    prop2.pNext          = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shader Binding Table (SBT)
    const uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt.init(m_device, m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    // #MICROMAP
    {
      nvutils::ScopedTimer  stimer("Create MICROMAP");
      VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
      nvvk::StagingUploader uploader;
      uploader.init(&m_alloc);
      m_micromap->createMicromapData(cmd, uploader, m_meshes[0], m_settings.subdivlevel, m_settings.radius, m_settings.micromapFormat);
      uploader.cmdUploadAppended(cmd);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      m_micromap->cleanBuildData();
      uploader.deinit();
    }
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    m_gBuffers.update(cmd, size);

    writeRtDesc();
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      nvgui::CameraWidget(g_cameraManip);

      using namespace nvgui;

      // #MICROMAP - begin
      ImGui::Text("Micro-Mesh");
      PropertyEditor::begin();
      if(PropertyEditor::Checkbox("Enable", &m_settings.enableOpacity))
      {
        vkDeviceWaitIdle(m_device);
        destroyAccelerationStructures();
        createBottomLevelAS();
        createTopLevelAS();
        writeRtDesc();
      }

      bool subdiv_changed{false};
      subdiv_changed |= PropertyEditor::SliderInt("Subdivision Level", &m_settings.subdivlevel, 0,
                                                  m_mmProperties.maxOpacity4StateSubdivisionLevel);

      subdiv_changed |= PropertyEditor::SliderFloat("Radius", &m_settings.radius, 0.0F, 1.0F);

      subdiv_changed |= PropertyEditor::entry("Micro-map format", [&] {
        return ImGui::RadioButton("2-States", (int*)&m_settings.micromapFormat, VK_OPACITY_MICROMAP_FORMAT_2_STATE_EXT);
      });
      subdiv_changed |= PropertyEditor::entry("", [&] {
        return ImGui::RadioButton("4-States", (int*)&m_settings.micromapFormat, VK_OPACITY_MICROMAP_FORMAT_4_STATE_EXT);
      });

      PropertyEditor::Checkbox("Show Wireframe", &m_settings.showWireframe);
      PropertyEditor::Checkbox("Use AnyHit", &m_settings.useAnyHit);

      if(subdiv_changed)
      {
        nvutils::ScopedTimer stimer("Create MICROMAP");
        vkDeviceWaitIdle(m_device);

        VkCommandBuffer cmd = m_app->createTempCmdBuffer();

        nvvk::StagingUploader uploader;
        uploader.init(&m_alloc);

        // Recreate all values
        m_micromap->createMicromapData(cmd, uploader, m_meshes[0], m_settings.subdivlevel, m_settings.radius, m_settings.micromapFormat);
        uploader.cmdUploadAppended(cmd);
        m_app->submitAndWaitTempCmdBuffer(cmd);
        m_micromap->cleanBuildData();
        uploader.deinit();

        // Recreate the acceleration structure
        destroyAccelerationStructures();
        createBottomLevelAS();
        createTopLevelAS();
        writeRtDesc();
      }
      // #MICROMAP - end

      PropertyEditor::end();
      ImGui::Text("Material");
      PropertyEditor::begin();
      PropertyEditor::SliderFloat("Metallic", &m_settings.metallic, 0.0F, 1.0F);
      PropertyEditor::SliderFloat("Roughness", &m_settings.roughness, 0.0F, 1.0F);
      PropertyEditor::SliderFloat("Intensity", &m_settings.intensity, 0.0F, 10.0F);
      PropertyEditor::end();
      ImGui::Separator();
      ImGui::Text("Sun Orientation");
      PropertyEditor::begin();
      glm::vec3 dir = m_skyParams.sunDirection;
      nvgui::azimuthElevationSliders(dir, false, false);
      m_skyParams.sunDirection = dir;
      PropertyEditor::end();
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);


    // Update the uniform buffer containing frame info
    shaderio::FrameInfo finfo{};
    finfo.view    = g_cameraManip->getViewMatrix();
    finfo.proj    = g_cameraManip->getPerspectiveMatrix();
    finfo.projInv = glm::inverse(finfo.proj);
    finfo.viewInv = glm::inverse(finfo.view);
    finfo.camPos  = g_cameraManip->getEye();
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);

    // Update the sky
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(shaderio::SkySimpleParameters), &m_skyParams);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

    // Ray trace
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0, 1,
                            m_descriptorPack.getSetPtr(), 0, nullptr);

    m_pushConst.intensity = m_settings.intensity;
    m_pushConst.metallic  = m_settings.metallic;
    m_pushConst.roughness = m_settings.roughness;
    m_pushConst.maxDepth  = m_settings.maxDepth;
    m_pushConst.numBaseTriangles = m_settings.showWireframe ? (m_settings.enableOpacity ? 1 << m_settings.subdivlevel : 1) : 0;
    m_pushConst.radius    = m_settings.radius;
    m_pushConst.useAnyhit = m_settings.useAnyHit ? 1 : 0;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);

    const nvvk::SBTGenerator::Regions& regions = m_sbt.getSBTRegions();
    const VkExtent2D&                  size    = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &regions.raygen, &regions.miss, &regions.hit, &regions.callable, size.width, size.height, 1);
  }

private:
  void createScene()
  {
    // Adding a plane & material
    m_materials.push_back({glm::vec4(.7F, .7F, .7F, 1.0F)});
    m_meshes.emplace_back(nvutils::createPlane(3, 1.0F, 1.0F));
    nvutils::Node& node = m_nodes.emplace_back();
    node.mesh           = static_cast<int>(m_meshes.size()) - 1;
    node.material       = static_cast<int>(m_materials.size()) - 1;
    node.translation    = {0.0F, 0.0F, 0.0F};

    // Setting camera to see the scene
    g_cameraManip->setClipPlanes({0.01F, 100.0F});
    g_cameraManip->setLookat({0.96F, 1.33F, 1.3F}, {0.0F, 0.0F, 0.0F}, {0.0, 1.0F, 0.0F});

    // Default Sky values
    m_skyParams = {};
  }

  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);


    m_bMeshes.resize(m_meshes.size());

    const VkBufferUsageFlags rt_usage_flag = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_bMeshes[i];
      NVVK_CHECK(m_alloc.createBuffer(m.vertices, std::span(m_meshes[i].vertices).size_bytes(), rt_usage_flag));
      NVVK_CHECK(m_alloc.createBuffer(m.indices, std::span(m_meshes[i].triangles).size_bytes(), rt_usage_flag));
      NVVK_CHECK(uploader.appendBuffer(m.vertices, 0, std::span(m_meshes[i].vertices)));
      NVVK_CHECK(uploader.appendBuffer(m.indices, 0, std::span(m_meshes[i].triangles)));
      NVVK_DBG_NAME(m.vertices.buffer);
      NVVK_DBG_NAME(m.indices.buffer);
    }

    // Create the buffer of the current frame, changing at each frame
    NVVK_CHECK(m_alloc.createBuffer(m_bFrameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
    NVVK_DBG_NAME(m_bFrameInfo.buffer);

    // Create the buffer of sky parameters, updated at each frame
    NVVK_CHECK(m_alloc.createBuffer(m_bSkyParams, sizeof(shaderio::SkySimpleParameters),
                                    VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
    NVVK_DBG_NAME(m_bSkyParams.buffer);

    // Primitive instance information
    std::vector<shaderio::InstanceInfo> inst_info;
    inst_info.reserve(m_nodes.size());
    for(const nvutils::Node& node : m_nodes)
    {
      shaderio::InstanceInfo info{};
      info.transform  = node.localMatrix();
      info.materialID = node.material;
      inst_info.push_back(info);
    }

    NVVK_CHECK(m_alloc.createBuffer(m_bInstInfoBuffer, std::span(inst_info).size_bytes(),
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bInstInfoBuffer, 0, std::span(inst_info)));
    NVVK_DBG_NAME(m_bInstInfoBuffer.buffer);

    NVVK_CHECK(m_alloc.createBuffer(m_bMaterials, std::span(m_materials).size_bytes(),
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bMaterials, 0, std::span(m_materials)));
    NVVK_DBG_NAME(m_bMaterials.buffer);

    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    uploader.deinit();
  }


  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  static nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const nvutils::PrimitiveMesh& prim,
                                                                     VkDeviceAddress               vertexAddress,
                                                                     VkDeviceAddress               indexAddress)
  {
    const auto max_primitive_count = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride             = sizeof(nvutils::PrimitiveVertex);
    triangles.indexType                = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = indexAddress;
    triangles.maxVertex                = static_cast<uint32_t>(prim.vertices.size()) - 1;
    //triangles.transformData; // Identity

    nvvk::AccelerationStructureGeometryInfo result;
    // Identify the above data as containing opaque triangles.
    result.geometry = VkAccelerationStructureGeometryKHR{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry     = {triangles},
        .flags        = VK_GEOMETRY_OPAQUE_BIT_KHR | VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,
    };

    result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = max_primitive_count};

    return result;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    nvutils::ScopedTimer stimer("Create BLAS");
    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::AccelerationStructureBuildData> blasBuildData;
    blasBuildData.reserve(m_meshes.size());
    m_blas.resize(m_meshes.size());


    // #MICROMAP
    assert(m_meshes.size() == 1);  // The micromap is created for only one mesh
    std::vector<VkAccelerationStructureTrianglesOpacityMicromapEXT> geometry_opacity;  // hold data until BLAS is created
    geometry_opacity.reserve(m_meshes.size());

    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      nvvk::AccelerationStructureBuildData buildData{VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};

      const VkDeviceAddress vertex_address = m_bMeshes[p_idx].vertices.address;
      const VkDeviceAddress index_address  = m_bMeshes[p_idx].indices.address;

      nvvk::AccelerationStructureGeometryInfo geo = primitiveToGeometry(m_meshes[p_idx], vertex_address, index_address);

      // #MICROMAP
      VkAccelerationStructureTrianglesOpacityMicromapEXT opacity_geometry_micromap = {
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT};

      if(m_settings.enableOpacity)
      {
        const VkDeviceAddress indexT_address = m_micromap->indexBuffer().address;

        opacity_geometry_micromap.indexType                 = VK_INDEX_TYPE_UINT32;
        opacity_geometry_micromap.indexBuffer.deviceAddress = indexT_address;
        opacity_geometry_micromap.indexStride               = sizeof(int32_t);
        opacity_geometry_micromap.baseTriangle              = 0;
        opacity_geometry_micromap.micromap                  = m_micromap->micromap();

        // Adding micromap
        geometry_opacity.emplace_back(opacity_geometry_micromap);
        geo.geometry.geometry.triangles.pNext = &geometry_opacity.back();
      }

      buildData.addGeometry(geo);
      auto sizeInfo = buildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
      blasBuildData.push_back(buildData);
    }

    VkDeviceSize maxScratchSize = nvvk::getMaxScratchSize(blasBuildData);
    // Scratch buffer
    nvvk::Buffer scratchBuffer;
    NVVK_CHECK(m_alloc.createBuffer(scratchBuffer, maxScratchSize,
                                    VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
                                        | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR));
    NVVK_DBG_NAME(scratchBuffer.buffer);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      VkAccelerationStructureCreateInfoKHR createInfo = blasBuildData[p_idx].makeCreateInfo();
      NVVK_CHECK(m_alloc.createAcceleration(m_blas[p_idx], createInfo));
      NVVK_DBG_NAME(m_blas[p_idx].accel);
      blasBuildData[p_idx].cmdBuildAccelerationStructure(cmd, m_blas[p_idx].accel, scratchBuffer.address);
    }

    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_alloc.destroyBuffer(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    nvutils::ScopedTimer stimer("Create TLAS");

    // #MICROMAP
    const VkBuildAccelerationStructureFlagsKHR buildFlags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
        | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_OPACITY_MICROMAP_UPDATE_EXT;

    nvvk::AccelerationStructureBuildData            tlasBuildData{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(m_nodes.size());
    for(const nvutils::Node& node : m_nodes)
    {
      const VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};
      //flags |= VK_GEOMETRY_INSTANCE_FORCE_OPACITY_MICROMAP_2_STATE_EXT; // #MICROMAP

      VkAccelerationStructureInstanceKHR ray_inst{};
      ray_inst.transform           = nvvk::toTransformMatrixKHR(node.localMatrix());  // Position of the instance
      ray_inst.instanceCustomIndex = node.mesh;                                       // gl_InstanceCustomIndexEXT
      ray_inst.accelerationStructureReference         = m_blas[node.mesh].address;
      ray_inst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
      ray_inst.flags                                  = flags;
      ray_inst.mask                                   = 0xFF;
      tlasInstances.emplace_back(ray_inst);
    }

    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);


    nvvk::Buffer instanceBuffer;
    NVVK_CHECK(m_alloc.createBuffer(instanceBuffer, std::span(tlasInstances).size_bytes(),
                                    VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                        | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
    NVVK_CHECK(uploader.appendBuffer(instanceBuffer, 0, std::span(tlasInstances)));
    NVVK_DBG_NAME(instanceBuffer.buffer);
    uploader.cmdUploadAppended(cmd);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    auto geo = tlasBuildData.makeInstanceGeometry(tlasInstances.size(), instanceBuffer.address);
    tlasBuildData.addGeometry(geo);
    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, buildFlags);

    // Scratch buffer
    nvvk::Buffer scratchBuffer;
    NVVK_CHECK(m_alloc.createBuffer(scratchBuffer, sizeInfo.buildScratchSize,
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                        | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR));
    NVVK_DBG_NAME(scratchBuffer.buffer);

    VkAccelerationStructureCreateInfoKHR createInfo = tlasBuildData.makeCreateInfo();
    NVVK_CHECK(m_alloc.createAcceleration(m_tlas, createInfo));
    NVVK_DBG_NAME(m_tlas.accel);
    tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, scratchBuffer.address);

    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();

    m_alloc.destroyBuffer(scratchBuffer);
    m_alloc.destroyBuffer(instanceBuffer);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_skyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_vertex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);

    // Create the descriptor set layout, pool with space for 1 set, and allocate that set
    NVVK_CHECK(m_descriptorPack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_descriptorPack.getLayout());
    NVVK_DBG_NAME(m_descriptorPack.getPool());
    NVVK_DBG_NAME(m_descriptorPack.getSet(0));

    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eClosestHit,
      eAnyHit,
      eShaderGroupCount
    };

    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    for(auto& s : stages)
    {
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    }

#if USE_SLANG
    VkShaderModuleCreateInfo moduleInfo = nvsamples::getShaderModuleCreateInfo(mm_opacity_slang);

    stages[eRaygen].pNext     = &moduleInfo;
    stages[eRaygen].pName     = "rgenMain";
    stages[eRaygen].stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].pNext       = &moduleInfo;
    stages[eMiss].pName       = "rmissMain";
    stages[eMiss].stage       = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].pNext = &moduleInfo;
    stages[eClosestHit].pName = "rchitMain";
    stages[eClosestHit].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eAnyHit].pNext     = &moduleInfo;
    stages[eAnyHit].pName     = "rahitMain";
    stages[eAnyHit].stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
#else
    VkShaderModuleCreateInfo rgenModuleInfo = nvsamples::getShaderModuleCreateInfo(mm_opacity_rgen_glsl);
    stages[eRaygen].pNext                   = &rgenModuleInfo;
    stages[eRaygen].pName                   = "main";
    stages[eRaygen].stage                   = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    VkShaderModuleCreateInfo rmissModuleInfo = nvsamples::getShaderModuleCreateInfo(mm_opacity_rmiss_glsl);
    stages[eMiss].pNext                      = &rmissModuleInfo;
    stages[eMiss].pName                      = "main";
    stages[eMiss].stage                      = VK_SHADER_STAGE_MISS_BIT_KHR;

    VkShaderModuleCreateInfo rchitModuleInfo = nvsamples::getShaderModuleCreateInfo(mm_opacity_rchit_glsl);
    stages[eClosestHit].pNext                = &rchitModuleInfo;
    stages[eClosestHit].pName                = "main";
    stages[eClosestHit].stage                = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

    VkShaderModuleCreateInfo rahitModuleInfo = nvsamples::getShaderModuleCreateInfo(mm_opacity_rahit_glsl);
    stages[eAnyHit].pNext                    = &rahitModuleInfo;
    stages[eAnyHit].pName                    = "main";
    stages[eAnyHit].stage                    = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
#endif

    // #MICROMAP

    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader       = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shader_groups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    shader_groups.push_back(group);

    // Hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    group.anyHitShader     = eAnyHit;
    shader_groups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    const VkPushConstantRange push_constant{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(shaderio::PushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorPack.getLayout()}, {push_constant}));
    NVVK_DBG_NAME(m_pipelineLayout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.flags      = VK_PIPELINE_CREATE_RAY_TRACING_OPACITY_MICROMAP_BIT_EXT;  // #MICROMAP
    ray_pipeline_info.stageCount = static_cast<uint32_t>(stages.size());                     // Stages are shaders
    ray_pipeline_info.pStages    = stages.data();
    ray_pipeline_info.groupCount = static_cast<uint32_t>(shader_groups.size());
    ray_pipeline_info.pGroups    = shader_groups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = 10;  // Ray depth
    ray_pipeline_info.layout                       = m_pipelineLayout;
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, &m_pipeline));
    NVVK_DBG_NAME(m_pipeline);

    // Creating the SBT
    // Prepare SBT data from ray pipeline
    size_t bufferSize = m_sbt.calculateSBTBufferSize(m_pipeline, ray_pipeline_info);

    // Create SBT buffer using the size from above
    NVVK_CHECK(m_alloc.createBuffer(m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                    m_sbt.getBufferAlignment()));
    NVVK_DBG_NAME(m_sbtBuffer.buffer);

    // Pass the manual mapped pointer to fill the SBT data
    NVVK_CHECK(m_sbt.populateSBTBuffer(m_sbtBuffer.address, bufferSize, m_sbtBuffer.mapping));


    // Removing temp modules
#if USE_SLANG
#else
    for(const VkPipelineShaderStageCreateInfo& s : stages)
      vkDestroyShaderModule(m_device, s.module, nullptr);
#endif
  }

  void writeRtDesc()
  {
    // Write to descriptors
    std::vector<VkDescriptorBufferInfo> vertex_desc;
    std::vector<VkDescriptorBufferInfo> index_desc;
    vertex_desc.reserve(m_bMeshes.size());
    index_desc.reserve(m_bMeshes.size());
    for(auto& m : m_bMeshes)
    {
      vertex_desc.push_back({m.vertices.buffer, 0, VK_WHOLE_SIZE});
      index_desc.push_back({m.indices.buffer, 0, VK_WHOLE_SIZE});
    }

    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descriptorPack.makeWrite(B_tlas), m_tlas);
    writeContainer.append(m_descriptorPack.makeWrite(B_outImage), m_gBuffers.getColorImageView(), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(m_descriptorPack.makeWrite(B_frameInfo), m_bFrameInfo);
    writeContainer.append(m_descriptorPack.makeWrite(B_skyParam), m_bSkyParams);
    writeContainer.append(m_descriptorPack.makeWrite(B_materials), m_bMaterials);
    writeContainer.append(m_descriptorPack.makeWrite(B_instances), m_bInstInfoBuffer);
    writeContainer.append(m_descriptorPack.makeWrite(B_vertex), vertex_desc.data());
    writeContainer.append(m_descriptorPack.makeWrite(B_index), index_desc.data());


    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }

  void destroyAccelerationStructures()
  {
    for(auto& blas : m_blas)
    {
      m_alloc.destroyAcceleration(blas);
    }
    m_blas.clear();
    m_alloc.destroyAcceleration(m_tlas);
  }

  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    m_descriptorPack.deinit();


    for(PrimitiveMeshVk& m : m_bMeshes)
    {
      m_alloc.destroyBuffer(m.vertices);
      m_alloc.destroyBuffer(m.indices);
    }
    m_alloc.destroyBuffer(m_bFrameInfo);
    m_alloc.destroyBuffer(m_bInstInfoBuffer);
    m_alloc.destroyBuffer(m_bMaterials);
    m_alloc.destroyBuffer(m_bSkyParams);
    m_alloc.destroyBuffer(m_sbtBuffer);


    destroyAccelerationStructures();
    m_micromap.reset();

    m_gBuffers.deinit();
    m_sbt.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*              m_app{nullptr};
  nvvk::ResourceAllocator          m_alloc;
  nvvk::GBuffer                    m_gBuffers;  // G-Buffers: color + depth
  std::unique_ptr<MicromapProcess> m_micromap;


  VkFormat m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkDevice m_device      = VK_NULL_HANDLE;                 // Convenient

  shaderio::SkySimpleParameters m_skyParams{};

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;
  nvvk::Buffer                 m_bFrameInfo;
  nvvk::Buffer                 m_bInstInfoBuffer;
  nvvk::Buffer                 m_bMaterials;
  nvvk::Buffer                 m_bSkyParams;

  nvvk::SamplerPool m_samplerPool{};  // The sampler pool, used to create a sampler for the texture

  // Data and setting
  struct Material
  {
    glm::vec4 color{1.F};
  };
  std::vector<nvutils::PrimitiveMesh> m_meshes;
  std::vector<nvutils::Node>          m_nodes;
  std::vector<Material>               m_materials;

  // Pipeline
  shaderio::PushConstant m_pushConst{};                      // Information sent to the shader
  VkPipelineLayout       m_pipelineLayout = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline             m_pipeline       = VK_NULL_HANDLE;  // The graphic pipeline to render
  nvvk::DescriptorPack   m_descriptorPack{};                 // Descriptor bindings, layout, pool, and set


  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceOpacityMicromapPropertiesEXT m_mmProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_PROPERTIES_EXT};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_asProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  nvvk::SBTGenerator m_sbt;  // Shader binding table wrapper
  nvvk::Buffer       m_sbtBuffer;

  std::vector<nvvk::AccelerationStructure> m_blas;  // Hold the bottom-level AS
  nvvk::AccelerationStructure              m_tlas;  // Top-level acceleration structure
};

//////////////////////////////////////////////////////////////////////////
///
///
///
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;  // Base application information

  // Command line parsing
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  // #MICROMAP
  nvvk::ValidationSettings vvl{
      .unique_handles = false,  // This is required for the validation layers to work properly
  };

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  // #MICROMAP
  VkPhysicalDeviceOpacityMicromapFeaturesEXT mm_opacity_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT};

  // Setting how we want Vulkan context to be created
  nvvk::ContextInitInfo vkSetup;
  vkSetup.instanceCreateInfoExt = vvl.buildPNextChain();
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accel_feature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rt_pipeline_feature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, &mm_opacity_features});


#if(VK_HEADER_VERSION >= 283)
  // To enable ray tracing validation, set the NV_ALLOW_RAYTRACING_VALIDATION=1 environment variable
  // https://developer.nvidia.com/blog/ray-tracing-validation-at-the-driver-level/
  // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_ray_tracing_validation.html
  VkPhysicalDeviceRayTracingValidationFeaturesNV validationFeatures = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_VALIDATION_FEATURES_NV};
  vkSetup.deviceExtensions.push_back({VK_NV_RAY_TRACING_VALIDATION_EXTENSION_NAME, &validationFeatures, false});
#endif


  // Creation of the Vulkan context
  nvvk::Context vkContext;  // Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }


  // Setting up the application
  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = false;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // #MICROMAP
  if(mm_opacity_features.micromap == VK_FALSE)
  {
    LOGE("ERROR: Micro-Mesh not supported");
    exit(1);
  }


  // Camera manipulator (global)
  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(g_cameraManip);


  // Add all application elements
  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());  // Menu / Quit
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app.addElement(std::make_shared<MicomapOpacity>());


  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
