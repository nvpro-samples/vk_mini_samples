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

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION  // Implementation of the image loading library
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define IMGUI_DEFINE_MATH_OPERATORS  // ImGUI ImVec maths
#include <glm/glm.hpp>

#include <imgui/imgui.h>

#include "shaders/shaderio_rt_gltf.h"

#include <filesystem>

// Local shaders
#include "_autogen/raytrace.slang.h"
#include "_autogen/tonemapper.slang.h"


// All the includes
#include "common/utils.hpp"
#include "nvvk/validation_settings.hpp"
#include <backends/imgui_impl_vulkan.h>
#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/axis.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/tonemapper.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <nvslang/slang.hpp>
#include <nvutils/bounding_box.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/primitives.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/staging.hpp>
#include <nvvkgltf/tinygltf_utils.hpp>

// The maximum depth recursion for the ray tracer
uint32_t MAXRAYRECURSIONDEPTH = 10;


std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};


struct GltfMesh
{
  shaderio::TriangleMesh mesh;
  VkIndexType            indexType = {};
  nvvk::Buffer           gltfBuffer;
  nvvk::Buffer           accessor;  // TriangleMesh as buffer
  nvutils::Bbox          bbox;
};

struct GltfInstance
{
  uint32_t  meshIndex{};
  glm::mat4 transform;
  glm::vec3 color{1, 1, 1};
};

// Main class for the sample
struct RaytraceGltfSettings
{
  bool showAxis = true;
};


//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class Raytracing : public nvapp::IAppElement
{
public:
  Raytracing()           = default;
  ~Raytracing() override = default;

  void onAttach(nvapp::Application* app) override
  {
    SCOPED_TIMER(__FUNCTION__);
    m_app = app;

    std::filesystem::path exePath = nvutils::getExecutablePath().parent_path();
    std::filesystem::path exeName = nvutils::getExecutablePath().stem();

    // Slang compiler
    {
      m_slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
      m_slangCompiler.defaultTarget();
      m_slangCompiler.defaultOptions();
      m_slangCompiler.addOption({slang::CompilerOptionName::DebugInformation, {slang::CompilerOptionValueKind::Int, 1}});
      m_slangCompiler.addOption({slang::CompilerOptionName::Optimization, {slang::CompilerOptionValueKind::Int, 0}});
    }

    // Default camera
    g_cameraManip->setClipPlanes({0.1F, 100.0F});
    g_cameraManip->setLookat({-6.0f, 3.0f, 7.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    g_cameraManip->setFov(45.f);

    // Initialize the VMA allocator
    m_allocator.init({
        .physicalDevice   = m_app->getPhysicalDevice(),
        .device           = m_app->getDevice(),
        .instance         = m_app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });
    // Finding leak
    // m_allocator.setLeakID(22); // Set to a number to enable leak checking (SBT Buffer)

    // Tonemapper
    {
      auto code = std::span<const uint32_t>(tonemapper_slang);
      m_tonemapper.init(&m_allocator, code);
    }

    // Initialize the utility for uploads
    m_stagingUploader.init(&m_allocator, true);

    // Acquiring the sampler which will be used for displaying the GBuffer
    m_samplerPool.init(app->getDevice());
    VkSampler linearSampler;
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // Create the G-Buffer
    nvvk::GBufferInitInfo gBufferInit{
        .allocator    = &m_allocator,
        .colorFormats = {VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM},  // Only one GBuffer color attachment
        .depthFormat  = nvvk::findDepthFormat(m_app->getPhysicalDevice()),
        .imageSampler = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    };
    m_gBuffers.init(gBufferInit);

    // Get ray tracing properties
    m_rtProp.pNext = &m_accelStructProps;
    VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProp};
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    m_pushConst.maxDepth = std::min(m_rtProp.maxRayRecursionDepth, 5u);

    // Load the glTF file
    m_meshes.push_back(importGltfResources(loadGltfResources(nvutils::findFile("teapot.gltf", nvsamples::getResourcesDirs()))));
    m_meshes.push_back(importGltfResources(loadGltfResources(nvutils::findFile("plane.gltf", nvsamples::getResourcesDirs()))));
    m_instances.push_back({0 /*teapot*/, glm::mat4(1), glm::vec3(.2f, .8f, .2f)});
    m_instances.push_back({1 /*plane*/, glm::translate(glm::mat4(1), glm::vec3(0, -2, 0)), glm::vec3(.6f)});

    createResources();
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();

    // Fit the camera to the scene
    nvutils::Bbox sceneBB;
    for(auto& m : m_meshes)
    {
      sceneBB.insert(m.bbox);
    }
    g_cameraManip->fit(sceneBB.min(), sceneBB.max());
    g_cameraManip->setSpeed(sceneBB.radius());
  }

  //---------------------------------------------------------------------------------------------------------------
  // Destroying the resources
  //
  void onDetach()
  {
    vkQueueWaitIdle(m_app->getQueue(0).queue);
    auto device = m_app->getDevice();

    m_allocator.destroyBuffer(m_bSceneInfo);
    m_allocator.destroyBuffer(m_bInstInfo);
    m_allocator.destroyBuffer(m_bMeshInfo);
    m_allocator.destroyBuffer(m_sbtBuffer);
    for(auto& mesh : m_meshes)
    {
      m_allocator.destroyBuffer(mesh.gltfBuffer);
      m_allocator.destroyBuffer(mesh.accessor);
    }

    m_allocator.destroyAcceleration(m_tlas);
    for(auto& blas : m_blas)
    {
      m_allocator.destroyAcceleration(blas);
    }

    m_allocator.destroyBuffer(m_sbtBuffer);

    vkDestroyPipeline(device, m_rtPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_rtPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, m_rtDescriptorSetLayout, nullptr);

    m_tonemapper.deinit();
    m_gBuffers.deinit();
    m_samplerPool.deinit();
    m_stagingUploader.deinit();
    m_allocator.deinit();

    m_app = nullptr;
  }

  //---------------------------------------------------------------------------------------------------------------
  // When the viewport is resized, the GBuffer must be resized
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) { NVVK_CHECK(m_gBuffers.update(cmd, size)); }

  //---------------------------------------------------------------------------------------------------------------
  // The rendering function, called each frame
  void onRender(VkCommandBuffer cmd)
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    {
      nvvk::DebugUtil::ScopedCmdLabel scopedCmdLabel(cmd, "RaytraceGltf");

      // Update uniform buffers
      m_sceneInfo.projInvMatrix = glm::inverse(g_cameraManip->getPerspectiveMatrix());
      m_sceneInfo.viewInvMatrix = glm::inverse(g_cameraManip->getViewMatrix());
      vkCmdUpdateBuffer(cmd, m_bSceneInfo.buffer, 0, sizeof(shaderio::RtGltfSceneInfo), &m_sceneInfo);
      nvvk::cmdBufferMemoryBarrier(cmd, {m_bSceneInfo.buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR});

      // Update the shader information
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
      pushDescSet(cmd);
      vkCmdPushConstants(cmd, m_rtPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::RtGltfPushConstant), &m_pushConst);

      // Ray trace
      const VkExtent2D& size = m_app->getViewportSize();
      vkCmdTraceRaysKHR(cmd, &m_sbtRegions.raygen, &m_sbtRegions.miss, &m_sbtRegions.hit, &m_sbtRegions.callable,
                        size.width, size.height, 1);
    }

    // Barrier to make sure the image is ready for Tonemapping
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Tonemap the image
    m_tonemapper.runCompute(cmd, m_gBuffers.getSize(), m_tonemapperData, m_gBuffers.getDescriptorImageInfo(0),
                            m_gBuffers.getDescriptorImageInfo(1));
  }

  //---------------------------------------------------------------------------------------------------------------
  // The ImGui rendering function, called each frame
  void onUIRender()
  {
    namespace PE = nvgui::PropertyEditor;

    if(ImGui::Begin("Viewport"))
    {
      ImGui::Image(ImTextureID(m_gBuffers.getDescriptorSet(1)), ImGui::GetContentRegionAvail());

      // Adding Axis at the bottom left corner of the viewport
      if(m_settings.showAxis)
      {
        nvgui::Axis(g_cameraManip->getViewMatrix(), 25.f);
      }
      ImGui::End();
    }

    // This is the settings window, which is docked to the left of the viewport
    if(ImGui::Begin("Settings"))
    {
      nvgui::CameraWidget(g_cameraManip);

      ImGui::SeparatorText("Settings");

      if(ImGui::TreeNode("Tone-mapper"))
      {
        nvgui::tonemapperWidget(m_tonemapperData);
        ImGui::TreePop();
      }
      if(ImGui::TreeNode("Sky"))
      {
        nvgui::skyPhysicalParameterUI(m_sceneInfo.skyParams);
        ImGui::TreePop();
      }
      PE::begin();
      PE::SliderFloat("Metallic", &m_pushConst.metallic, 0.0f, 1.0f);
      PE::SliderFloat("Roughness", &m_pushConst.roughness, 0.01f, 1.0f);
      PE::end();


      ImGui::End();
    }


    if(ImGui::IsKeyPressed(ImGuiKey_F5))
    {
      vkQueueWaitIdle(m_app->getQueue(0).queue);
      createRtxPipeline();
    }
  }


  // Create the resources used by the application: vertex buffer, the data buffer, the scene information buffer, and the images
  void createResources()
  {
    SCOPED_TIMER(__FUNCTION__);

    m_allocator.destroyBuffer(m_bSceneInfo);
    m_allocator.destroyBuffer(m_bInstInfo);
    m_allocator.destroyBuffer(m_bMeshInfo);

    // Create a buffer (UBO) to store the scene information
    NVVK_CHECK(m_allocator.createBuffer(m_bSceneInfo, sizeof(shaderio::RtGltfSceneInfo),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO));
    NVVK_DBG_NAME(m_bSceneInfo.buffer);

    assert(m_stagingUploader.isAppendedEmpty());
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    {
      // Create the mesh access data
      std::vector<shaderio::RtMeshInfo> meshes;
      for(auto& m : m_meshes)
      {
        meshes.push_back({(shaderio::TriangleMesh*)m.accessor.address, m.gltfBuffer.address});
      }
      NVVK_CHECK(m_allocator.createBuffer(m_bMeshInfo, std::span(meshes).size_bytes(), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_bMeshInfo, 0, std::span(meshes)));
      NVVK_DBG_NAME(m_bMeshInfo.buffer);

      // Create the instance access data
      NVVK_CHECK(m_allocator.createBuffer(m_bInstInfo, std::span(m_instances).size_bytes(), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_bInstInfo, 0, std::span(m_instances)));
      NVVK_DBG_NAME(m_bInstInfo.buffer);
    }
    m_stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Update the scene information
    m_sceneInfo.instances = (shaderio::RtInstanceInfo*)m_bInstInfo.address;
    m_sceneInfo.meshes    = (shaderio::RtMeshInfo*)m_bMeshInfo.address;
  }

  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const GltfMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    nvvk::AccelerationStructureGeometryInfo result = {};

    const auto triangleCount = static_cast<uint32_t>(prim.mesh.indices.count / 3U);

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,  // vec3 vertex position data
        .vertexData   = {.deviceAddress = vertexAddress},
        .vertexStride = prim.mesh.positions.byteStride,
        .maxVertex    = prim.mesh.positions.count - 1,
        .indexType    = prim.indexType,
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
    SCOPED_TIMER(__FUNCTION__);

    for(auto& b : m_blas)
    {
      m_allocator.destroyAcceleration(b);
    }

    size_t   numMeshes = m_meshes.size();
    VkDevice device    = m_app->getDevice();

    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::AccelerationStructureBuildData> blasBuildData;
    blasBuildData.reserve(numMeshes);
    m_blas.resize(numMeshes);  // All BLAS

    // Get the build information for all the BLAS
    for(uint32_t p_idx = 0; p_idx < numMeshes; p_idx++)
    {
      nvvk::AccelerationStructureBuildData buildData{VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR};

      VkDeviceAddress       baseBufferAddress   = m_meshes[p_idx].gltfBuffer.address;
      const VkDeviceAddress vertexBufferAddress = baseBufferAddress + m_meshes[p_idx].mesh.positions.offset;
      const VkDeviceAddress indexBufferAddress  = baseBufferAddress + m_meshes[p_idx].mesh.indices.offset;

      auto geo = primitiveToGeometry(m_meshes[p_idx], vertexBufferAddress, indexBufferAddress);
      buildData.addGeometry(geo);


      buildData.finalizeGeometry(device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                             | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);

      blasBuildData.emplace_back(buildData);
    }

    // Find the most optimal size for our scratch buffer, and get the addresses of the scratch buffers
    // to allow a maximum of BLAS to be built in parallel, within the budget
    nvvk::AccelerationStructureBuilder blasBuilder;
    blasBuilder.init(&m_allocator);
    VkDeviceSize hintScratchBudget = 2'000'000;  // Limiting the size of the scratch buffer to 2MB
    VkDeviceSize scratchSize       = blasBuilder.getScratchSize(hintScratchBudget, blasBuildData);

    const VkDeviceSize alignment = m_accelStructProps.minAccelerationStructureScratchOffsetAlignment;
    nvvk::Buffer       scratchBuffer;
    NVVK_CHECK(m_allocator.createBuffer(scratchBuffer, scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                        VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT, alignment));

    // Start the build and compaction of the BLAS
    VkDeviceSize hintBuildBudget = 2'000'000;  // Limiting the size of the scratch buffer to 2MB
    bool         finished        = false;
    LOGI("\n");

    std::span<nvvk::AccelerationStructureBuildData> buildDataSpan = blasBuildData;
    std::span<nvvk::AccelerationStructure>          blasSpan      = m_blas;

    do
    {
      {
        // Create, build and query the size of the BLAS, up to the 2MBi
        VkCommandBuffer cmd    = m_app->createTempCmdBuffer();
        VkResult        result = blasBuilder.cmdCreateBlas(cmd, buildDataSpan, blasSpan, scratchBuffer.address,
                                                           scratchBuffer.bufferSize, hintBuildBudget);
        if(result == VK_SUCCESS)
        {
          finished = true;
        }
        else if(result != VK_INCOMPLETE)
        {
          // Any result other than VK_SUCCESS or VK_INCOMPLETE is an error
          assert(0 && "Error building BLAS");
        }
        // VK_INCOMPLETE means continue the loop
        m_app->submitAndWaitTempCmdBuffer(cmd);
      }
      {
        // Compacting the BLAS, and destroy the previous ones
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();
        blasBuilder.cmdCompactBlas(cmd, buildDataSpan, blasSpan);
        m_app->submitAndWaitTempCmdBuffer(cmd);
        blasBuilder.destroyNonCompactedBlas();
      }
    } while(!finished);

    // Giving a name to the BLAS
    for(size_t i = 0; i < m_blas.size(); i++)
    {
      NVVK_DBG_NAME(m_blas[i].accel);
    }

    // Statistics
    std::string indent = nvutils::ScopedTimer::indent();
    std::string stats  = blasBuilder.getStatistics().toString();
    LOGI("%s%s\n", indent.c_str(), stats.c_str());

    // Cleanup
    m_allocator.destroyBuffer(scratchBuffer);
    blasBuilder.deinit();
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    SCOPED_TIMER(__FUNCTION__);

    m_allocator.destroyAcceleration(m_tlas);

    VkDevice device = m_app->getDevice();

    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(m_instances.size());
    for(const GltfInstance& node : m_instances)
    {
      VkAccelerationStructureInstanceKHR tlasInst{
          .transform           = nvvk::toTransformMatrixKHR(node.transform),  // Position of the instance
          .instanceCustomIndex = static_cast<uint32_t>(node.meshIndex),       // gl_InstanceCustomIndexEX
          .mask                = 0xFF,                                        // All objects
          .instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all object
          .flags = VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR | VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
          .accelerationStructureReference = m_blas[node.meshIndex].address,
      };
      tlasInstances.emplace_back(tlasInst);
    }

    assert(m_stagingUploader.isAppendedEmpty());
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    // Create the instances buffer, add a barrier to ensure the data is copied before the TLAS build
    nvvk::Buffer instancesBuffer;
    NVVK_CHECK(m_allocator.createBuffer(instancesBuffer, std::span<VkAccelerationStructureInstanceKHR>(tlasInstances).size_bytes(),
                                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(m_stagingUploader.appendBuffer(instancesBuffer, 0, std::span<VkAccelerationStructureInstanceKHR>(tlasInstances)));
    NVVK_DBG_NAME(instancesBuffer.buffer);
    m_stagingUploader.cmdUploadAppended(cmd);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);


    nvvk::AccelerationStructureBuildData    tlasBuildData{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
    nvvk::AccelerationStructureGeometryInfo geometryInfo =
        tlasBuildData.makeInstanceGeometry(tlasInstances.size(), instancesBuffer.address);
    tlasBuildData.addGeometry(geometryInfo);
    // Get the size of the TLAS
    auto sizeInfo = tlasBuildData.finalizeGeometry(device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    // Create the scratch buffer
    nvvk::Buffer scratchBuffer;
    NVVK_CHECK(m_allocator.createBuffer(scratchBuffer, sizeInfo.buildScratchSize,
                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
                                        m_accelStructProps.minAccelerationStructureScratchOffsetAlignment));

    // Create the TLAS
    NVVK_CHECK(m_allocator.createAcceleration(m_tlas, tlasBuildData.makeCreateInfo()));
    NVVK_DBG_NAME(m_tlas.accel);
    tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, scratchBuffer.address);


    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    m_allocator.destroyBuffer(scratchBuffer);
    m_allocator.destroyBuffer(instancesBuffer);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    SCOPED_TIMER(__FUNCTION__);

    VkDevice device = m_app->getDevice();

    if(!m_slangCompiler.compileFile("raytrace.slang"))
    {
      LOGE("Error compiling gltf.rast.slang\n");
      return;
    }

    // Clean the previous pipeline
    vkDestroyPipeline(device, m_rtPipeline, nullptr);
    vkDestroyPipelineLayout(device, m_rtPipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, m_rtDescriptorSetLayout, nullptr);

    // Create the descriptor set layout
    m_rtDescriptorBindings.clear();
    m_rtDescriptorBindings.addBinding({.binding         = shaderio::BindingIndex::eTlas,
                                       .descriptorType  = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                                       .descriptorCount = 1,
                                       .stageFlags      = VK_SHADER_STAGE_ALL});
    m_rtDescriptorBindings.addBinding({.binding         = shaderio::BindingIndex::eOutImage,
                                       .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                       .descriptorCount = 1,
                                       .stageFlags      = VK_SHADER_STAGE_ALL});
    m_rtDescriptorBindings.addBinding({.binding         = shaderio::BindingIndex::eSceneDesc,
                                       .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                       .descriptorCount = 1,
                                       .stageFlags      = VK_SHADER_STAGE_ALL});

    NVVK_CHECK(m_rtDescriptorBindings.createDescriptorSetLayout(m_app->getDevice(), VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
                                                                &m_rtDescriptorSetLayout));
    NVVK_DBG_NAME(m_rtDescriptorSetLayout);

    // Creating all shaders: raygen, miss, chit
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

    const VkShaderModuleCreateInfo moduleInfo{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = m_slangCompiler.getSpirvSize(),
        .pCode    = m_slangCompiler.getSpirv(),
    };

    stages[eRaygen].pNext     = &moduleInfo;
    stages[eRaygen].pName     = "rgenMain";
    stages[eRaygen].stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].pNext       = &moduleInfo;
    stages[eMiss].pName       = "rmissMain";
    stages[eMiss].stage       = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].pNext = &moduleInfo;
    stages[eClosestHit].pName = "rchitMain";
    stages[eClosestHit].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;


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
    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_ALL,
        .offset     = 0,
        .size       = sizeof(shaderio::RtGltfPushConstant),
    };

    // Create the pipeline layout
    NVVK_CHECK(nvvk::createPipelineLayout(device, &m_rtPipelineLayout, {m_rtDescriptorSetLayout}, {pushConstantRange}));
    NVVK_DBG_NAME(m_rtPipelineLayout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{
        .sType                        = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .stageCount                   = static_cast<uint32_t>(stages.size()),  // Stages are shader
        .pStages                      = stages.data(),
        .groupCount                   = static_cast<uint32_t>(shader_groups.size()),
        .pGroups                      = shader_groups.data(),
        .maxPipelineRayRecursionDepth = m_pushConst.maxDepth,  // Ray depth (maximal number of reflection)
        .layout                       = m_rtPipelineLayout,
    };
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR(device, {}, {}, 1, &ray_pipeline_info, nullptr, &m_rtPipeline));
    NVVK_DBG_NAME(m_rtPipeline);

    // Creating the SBT
    {
      // Shader Binding Table (SBT) setup
      nvvk::SBTGenerator sbtGenerator;
      sbtGenerator.init(m_app->getDevice(), m_rtProp);

      // Prepare SBT data from ray pipeline
      size_t bufferSize = sbtGenerator.calculateSBTBufferSize(m_rtPipeline, ray_pipeline_info);

      // Create SBT buffer using the size from above
      NVVK_CHECK(m_allocator.createBuffer(m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                          VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                          sbtGenerator.getBufferAlignment()));
      NVVK_DBG_NAME(m_sbtBuffer.buffer);

      // Pass the manual mapped pointer to fill the sbt data
      NVVK_CHECK(sbtGenerator.populateSBTBuffer(m_sbtBuffer.address, bufferSize, m_sbtBuffer.mapping));

      // Retrieve the regions, which are using addresses based on the m_sbtBuffer.address
      m_sbtRegions = sbtGenerator.getSBTRegions();

      sbtGenerator.deinit();
    }
  }

  void pushDescSet(VkCommandBuffer cmd) const
  {
    nvvk::WriteSetContainer writes{};
    writes.append(m_rtDescriptorBindings.getWriteSet(shaderio::BindingIndex::eTlas), m_tlas);
    writes.append(m_rtDescriptorBindings.getWriteSet(shaderio::BindingIndex::eOutImage), m_gBuffers.getColorImageView(),
                  VK_IMAGE_LAYOUT_GENERAL);
    writes.append(m_rtDescriptorBindings.getWriteSet(shaderio::BindingIndex::eSceneDesc), m_bSceneInfo);

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0, writes.size(), writes.data());
  }

  void onLastHeadlessFrame()
  {
    // Saving the rendered image
    std::filesystem::path outputFilename = nvutils::getExecutablePath().replace_extension(".jpg");
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(), outputFilename);

    // Saving the Full UI
    // - Create a temporary GBuffer
    // - Render the UI in the GBuffer
    // - Save the GBuffer to a file
    nvvk::GBuffer tempGBuffer;
    VkSampler     linearSampler;
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);
    tempGBuffer.init({
        .allocator      = &m_allocator,
        .colorFormats   = {VK_FORMAT_B8G8R8A8_UNORM},  // Only one GBuffer color attachment
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    });
    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      NVVK_CHECK(tempGBuffer.update(cmd, m_gBuffers.getSize()));
      m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    // Image to render to
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = tempGBuffer.getColorImageView();

    // Details of the dynamic rendering
    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;
    renderingInfo.renderArea           = {{0, 0}, tempGBuffer.getSize()};

    // Rendering the UI
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    {
      nvvk::cmdImageMemoryBarrier(cmd, {tempGBuffer.getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
      vkCmdBeginRendering(cmd, &renderingInfo);
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
      vkCmdEndRendering(cmd);
      nvvk::cmdImageMemoryBarrier(cmd, {tempGBuffer.getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
    }
    m_app->submitAndWaitTempCmdBuffer(cmd);
    vkDeviceWaitIdle(m_app->getDevice());

    // Saving image
    m_app->saveImageToFile(tempGBuffer.getColorImage(), tempGBuffer.getSize(),
                           nvutils::getExecutablePath().replace_extension(".screenshot.jpg").string(), 95);
    tempGBuffer.deinit();
  }

  void onFileDrop(const char* filename)
  {
    vkQueueWaitIdle(m_app->getQueue(0).queue);

    glm::mat4 randomTransfo =
        glm::translate(glm::mat4(2.0f), glm::vec3(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX,
                                                  static_cast<float>(rand()) / RAND_MAX));
    glm::vec3 randomColor = glm::vec3(static_cast<float>(rand()) / RAND_MAX, static_cast<float>(rand()) / RAND_MAX,
                                      static_cast<float>(rand()) / RAND_MAX);

    m_meshes.push_back(importGltfResources(loadGltfResources(filename)));
    m_instances.push_back({uint32_t(m_meshes.size() - 1), randomTransfo, randomColor});
    g_cameraManip->fit(m_meshes.back().bbox.min(), m_meshes.back().bbox.max());
    nvutils::Bbox& meshBB = m_meshes.back().bbox;
    g_cameraManip->setClipPlanes(
        {std::max(0.1f, meshBB.min().z * 0.1f), std::max(meshBB.radius() * 5.0f, g_cameraManip->getClipPlanes().y)});

    createResources();
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();
  }

  tinygltf::Model loadGltfResources(const std::filesystem::path& filename)
  {
    SCOPED_TIMER(__FUNCTION__);

    tinygltf::TinyGLTF tinyLoader;
    tinygltf::Model    model;
    std::string        err, warn;
    if(filename.extension() == ".gltf")
    {
      if(!tinyLoader.LoadASCIIFromFile(&model, &err, &warn, filename.string()))
      {
        LOGE("Error loading glTF file: %s\n", err.c_str());
        assert(0 && "No fallback");
        return {};
      }
    }
    else
    {
      if(!tinyLoader.LoadBinaryFromFile(&model, &err, &warn, filename.string()))
      {
        LOGE("Error loading glTF file: %s\n", err.c_str());
        assert(0 && "No fallback");
        return {};
      }
    }
    LOGI("Loaded glTF file: %s\n", filename.string().c_str());
    return model;
  }

  GltfMesh importGltfResources(const tinygltf::Model& model)
  {
    SCOPED_TIMER(__FUNCTION__);
    // Validate model structure
    assert((model.buffers.size() == 1) && "Supporting only one buffer");
    // Get first node with a mesh
    auto it = std::find_if(model.nodes.begin(), model.nodes.end(), [](const auto& n) { return n.mesh != -1; });
    assert((it != model.nodes.end()) && "Need a mesh");
    int meshID = it->mesh;  // Retrieve the mesh ID

    const tinygltf::Mesh&      tinyMesh  = model.meshes[meshID];
    const tinygltf::Primitive& primitive = tinyMesh.primitives.front();
    assert((tinyMesh.primitives.size() == 1 && primitive.mode == TINYGLTF_MODE_TRIANGLES) && "Must have one triangle primitive");
    assert((primitive.indices != -1 && primitive.attributes.contains("POSITION") && primitive.attributes.contains("NORMAL"))
           && "Missing required attributes");

    auto getElementByteSize = [](int type) -> uint32_t {
      return type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? 2U :
             type == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT   ? 4U :
             type == TINYGLTF_COMPONENT_TYPE_FLOAT          ? 4U :
                                                              0U;
    };

    GltfMesh gltfMesh;

    // Extract indices
    auto& accessor   = model.accessors[primitive.indices];
    auto& bufferView = model.bufferViews[accessor.bufferView];
    assert((accessor.count % 3 == 0) && "Should be a multiple of 3");
    gltfMesh.mesh.indices = {
        .offset = uint32_t(bufferView.byteOffset + accessor.byteOffset),
        .count  = uint32_t(accessor.count),
        .byteStride = uint32_t(bufferView.byteStride ? bufferView.byteStride : getElementByteSize(accessor.componentType)),
    };
    gltfMesh.indexType = accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;

    // Lambda for attributes
    auto extractAttribute = [&](const std::string& name, auto& attr) {
      auto& acc = model.accessors[primitive.attributes.at(name)];
      auto& bv  = model.bufferViews[acc.bufferView];
      assert((acc.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) && "Should be floats");
      attr = {
          .offset     = uint32_t(bv.byteOffset + acc.byteOffset),
          .count      = uint32_t(acc.count),
          .byteStride = uint32_t(bv.byteStride ? uint32_t(bv.byteStride) : 3U * getElementByteSize(acc.componentType)),
      };
    };

    extractAttribute("POSITION", gltfMesh.mesh.positions);
    extractAttribute("NORMAL", gltfMesh.mesh.normals);

    // Upload buffers
    assert(m_stagingUploader.isAppendedEmpty());
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    {
      NVVK_CHECK(m_allocator.createBuffer(gltfMesh.gltfBuffer, std::span<const unsigned char>(model.buffers[0].data).size_bytes(),
                                          VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT
                                              | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
      NVVK_CHECK(m_stagingUploader.appendBuffer(gltfMesh.gltfBuffer, 0, std::span<const unsigned char>(model.buffers[0].data)));
      NVVK_DBG_NAME(gltfMesh.gltfBuffer.buffer);

      NVVK_CHECK(m_allocator.createBuffer(gltfMesh.accessor, std::span<const shaderio::TriangleMesh>(&gltfMesh.mesh, 1).size_bytes(),
                                          VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
      NVVK_CHECK(m_stagingUploader.appendBuffer(gltfMesh.accessor, 0, std::span<const shaderio::TriangleMesh>(&gltfMesh.mesh, 1)));
      NVVK_DBG_NAME(gltfMesh.accessor.buffer);
    }
    m_stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Compute bounding box
    const std::vector<double>& minVal = model.accessors[primitive.attributes.at("POSITION")].minValues;
    const std::vector<double>& maxVal = model.accessors[primitive.attributes.at("POSITION")].maxValues;
    if(minVal.empty() && maxVal.empty())
    {
      const auto&    positions = gltfMesh.mesh.positions;
      const uint8_t* data      = model.buffers[0].data.data();
      for(uint32_t i = 0; i < positions.count; i++)
      {
        gltfMesh.bbox.insert(*reinterpret_cast<const glm::vec3*>(data + positions.offset + i * positions.byteStride));
      }
    }
    else
    {
      gltfMesh.bbox.insert(glm::make_vec3(minVal.data()));
      gltfMesh.bbox.insert(glm::make_vec3(maxVal.data()));
    }
    return gltfMesh;
  }


  //--------------------------------------------------------------------------------------------------
  nvapp::Application*      m_app{};
  nvvk::ResourceAllocator  m_allocator{};  // The VMA allocator
  nvvk::StagingUploader    m_stagingUploader{};
  nvvk::GBuffer            m_gBuffers{};              // The G-Buffer
  nvvk::SamplerPool        m_samplerPool{};           // The sampler pool, used to create a sampler for the texture
  nvvk::DescriptorBindings m_rtDescriptorBindings{};  // The descriptor binding helper
  nvslang::SlangCompiler   m_slangCompiler{};         // The compiler for the shaders

  RaytraceGltfSettings m_settings{};

  std::vector<GltfMesh>     m_meshes{};     // The glTF meshes
  std::vector<GltfInstance> m_instances{};  // The glTF instances
  nvvk::Buffer              m_bSceneInfo;   // The buffer holding the scene information
  nvvk::Buffer              m_bMeshInfo{};  // All our meshes (array of mesh accessors)
  nvvk::Buffer              m_bInstInfo{};  // All our instances (array of instance accessors)

  VkDescriptorSetLayout m_rtDescriptorSetLayout{};  // Descriptor set layout for the scene info (set 1)
  VkPipelineLayout      m_rtPipelineLayout{};       // The pipeline layout use with graphics pipeline
  VkPipeline            m_rtPipeline{};             // The pipeline


  std::vector<nvvk::AccelerationStructure> m_blas{};        // Bottom-level AS
  nvvk::AccelerationStructure              m_tlas{};        // Top-level AS
  nvvk::Buffer                             m_sbtBuffer{};   // The SBT buffer (holding the raygen, miss, chit)
  nvvk::SBTGenerator::Regions              m_sbtRegions{};  // The SBT regions (raygen, miss, chit)

  // Raytracing properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_accelStructProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  // Shader related
  nvshaders::Tonemapper        m_tonemapper{};
  shaderio::TonemapperData     m_tonemapperData{};
  shaderio::RtGltfSceneInfo    m_sceneInfo{};
  shaderio::RtGltfPushConstant m_pushConst{};
};


//////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvapp::Application           application;  // The application
  nvapp::ApplicationCreateInfo appInfo;      // Information to create the application
  nvvk::Context                vkContext;    // The Vulkan context
  nvvk::ContextInitInfo        vkSetup;      // Information to create the Vulkan context

  // Parse the command line to get the application information
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;

  reg.addVector({"size", "Size of the window to be created", "s"}, &appInfo.windowSize);
  reg.add({"headless"}, &appInfo.headless, true);
  reg.add({"frames", "Number of frames to run in headless mode"}, &appInfo.headlessFrameCount);
  reg.add({"vsync"}, &appInfo.vSync);
  reg.add({"verbose", "Verbose output of the Vulkan context"}, &vkSetup.verbose);
  reg.add({"validation", "Enable validation layers", "v"}, &vkSetup.enableValidationLayers);

  cli.add(reg);
  cli.parse(argc, argv);

  //--------------------------------------------------------------------------------------------------
  // Vulkan setup
  // clang-format off
    VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  // clang-format on
  vkSetup = {
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions =
          {
              {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
              {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
              {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
              {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},
              {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},
          },
      .queues = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_TRANSFER_BIT},
  };

  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Validation layers settings
  nvvk::ValidationSettings vvlInfo{};
  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();

  // Create the Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  //--------------------------------------------------------------------------------------------------
  // Application setup
  appInfo.name           = nvutils::getExecutablePath().stem().string();
  appInfo.instance       = vkContext.getInstance();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.device         = vkContext.getDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create application
  application.init(appInfo);

  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(g_cameraManip);


  // Elements
  application.addElement(elemCamera);  // Controlling the camera movement
  application.addElement(std::make_shared<Raytracing>());
  application.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>());
  application.addElement(std::make_shared<nvapp::ElementDefaultMenu>());

  // Run the application
  application.run();

  // Cleanup
  application.deinit();
  vkContext.deinit();
}
