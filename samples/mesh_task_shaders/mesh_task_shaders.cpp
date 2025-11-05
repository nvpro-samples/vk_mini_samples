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

This sample demonstrates task and mesh shaders by generating wireframe
bounding boxes. A slider controls the number of rows of boxes.

*/
//////////////////////////////////////////////////////////////////////////

#define USE_SLANG true           // Slang: true for Slang, false for GLSL
#define MULTI_ENTRY_POINTS true  // Slang: true for multi entry points, false for single entry point
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")


// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#include <array>
#include <glm/glm.hpp>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "shaders/shaderio.h"  // Shared between host and device

#include <fmt/format.h>

#define VMA_IMPLEMENTATION

#include "_autogen/mesh_task.slang.h"  // Pre-compiled shader

#include "_autogen/mesh_task.task.slang.h"
#include "_autogen/mesh_task.mesh.slang.h"
#include "_autogen/mesh_task.frag.slang.h"


#include "_autogen/mesh_task.task.glsl.h"
#include "_autogen/mesh_task.mesh.glsl.h"
#include "_autogen/mesh_task.frag.glsl.h"

#include <common/utils.hpp>
#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvgui/camera.hpp>
#include <nvslang/slang.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/buffer_suballocator.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>

// The camera for the scene
std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};
nvutils::ProfilerManager                    g_profilerManager;  // #PROFILER

//////////////////////////////////////////////////////////////////////////
/// Demonstrates mesh and task shaders with bounding boxes
class MeshShaderBoxes : public nvapp::IAppElement
{
public:
  MeshShaderBoxes()           = default;
  ~MeshShaderBoxes() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    initMeshShaderProperties(m_app->getPhysicalDevice());

    m_allocator = std::make_unique<nvvk::ResourceAllocator>();
    NVVK_CHECK(m_allocator->init(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    }));

    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());

    // Acquiring the sampler which will be used for displaying the GBuffer
    m_samplerPool.init(app->getDevice());
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    m_gBuffers = std::make_unique<nvvk::GBuffer>();
    m_gBuffers->init({
        .allocator      = m_allocator.get(),
        .colorFormats   = {m_colorFormat},
        .depthFormat    = m_depthFormat,
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    });

    // Setting up the Slang compiler
    m_slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
    m_slangCompiler.defaultTarget();
    m_slangCompiler.defaultOptions();
    m_slangCompiler.addOption({slang::CompilerOptionName::DebugInformation, {slang::CompilerOptionValueKind::Int, 1}});
    m_slangCompiler.addOption({slang::CompilerOptionName::Optimization, {slang::CompilerOptionValueKind::Int, 0}});

    m_profilerTimeline = g_profilerManager.createTimeline({.name = "Primary Timeline"});
    m_profilerGpuTimer.init(m_profilerTimeline, m_app->getDevice(), m_app->getPhysicalDevice(), m_app->getQueue(0).familyIndex, false);


    createFrameInfoBuffer();
    createStatisticsBuffer();
    createPipeline();

    // Setup camera
    g_cameraManip->setClipPlanes({0.1F, 10000.0F});
    g_cameraManip->setLookat({-29.239626, 18.738466, -34.69831}, {13.1, -16.1, 6.1}, {0, 1, 0});
  }

  // Query and validate mesh shader capabilities
  void initMeshShaderProperties(VkPhysicalDevice physicalDevice)
  {
    VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceVulkan11Features device11Features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceFeatures2        deviceFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    meshShaderFeatures.pNext = &device11Features;
    deviceFeatures.pNext     = &meshShaderFeatures;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &deviceFeatures);


    // Query mesh shader properties
    VkPhysicalDeviceProperties2 deviceProps2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    m_meshShaderProps.pNext = &m_device11Props;
    deviceProps2.pNext      = &m_meshShaderProps;
    vkGetPhysicalDeviceProperties2(physicalDevice, &deviceProps2);

    // Check if mesh shader is supported
    if(!meshShaderFeatures.meshShader || !meshShaderFeatures.taskShader)
    {
      LOGE("Mesh shader or task shader not supported\n");
      std::exit(0);
    }

    // Calculate optimal workgroup size based on hardware capabilities
    // This follows the pattern: min(128, min(maxPreferred, min(maxSize[0], maxInvocations)))
    uint32_t meshShaderWorkgroupSize =
        std::min(128u, std::min(m_meshShaderProps.maxPreferredMeshWorkGroupInvocations,
                                std::min(m_meshShaderProps.maxMeshWorkGroupSize[0], m_meshShaderProps.maxMeshWorkGroupInvocations)));

    // NOTE: The shader uses a default workgroup size of 32 (defined in shaderio.h) which is optimal for NVIDIA hardware.
    // To use a different size, it can be overridden via MESHSHADER_WORKGROUP_SIZE macro in CMakeLists.txt EXTRA_FLAGS.
    // The runtime Slang compiler path uses the calculated meshShaderWorkgroupSize above via macro definition.

    LOGI("Mesh Shader Properties:\n");
    LOGI("  Workgroup size: %u\n", meshShaderWorkgroupSize);
    LOGI("  Max output vertices: %u\n", m_meshShaderProps.maxMeshOutputVertices);
    LOGI("  Max output primitives: %u\n", m_meshShaderProps.maxMeshOutputPrimitives);
    LOGI("  Max task work group count: [%u, %u, %u]\n", m_meshShaderProps.maxTaskWorkGroupCount[0],
         m_meshShaderProps.maxTaskWorkGroupCount[1], m_meshShaderProps.maxTaskWorkGroupCount[2]);
    LOGI("  Max mesh work group count: [%u, %u, %u]\n", m_meshShaderProps.maxMeshWorkGroupCount[0],
         m_meshShaderProps.maxMeshWorkGroupCount[1], m_meshShaderProps.maxMeshWorkGroupCount[2]);
    LOGI("  Rendering %u boxes with %u vertices and %u primitives\n", shaderio::BOXES_PER_MESH,
         shaderio::BOXES_PER_MESH * shaderio::VERTICES_PER_BOX, shaderio::BOXES_PER_MESH * shaderio::LINES_PER_BOX);
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);

    m_allocator->destroyBuffer(m_frameInfo);
    m_allocator->destroyBuffer(m_readbackDevice);
    m_allocator->destroyBuffer(m_readbackHost);

    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    m_descriptorPack.deinit();

    m_profilerGpuTimer.deinit();
    g_profilerManager.destroyTimeline(m_profilerTimeline);

    m_samplerPool.deinit();
    m_gBuffers->deinit();
    m_allocator->deinit();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { m_gBuffers->update(cmd, size); }

  void onUIRender() override
  {
    if(!m_gBuffers)
      return;

    {  // Setting menu
      ImGui::Begin("Settings");
      nvgui::CameraWidget(g_cameraManip);

      ImGui::Separator();
      ImGui::Text("Mesh Shader Parameters");

      // Calculate safe maximums based on hardware limits
      // Each task workgroup tests BOXES_PER_TASK boxes (one per thread), so multiply by BOXES_PER_TASK
      int maxBoxesX = std::min(1000, static_cast<int>(m_meshShaderProps.maxTaskWorkGroupCount[0] * shaderio::BOXES_PER_TASK));
      int maxBoxesZ = std::min(1000, static_cast<int>(m_meshShaderProps.maxTaskWorkGroupCount[1]));

      ImGui::SliderInt("Boxes in X", &m_totalBoxesX, 1, maxBoxesX);
      ImGui::SliderInt("Boxes in Z", &m_totalBoxesZ, 1, maxBoxesZ);
      ImGui::SliderFloat("Box Size", &m_boxSize, 0.1f, 2.0f);
      ImGui::SliderFloat("Spacing", &m_spacing, 1.0f, 5.0f);

      ImGui::Separator();
      ImGui::Text("Animation");
      ImGui::Checkbox("Enable Ripple & Rotation", &m_animate);
      if(m_animate)
      {
        ImGui::SliderFloat("Animation Speed", &m_animSpeed, 0.0f, 3.0f);
        ImGui::TextWrapped("Boxes ripple from center and rotate - watch culling stats change!");
      }

      // Display stats
      uint32_t totalBoxes = m_totalBoxesX * m_totalBoxesZ;
      uint32_t workgroupsX = (m_totalBoxesX + shaderio::BOXES_PER_TASK - 1) / shaderio::BOXES_PER_TASK;  // ceil(totalBoxesX / BOXES_PER_TASK)
      uint32_t workgroupsZ     = m_totalBoxesZ;
      uint64_t totalWorkgroups = static_cast<uint64_t>(workgroupsX) * static_cast<uint64_t>(workgroupsZ);
      ImGui::Separator();
      ImGui::Text("Stats:");
      ImGui::Text("Total Boxes: %u x %u = %u", m_totalBoxesX, m_totalBoxesZ, totalBoxes);
      ImGui::Text("Task Workgroups: %u x %u = %llu (%u boxes/wg)", workgroupsX, workgroupsZ, totalWorkgroups, shaderio::BOXES_PER_TASK);
      ImGui::Text("Workgroup Limits: %u x %u", m_meshShaderProps.maxTaskWorkGroupCount[0],
                  m_meshShaderProps.maxTaskWorkGroupCount[1]);
      ImGui::Text("Max Total Workgroups: %u", m_meshShaderProps.maxTaskWorkGroupTotalCount);

      // Read back the statistics from host buffer to display boxes drawn
      shaderio::Statistics* stats = reinterpret_cast<shaderio::Statistics*>(m_readbackHost.mapping);
      if(stats)
      {
        ImGui::Separator();
        ImGui::Text("Boxes Drawn: %u (%.1f%%)", stats->boxesDrawn, totalBoxes > 0 ? (100.0f * stats->boxesDrawn / totalBoxes) : 0.0f);
      }

      // Warning if exceeding total workgroup limit
      if(totalWorkgroups > m_meshShaderProps.maxTaskWorkGroupTotalCount)
      {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "WARNING: Exceeds total workgroup limit!");
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Dispatch will be scaled down automatically.");
      }

      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(ImTextureID(m_gBuffers->getDescriptorSet()), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onUIMenu() override
  {
    bool reloadShader = false;
    if(ImGui::BeginMenu("Tools"))
    {
      reloadShader = ImGui::MenuItem("Reload Shaders");
      ImGui::EndMenu();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_F5) || reloadShader)
    {
      vkQueueWaitIdle(m_app->getQueue(0).queue);
      vkDestroyPipeline(m_device, m_pipeline, nullptr);
      vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
      m_descriptorPack.deinit();
      createPipeline();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);
    m_profilerTimeline->frameAdvance();

    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

    // Update animation time
    if(m_animate)
    {
      m_time += ImGui::GetIO().DeltaTime * m_animSpeed;
    }

    // Clear device statistics buffer at the start of each frame
    vkCmdFillBuffer(cmd, m_readbackDevice.buffer, 0, sizeof(shaderio::Statistics), 0);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT);

    // Update Frame buffer uniform buffer
    shaderio::FrameInfo finfo{};
    finfo.view   = g_cameraManip->getViewMatrix();
    finfo.proj   = g_cameraManip->getPerspectiveMatrix();
    finfo.camPos = g_cameraManip->getEye();

    // Calculate frustum planes from view-projection matrix
    calculateFrustumPlanes(finfo.view, finfo.proj, finfo.frustumPlanes);

    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT);

    // Rendering to the GBuffer
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers->getColorImageView();
    colorAttachment.clearValue                = {m_clearColor};
    colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView                 = m_gBuffers->getDepthImageView();
    depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    // Create the rendering info
    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea           = DEFAULT_VkRect2D(m_gBuffers->getSize());
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;
    renderingInfo.pDepthAttachment     = &depthAttachment;

    // Allow to render to the GBuffer
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    // Start the rendering
    vkCmdBeginRendering(cmd, &renderingInfo);

    m_graphicState.cmdSetViewportAndScissor(cmd, m_app->getViewportSize());

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_descriptorPack.getSetPtr(), 0, nullptr);

    // Push constants
    shaderio::PushConstant pushConst{};
    pushConst.totalBoxesX    = static_cast<uint32_t>(m_totalBoxesX);
    pushConst.totalBoxesZ    = static_cast<uint32_t>(m_totalBoxesZ);
    pushConst.boxSize        = m_boxSize;
    pushConst.spacing        = m_spacing;
    pushConst.statisticsAddr = VkDeviceAddress(m_readbackDevice.address);
    pushConst.time           = m_time;
    pushConst.animSpeed      = m_animSpeed;
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT, 0,
                       sizeof(shaderio::PushConstant), &pushConst);

    // Draw using mesh shaders - launch a 2D grid of task shader workgroups
    // Each task workgroup tests up to BOXES_PER_TASK boxes (1 per thread), so dispatch ceil(totalBoxesX/BOXES_PER_TASK) workgroups
    uint32_t workgroupsX = (m_totalBoxesX + shaderio::BOXES_PER_TASK - 1) / shaderio::BOXES_PER_TASK;  // ceil division
    uint32_t workgroupsZ = m_totalBoxesZ;

    // IMPORTANT: VK_EXT_mesh_shader has limits on dispatch grid dimensions (typically 0xFFFF = 65535)
    // Clamp to hardware-reported limits to prevent validation errors and potential crashes
    workgroupsX = std::min(workgroupsX, m_meshShaderProps.maxTaskWorkGroupCount[0]);
    workgroupsZ = std::min(workgroupsZ, m_meshShaderProps.maxTaskWorkGroupCount[1]);

    // Clamp total workgroup count to maxTaskWorkGroupTotalCount
    uint64_t totalWorkgroups = static_cast<uint64_t>(workgroupsX) * static_cast<uint64_t>(workgroupsZ);
    if(totalWorkgroups > m_meshShaderProps.maxTaskWorkGroupTotalCount)
    {
      // Scale down to fit within the limit, maintaining aspect ratio as much as possible
      double scale =
          std::sqrt(static_cast<double>(m_meshShaderProps.maxTaskWorkGroupTotalCount) / static_cast<double>(totalWorkgroups));
      workgroupsX = std::max(1u, static_cast<uint32_t>(workgroupsX * scale));
      workgroupsZ = std::max(1u, static_cast<uint32_t>(workgroupsZ * scale));
    }

    vkCmdDrawMeshTasksEXT(cmd, workgroupsX, workgroupsZ, 1);

    vkCmdEndRendering(cmd);

    // Ensure atomic writes to device statistics buffer are complete, then copy to host buffer
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

    VkBufferCopy copyRegion{.size = sizeof(shaderio::Statistics)};
    vkCmdCopyBuffer(cmd, m_readbackDevice.buffer, m_readbackHost.buffer, 1, &copyRegion);

    // Allow to display the GBuffer
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  }

private:
  // Extract frustum planes from view-projection matrix
  // Planes are in the form: ax + by + cz + d = 0 (stored as vec4(a,b,c,d))
  // Reference: https://www.gamedevs.org/uploads/fast-extraction-viewing-frustum-planes-from-world-view-projection-matrix.pdf
  void calculateFrustumPlanes(const glm::mat4& view, const glm::mat4& proj, glm::vec4 planes[6])
  {
    glm::mat4 vp = proj * view;

    // Left plane
    planes[0] = glm::vec4(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]);
    // Right plane
    planes[1] = glm::vec4(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]);
    // Bottom plane
    planes[2] = glm::vec4(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]);
    // Top plane
    planes[3] = glm::vec4(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]);
    // Near plane
    planes[4] = glm::vec4(vp[0][3] + vp[0][2], vp[1][3] + vp[1][2], vp[2][3] + vp[2][2], vp[3][3] + vp[3][2]);
    // Far plane
    planes[5] = glm::vec4(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]);

    // Normalize planes
    for(int i = 0; i < 6; i++)
    {
      float length = glm::length(glm::vec3(planes[i]));
      planes[i] /= length;
    }
  }

  void createPipeline()
  {
    // Descriptor setup
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);

    // Create the descriptor layout, pool, and 1 set
    NVVK_CHECK(m_descriptorPack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_descriptorPack.getLayout());
    NVVK_DBG_NAME(m_descriptorPack.getPool());
    NVVK_DBG_NAME(m_descriptorPack.getSet(0));

    // Writing to the descriptors
    nvvk::WriteSetContainer writes{};
    writes.append(m_descriptorPack.makeWrite(0), m_frameInfo);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkPipelineRenderingCreateInfo prendInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prendInfo.colorAttachmentCount    = 1;
    prendInfo.pColorAttachmentFormats = &m_colorFormat;
    prendInfo.depthAttachmentFormat   = m_depthFormat;

    // The push constant information
    const VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT,
                                                .offset = 0,
                                                .size   = sizeof(shaderio::PushConstant)};

    // Create PipelineLayout
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorPack.getLayout()}, {pushConstantRange}));
    NVVK_DBG_NAME(m_pipelineLayout);

    // Creating the Pipeline with mesh shaders
    m_graphicState.rasterizationState.cullMode    = VK_CULL_MODE_NONE;
    m_graphicState.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;  // Wireframe mode (using line primitives)
    m_graphicState.rasterizationState.lineWidth   = 2.0f;

    // Helper to create the graphic pipeline
    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;

    // Adding the shaders to the pipeline (mesh shaders instead of vertex)
#if USE_SLANG

#if MULTI_ENTRY_POINTS
    // Recompiling only for Slang in Multi-Entry mode
    m_slangCompiler.clearMacros();
    std::vector<std::pair<std::string, std::string>> macros = {
        {"TASKSHADER_WORKGROUP_SIZE", std::to_string(m_device11Props.subgroupSize)},
        {"MESHSHADER_WORKGROUP_SIZE", std::to_string(m_meshShaderProps.maxPreferredMeshWorkGroupInvocations)},
    };
    for(const auto& [k, v] : macros)
      m_slangCompiler.addMacro({k.c_str(), v.c_str()});
    if(m_slangCompiler.compileFile("mesh_task.slang"))
    {
      size_t          codeSize = m_slangCompiler.getSpirvSize();
      const uint32_t* code     = m_slangCompiler.getSpirv();
      creator.addShader(VK_SHADER_STAGE_TASK_BIT_EXT, "taskMain", codeSize, code);
      creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "meshMain", codeSize, code);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", codeSize, code);
    }
    else
    {
      creator.addShader(VK_SHADER_STAGE_TASK_BIT_EXT, "taskMain", mesh_task_slang);
      creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "meshMain", mesh_task_slang);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", mesh_task_slang);
    }
#else
    creator.addShader(VK_SHADER_STAGE_TASK_BIT_EXT, "taskMain", mesh_task_task_slang);
    creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "meshMain", mesh_task_mesh_slang);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", mesh_task_frag_slang);
#endif
#else
    creator.addShader(VK_SHADER_STAGE_TASK_BIT_EXT, "main", mesh_task_task_glsl);
    creator.addShader(VK_SHADER_STAGE_MESH_BIT_EXT, "main", mesh_task_mesh_glsl);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", mesh_task_frag_glsl);
#endif

    NVVK_CHECK(creator.createGraphicsPipeline(m_device, nullptr, m_graphicState, &m_pipeline));
    NVVK_DBG_NAME(m_pipeline);
  }

  void createFrameInfoBuffer()
  {
    NVVK_CHECK(m_allocator->createBuffer(m_frameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT,
                                         VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
    NVVK_DBG_NAME(m_frameInfo.buffer);
  }

  void createStatisticsBuffer()
  {
    // Create device-local buffer for shader writes
    NVVK_CHECK(m_allocator->createBuffer(m_readbackDevice, sizeof(shaderio::Statistics),
                                         VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE));
    NVVK_DBG_NAME(m_readbackDevice.buffer);

    // Create host-visible mapped buffer for CPU readback
    NVVK_CHECK(m_allocator->createBuffer(m_readbackHost, sizeof(shaderio::Statistics),
                                         VK_BUFFER_USAGE_2_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
    NVVK_DBG_NAME(m_readbackHost.buffer);
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers->getColorImage(), m_gBuffers->getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  nvapp::Application*                      m_app{nullptr};
  std::shared_ptr<nvvk::ResourceAllocator> m_allocator;

  VkFormat                       m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
  VkFormat                       m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;
  VkClearColorValue              m_clearColor  = {{0.2F, 0.2F, 0.3F, 1.0F}};
  VkDevice                       m_device      = VK_NULL_HANDLE;
  std::unique_ptr<nvvk::GBuffer> m_gBuffers{};
  nvvk::SamplerPool              m_samplerPool{};

  // Resources
  nvvk::Buffer m_frameInfo;
  nvvk::Buffer m_readbackDevice;  // Device-local buffer for shader writes
  nvvk::Buffer m_readbackHost;    // Host-visible mapped buffer for CPU readback

  // Settings
  int   m_totalBoxesX = 100;  // Total number of boxes in X dimension
  int   m_totalBoxesZ = 100;  // Total number of boxes in Z dimension
  float m_boxSize     = 0.5f;
  float m_spacing     = 1.5f;
  bool  m_animate     = true;  // Enable animation
  float m_animSpeed   = 1.0f;  // Animation speed multiplier
  float m_time        = 0.0f;  // Current animation time

  // Mesh shader properties and limits (queried from device)
  VkPhysicalDeviceMeshShaderPropertiesEXT m_meshShaderProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
  VkPhysicalDeviceVulkan11Properties m_device11Props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};

  // Pipeline
  nvvk::GraphicsPipelineState m_graphicState;
  VkPipeline                  m_pipeline{};
  VkPipelineLayout            m_pipelineLayout{};
  nvvk::DescriptorPack        m_descriptorPack{};

  // Compilers
  nvslang::SlangCompiler m_slangCompiler{};

  // Profiler
  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer;
};

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  // Mesh shader feature and properties structures
  VkPhysicalDeviceMeshShaderPropertiesEXT meshShaderProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
  VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
  meshShaderFeatures.pNext = &meshShaderProps;
  VkPhysicalDeviceFragmentShadingRateFeaturesKHR shadingRateFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR};

  nvvk::ContextInitInfo vkSetup;
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
  vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  // Enable mesh shader extension
  vkSetup.deviceExtensions.push_back({VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshShaderFeatures});
  vkSetup.deviceExtensions.push_back({VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &shadingRateFeatures});

  // Adding validation layers
  nvvk::ValidationSettings vvlInfo{};
  vvlInfo.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();

  // Create Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Failed to create Vulkan context\n");
    std::exit(0);
  }

  // Application setup
  appInfo.name           = fmt::format("{} ({})", nvutils::getExecutablePath().stem().string(), SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Setting up the layout of the application
  appInfo.dockSetup = [](ImGuiID viewportID) {
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.2F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(settingID, ImGuiDir_Down, 0.35F, nullptr, &settingID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
  };

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Camera manipulator (global)
  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(g_cameraManip);
  auto        profilerSettings = std::make_shared<nvapp::ElementProfiler::ViewSettings>();
  static auto elemProfiler     = std::make_shared<nvapp::ElementProfiler>(&g_profilerManager, profilerSettings);


  // Add all application elements
  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));
  app.addElement(std::make_shared<MeshShaderBoxes>());
  app.addElement(elemProfiler);

  app.run();
  app.deinit();

  vkContext.deinit();

  return 0;
}
