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

 This sample shows how to draw wireframe on top of solid geometry in a 
 single pass, using gl_BaryCoordEXT (same as gl_BaryCoordNV)

 https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_fragment_shader_barycentric.html

*/
//////////////////////////////////////////////////////////////////////////

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

// clang-format off
#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
#define VMA_IMPLEMENTATION
#define IMGUI_DEFINE_MATH_OPERATORS
// clang-format on

#include <imgui/imgui.h>

#include <array>

namespace shaderio {
using namespace glm;
#include "shaders/shaderio.h"
}  // namespace shaderio

// Shaders
#include "_autogen/bary.frag.glsl.h"
#include "_autogen/bary.slang.h"
#include "_autogen/bary.vert.glsl.h"

#include "nvvk/validation_settings.hpp"
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/property_editor.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/primitives.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>


// thickness;color;thicknessVar;smoothing;screenSpace;backFaceColor;enableDash;dashRepeats;dashLength;onlyWire;
std::vector<shaderio::WireframeSettings> presets{
    {1.0F, {0.8F, 0.F, 0.F}, {1.0F, 1.0F}, 1.0F, 1, {0.5F, 0.5F, 0.5F}, 0, 5, 0.5F, 0},       // default
    {1.0F, {0.F, 0.8F, 0.F}, {1.0F, 1.0F}, 0.5F, 1, {0.5F, 0.5F, 0.5F}, 0, 5, 0.5F, 1},       // Wire dot
    {0.1F, {0.9F, 0.9F, 0.F}, {0.0F, 1.0F}, 0.1F, 0, {0.5F, 0.5F, 0.5F}, 0, 5, 0.5F, 0},      // Star
    {0.1F, {0.7F, 0.0F, 0.01F}, {0.7F, 0.0F}, 0.1F, 0, {0.07F, 0.0F, 0.0F}, 1, 10, 0.8F, 1},  // Flake
    {0.3F, {1.F, 1.0F, 1.F}, {1.F, 1.F}, 2.F, 1, {0.06F, 0.06F, 0.06F}, 0, 8, 0.4F, 0},       // Thin
    {0.5F, {.8F, .8F, .8F}, {1.F, 1.F}, 1.F, 1, {0.1F, 0.1F, 0.1F}, 1, 1, 1.F, 1},            // Wire line
    {0.5F, {.8F, .8F, .8F}, {1.F, 1.F}, 1.F, 1, {0.1F, 0.1F, 0.1F}, 1, 20, 0.5F, 1},          // Stipple
};


//////////////////////////////////////////////////////////////////////////
/// </summary> Display an image on a quad.
class BaryWireframe : public nvapp::IAppElement
{
public:
  BaryWireframe()           = default;
  ~BaryWireframe() override = default;

  void setCameraManipulator(std::shared_ptr<nvutils::CameraManipulator> camera) { m_cameraManip = camera; }

  void onAttach(nvapp::Application* app) override
  {
    nvutils::ScopedTimer st(__FUNCTION__);

    m_app    = app;
    m_device = m_app->getDevice();

    // Allocator
    m_alloc.init(VmaAllocatorCreateInfo{
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });

    // Acquiring the sampler which will be used for displaying the GBuffer
    m_samplerPool.init(app->getDevice());
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // GBuffer
    m_depthFmt = nvvk::findDepthFormat(app->getPhysicalDevice());
    m_gBuffers.init({
        .allocator      = &m_alloc,
        .colorFormats   = {m_colorFmt},  // Only one GBuffer color attachment
        .depthFormat    = m_depthFmt,
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    });

    m_settings = presets[0];
    createScene();
    createVkBuffers();
    createPipeline();
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_device));
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& viewportSize) override
  {
    NVVK_CHECK(m_gBuffers.update(cmd, viewportSize));
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      nvgui::CameraWidget(m_cameraManip);

      // Objects
      const char* items[] = {"Sphere", "Cube", "Tetrahedron", "Octahedron", "Icosahedron", "Cone"};
      int         flag    = ImGuiSliderFlags_Logarithmic;
      float       maxT    = m_settings.screenSpace ? 10.0F : 0.3f;
      namespace PE        = nvgui::PropertyEditor;
      PE::begin();
      PE::Combo("Geometry", &m_currentObject, items, IM_ARRAYSIZE(items));
      PE::Checkbox("Screen Space", (bool*)&m_settings.screenSpace);
      PE::Checkbox("Only Wire", (bool*)&m_settings.onlyWire);
      PE::ColorEdit3("Color", &m_settings.color.x);
      PE::ColorEdit3("Back Color", &m_settings.backFaceColor.x);
      PE::SliderFloat("Thickness", &m_settings.thickness, 0.F, maxT, "%.3f", flag);
      PE::SliderFloat2("Edge Variation", &m_settings.thicknessVar.x, 0.F, 1.F);
      PE::SliderFloat("Smoothing", &m_settings.smoothing, 0.F, 2.F);
      PE::Checkbox("Stipple", (bool*)&m_settings.enableStipple);
      PE::SliderInt("Repeats", &m_settings.stippleRepeats, 0, 20);
      PE::SliderFloat("Length", &m_settings.stippleLength, 0.F, 1.F);
      PE::end();
      PE::begin();
      {
        static int  item_current = 0;
        const char* items[]      = {"Default", "Wireframe dot", "Star", "Flake", " Thin", "Wireframe line", "Stipple"};
        if(PE::entry("Preset", [&] { return ImGui::Combo("##1", &item_current, items, IM_ARRAYSIZE(items)); }))
          m_settings = presets[item_current];
      }
      PE::end();

      ImGui::End();  // "Settings"
    }

    {
      // Display the G-Buffer image
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);

    // Update Frame buffer uniform buffer
    shaderio::FrameInfo finfo{};
    finfo.view   = m_cameraManip->getViewMatrix();
    finfo.proj   = m_cameraManip->getPerspectiveMatrix();
    finfo.camPos = m_cameraManip->getEye();
    vkCmdUpdateBuffer(cmd, m_frameInfoBuf.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);

    // Update the sample settings
    vkCmdUpdateBuffer(cmd, m_settingsBuf.buffer, 0, sizeof(shaderio::WireframeSettings), &m_settings);

    // Making sure the information is transfered
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);


    // Drawing the primitives in a G-Buffer
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView();
    colorAttachment.clearValue                = {m_clearColor};
    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView                 = m_gBuffers.getDepthImageView();
    depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    // Create the rendering info
    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea           = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;
    renderingInfo.pDepthAttachment     = &depthAttachment;

    // Start the rendering
    vkCmdBeginRendering(cmd, &renderingInfo);
    // Set the viewport and scissor
    nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_gBuffers.getSize());
    // Bind the pipeline and descriptor set
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorPack.sets[0], 0, nullptr);
    const VkDeviceSize offsets{0};
    auto&              node = m_nodes[m_currentObject];
    {
      VulkanMeshBuffers& mesh = m_meshBuffers[node.mesh];
      // Push constant information
      m_pushConst.transfo    = node.localMatrix();
      m_pushConst.color      = m_materials[node.material].color;
      m_pushConst.clearColor = glm::make_vec4(m_clearColor.float32);
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(shaderio::PushConstant), &m_pushConst);

      vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      auto numIndices = static_cast<uint32_t>(m_meshes[node.mesh].triangles.size() * 3);
      vkCmdDrawIndexed(cmd, numIndices, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
  }

private:
  void createScene()
  {
    nvutils::ScopedTimer st(__FUNCTION__);
    // Meshes
    m_meshes.emplace_back(nvutils::createSphereUv());
    m_meshes.emplace_back(nvutils::createCube());
    m_meshes.emplace_back(nvutils::createTetrahedron());
    m_meshes.emplace_back(nvutils::createOctahedron());
    m_meshes.emplace_back(nvutils::createIcosahedron());
    m_meshes.emplace_back(nvutils::createConeMesh());
    const int numMeshes = static_cast<int>(m_meshes.size());

    // Materials (colorful)
    for(int i = 0; i < numMeshes; i++)
    {
      const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
      const glm::vec3 v    = static_cast<glm::vec3>(sin(freq) * 0.5F + 0.5F);
      m_materials.push_back({glm::vec4(v, 1)});
    }

    // Instances
    for(int i = 0; i < numMeshes; i++)
    {
      nvutils::Node& node = m_nodes.emplace_back();
      node.mesh           = i;
      node.material       = i;
    }

    m_cameraManip->setClipPlanes({0.1F, 100.0F});
    m_cameraManip->setLookat({0.0F, 1.0F, 2.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  }

  void createPipeline()
  {
    nvutils::ScopedTimer st(__FUNCTION__);

    nvvk::DescriptorBindings& bindings = m_descriptorPack.bindings;
    bindings.addBinding(BIND_FRAME_INFO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    bindings.addBinding(BIND_SETTINGS, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);

    NVVK_CHECK(m_descriptorPack.initFromBindings(m_device, 1));
    NVVK_DBG_NAME(m_descriptorPack.layout);
    NVVK_DBG_NAME(m_descriptorPack.pool);
    NVVK_DBG_NAME(m_descriptorPack.sets[0]);

    const VkPushConstantRange pushConstant = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                              sizeof(shaderio::PushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorPack.layout}, {pushConstant}));
    NVVK_DBG_NAME(m_pipelineLayout);

    // Writing to descriptors
    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(bindings.getWriteSet(BIND_FRAME_INFO, m_descriptorPack.sets[0]), m_frameInfoBuf);
    writeContainer.append(bindings.getWriteSet(BIND_SETTINGS, m_descriptorPack.sets[0]), m_settingsBuf);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

    // Creating the Pipeline
    nvvk::GraphicsPipelineState graphicState;  // State of the graphic pipeline
    graphicState.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    graphicState.vertexBindings              = {{.stride = sizeof(nvutils::PrimitiveVertex), .divisor = 1}};
    graphicState.vertexAttributes            = {
        {.location = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(nvutils::PrimitiveVertex, pos)},
        {.location = 1, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(nvutils::PrimitiveVertex, nrm)}};

    // Helper to create the graphic pipeline
    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFmt};
    creator.renderingState.depthAttachmentFormat = m_depthFmt;

#if USE_SLANG
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", std::span(bary_slang));
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", std::span(bary_slang));
#else
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", std::span(bary_vert_glsl));
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", std::span(bary_frag_glsl));
#endif

    NVVK_CHECK(creator.createGraphicsPipeline(m_device, nullptr, graphicState, &m_pipeline));
    NVVK_DBG_NAME(m_pipeline);
  }


  void createVkBuffers()
  {
    nvutils::ScopedTimer st(__FUNCTION__);
    VkCommandBuffer      cmd = m_app->createTempCmdBuffer();

    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);

    m_meshBuffers.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      VulkanMeshBuffers& m = m_meshBuffers[i];
      NVVK_CHECK(m_alloc.createBuffer(m.vertices, std::span(m_meshes[i].vertices).size_bytes(), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m.vertices, 0, std::span(m_meshes[i].vertices)));
      NVVK_DBG_NAME(m.vertices.buffer);

      NVVK_CHECK(m_alloc.createBuffer(m.indices, std::span(m_meshes[i].triangles).size_bytes(), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m.indices, 0, std::span(m_meshes[i].triangles)));
      NVVK_DBG_NAME(m.indices.buffer);
    }

    NVVK_CHECK(m_alloc.createBuffer(m_frameInfoBuf, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT));
    NVVK_DBG_NAME(m_frameInfoBuf.buffer);

    NVVK_CHECK(m_alloc.createBuffer(m_settingsBuf, sizeof(shaderio::WireframeSettings), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT));
    NVVK_DBG_NAME(m_settingsBuf.buffer);
    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();
  }


  void destroyResources()
  {
    for(VulkanMeshBuffers& mesh : m_meshBuffers)
    {
      m_alloc.destroyBuffer(mesh.vertices);
      m_alloc.destroyBuffer(mesh.indices);
    }
    m_alloc.destroyBuffer(m_frameInfoBuf);
    m_alloc.destroyBuffer(m_settingsBuf);

    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    m_descriptorPack.deinit();

    m_gBuffers.deinit();
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
  nvapp::Application*     m_app{};          // Application
  nvvk::ResourceAllocator m_alloc;          // Allocator
  nvvk::SamplerPool       m_samplerPool{};  // The sampler pool, used to create a sampler for the texture
  VkDevice                m_device{};       // Vulkan device

  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip;  // Camera manipulator

  // G-Buffers
  nvvk::GBuffer     m_gBuffers;                                 // G-Buffers: color + depth
  VkFormat          m_colorFmt   = VK_FORMAT_R8G8B8A8_UNORM;    // Color format of the image
  VkFormat          m_depthFmt   = VK_FORMAT_UNDEFINED;         // Depth format of the depth buffer
  VkClearColorValue m_clearColor = {{0.3F, 0.3F, 0.3F, 1.0F}};  // Clear color

  // Pipeline
  VkPipeline           m_pipeline{};        // Graphic pipeline to render
  VkPipelineLayout     m_pipelineLayout{};  // Pipeline layout
  nvvk::DescriptorPack m_descriptorPack{};  // Descriptor bindings, layout, pool, and set

  // Resources
  struct VulkanMeshBuffers
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<VulkanMeshBuffers> m_meshBuffers;
  nvvk::Buffer                   m_frameInfoBuf;
  nvvk::Buffer                   m_settingsBuf;

  // Data and setting
  struct Material
  {
    glm::vec4 color{1.F};
  };
  std::vector<nvutils::PrimitiveMesh> m_meshes;
  std::vector<nvutils::Node>          m_nodes;
  std::vector<Material>               m_materials;

  // Push constant
  shaderio::PushConstant m_pushConst{};  // Information sent to the shader
  int                    m_currentObject{};

  // Settings
  shaderio::WireframeSettings m_settings{};
};

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  // Command parser
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);  // Restore this line
  cli.parse(argc, argv);

  // Vulkan context and extension feature needed.
  VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, &baryFeature}},
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Add the validation layer
  nvvk::ValidationSettings validation;
  vkSetup.instanceCreateInfoExt = validation.buildPNextChain();

  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Camera manipulator (global)
  auto cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera    = std::make_shared<nvapp::ElementCamera>();
  auto baryWireframe = std::make_shared<BaryWireframe>();
  elemCamera->setCameraManipulator(cameraManip);
  baryWireframe->setCameraManipulator(cameraManip);

  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));
  app.addElement(baryWireframe);

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
