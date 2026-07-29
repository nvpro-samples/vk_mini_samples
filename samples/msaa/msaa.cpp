/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


//////////////////////////////////////////////////////////////////////////
/*

 This sample demonstrates multisample anti-aliasing (MSAA) using nvvk::RenderTarget.
 The render target owns the multisampled color+depth images and a single-sample
 color resolve target; the scene is rendered multisampled and resolved to the
 single-sample image (displayed via ImGui). The sample count can be changed at
 runtime, which reconfigures and recreates the render target.

*/
//////////////////////////////////////////////////////////////////////////

#define USE_SLANG true
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION

#include <array>

#include <GLFW/glfw3.h>
#include <fmt/format.h>
#include <glm/glm.hpp>
#include <volk.h>
#undef APIENTRY

#include "shaders/shaderio_msaa.h"  // Shared between host and device

#include "common/utils.hpp"

#include <nvapp/application.hpp>           // The Application base
#include <nvapp/elem_camera.hpp>           // To handle the camera movement
#include <nvapp/elem_default_menu.hpp>     // Display a menu
#include <nvapp/elem_default_title.hpp>    // Change the window title
#include <nvgui/camera.hpp>                // Camera widget
#include <nvapp/imgui_texture.hpp>         // nvapp::ImTexture for ImGui display
#include <nvgui/property_editor.hpp>       // Formatting UI
#include <nvslang/slang.hpp>               // Slang compiler
#include <nvutils/camera_manipulator.hpp>  // To manipulate the camera
#include <nvutils/file_operations.hpp>     // Various
#include <nvutils/parameter_parser.hpp>    // To parse the command line
#include <nvutils/primitives.hpp>          // Create a cube
#include <nvutils/timers.hpp>              // Timing
#include <nvvk/check_error.hpp>            // Vulkan error checking
#include <nvvk/context.hpp>                // Vulkan context creation
#include <nvvk/debug_util.hpp>             // Debug names and more
#include <nvvk/descriptors.hpp>            // Help creation descriptor sets
#include <nvvk/formats.hpp>
#include <nvvk/graphics_pipeline.hpp>   // Helper to create a graphic pipeline
#include <nvvk/helpers.hpp>             // Find format
#include <nvvk/render_target.hpp>       // Offscreen render target with built-in MSAA + resolve
#include <nvvk/resource_allocator.hpp>  // The GPU resource allocator
#include <nvvk/staging.hpp>             // Staging buffer for upload


std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};

// Camera manipulation


//////////////////////////////////////////////////////////////////////////
// This class is the main application class that handles the creation of the scene,
// the rendering of the scene, and the UI menu.
class Msaa : public nvapp::IAppElement
{
public:
  Msaa() { createScene(); }
  ~Msaa() override = default;

  // This function is called when the application is attached to the window.
  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    // Allocator
    m_alloc.init({
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });

    // Slang compiler
    {
      m_slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
      m_slangCompiler.defaultTarget();
      m_slangCompiler.defaultOptions();
      m_slangCompiler.addOption({slang::CompilerOptionName::DebugInformation,
                                 {slang::CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_MAXIMAL}});
      m_slangCompiler.addOption({slang::CompilerOptionName::Optimization,
                                 {slang::CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_NONE}});
    }

    // Offscreen render target with built-in MSAA: the scene is rendered into a
    // multisampled color+depth image and the color is resolved to a single-sample
    // image for display. RenderTarget creates and resolves those images itself,
    // so the sample no longer manages any MSAA image/view by hand.
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    NVVK_CHECK(m_renderTarget.init({.alloc        = &m_alloc,
                                    .colorFormats = {m_colorFormat},
                                    .depthFormat  = m_depthFormat,
                                    .msaa = {.samples = m_msaaSamples, .colorResolve = VK_RESOLVE_MODE_AVERAGE_BIT},
                                    .debugName = "MSAA"}));


    createSceneBuffers();
    createPipeline();
    compileShaders();
  }

  // This function is called when the application is detached from the window.
  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);

    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    m_dsetBind.clear();

    for(PrimitiveMeshVk& m : m_meshVk)
    {
      m_alloc.destroyBuffer(m.vertices);
      m_alloc.destroyBuffer(m.indices);
    }
    m_alloc.destroyBuffer(m_frameInfo);

    m_viewportImage.deinit();
    m_renderTarget.deinit();

    vkDestroyShaderEXT(m_device, m_vertShader, nullptr);
    vkDestroyShaderEXT(m_device, m_fragShader, nullptr);

    m_alloc.deinit();
  }

  // This function is called when the UI menu is rendered.
  void onUIMenu() override
  {
    static bool close_app{false};
    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
      {
        close_app = true;
      }
      ImGui::EndMenu();
    }
    bool doCompileShaders = false;
    if(ImGui::BeginMenu("Tools"))
    {
      doCompileShaders = ImGui::MenuItem("Compile Shaders", "F5");
      ImGui::EndMenu();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_F5) || doCompileShaders)
    {
      compileShaders();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      close_app = true;
    }
    if(close_app)
    {
      m_app->close();
    }
  }

  // This function is called when the application is resized.
  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    NVVK_CHECK(m_renderTarget.update(cmd, size));
    // Views are recreated on resize: refresh the ImGui texture that displays the
    // resolved image in the viewport.
    m_viewportImage.update(m_renderTarget.getUiImageView());
  }

  // This function is called when the UI is rendered.
  void onUIRender() override
  {

    {  // Setting menu
      ImGui::Begin("Settings");
      nvgui::CameraWidget(g_cameraManip);

      // #MSAA
      ImGui::Separator();
      ImGui::Text("MSAA Settings");
      VkImageFormatProperties image_format_properties;
      vkGetPhysicalDeviceImageFormatProperties(m_app->getPhysicalDevice(), m_renderTarget.getColorFormat(),
                                               VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0, &image_format_properties);
      // sampleCounts is 3, 7 or 15, following line find n, in 2^n == sampleCounts+1
      const int max_sample_items = static_cast<int>(log2(static_cast<float>(image_format_properties.sampleCounts)) + 1.0F);
      // Same for the current VkSampleCountFlag, which is a power of two
      int                        item_combo = static_cast<int>(log2(static_cast<float>(m_msaaSamples)));
      std::array<const char*, 7> items      = {"1", "2", "4", "8", "16", "32", "64"};
      ImGui::Text("Sample Count");
      ImGui::SameLine();
      if(ImGui::Combo("##Sample Count", &item_combo, items.data(), max_sample_items))
      {
        auto samples  = static_cast<int32_t>(powf(2, static_cast<float>(item_combo)));
        m_msaaSamples = static_cast<VkSampleCountFlagBits>(samples);

        // Reconfigure the render target for the new sample count and recreate its
        // GPU resources. MSAA requires a resolve mode; a 1x sample count uses none.
        vkDeviceWaitIdle(m_device);  // Flushing the graphic pipeline
        const VkResolveModeFlagBits resolveMode =
            m_msaaSamples == VK_SAMPLE_COUNT_1_BIT ? VK_RESOLVE_MODE_NONE : VK_RESOLVE_MODE_AVERAGE_BIT;
        NVVK_CHECK(m_renderTarget.setSampleCount(m_msaaSamples, resolveMode));
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();
        NVVK_CHECK(m_renderTarget.update(cmd, m_app->getViewportSize()));
        m_app->submitAndWaitTempCmdBuffer(cmd);
        m_viewportImage.update(m_renderTarget.getUiImageView());
      }

      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the resolved image
      ImGui::Image(m_viewportImage, ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }

    {  // Window Title
      static float dirty_timer = 0.0F;
      dirty_timer += ImGui::GetIO().DeltaTime;
      if(dirty_timer > 1.0F)  // Refresh every seconds
      {
        std::string title = fmt::format("{} {}x{} | {} FPS / {:.3f}ms", nvutils::getExecutablePath().stem().string(),
                                        static_cast<int>(m_renderTarget.getSize().width),
                                        static_cast<int>(m_renderTarget.getSize().height),
                                        static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);

        glfwSetWindowTitle(m_app->getWindowHandle(), title.c_str());
      }
    }
  }

  // This function is called when the application is rendering a frame.
  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    // Update Frame buffer uniform buffer
    shaderio::FrameInfo frameInfo{};
    frameInfo.view   = g_cameraManip->getViewMatrix();
    frameInfo.proj   = g_cameraManip->getPerspectiveMatrix();
    frameInfo.camPos = g_cameraManip->getEye();
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &frameInfo);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

    // #MSAA - RenderTarget owns the multisampled color+depth images and the
    // single-sample resolve target. fillState() wires the resolve automatically
    // when the sample count is > 1, so there is no manual MSAA bookkeeping here.
    nvvk::RenderTargetState rtState;
    m_renderTarget.fillState(rtState);
    rtState.colorAttachments[0].clearValue = {.color = m_clearColor};
    rtState.depthAttachment.clearValue     = {.depthStencil = {1.0F, 0}};
    nvvk::RenderTargetState::AttachmentOps ops{};  // default: clear+store on color & depth, don't care on stencil
    rtState.cmdBeginRendering(cmd, ops);

    // Set the dynamic graphics pipeline state (rasterization samples must match the target)
    m_gfxState.multisampleState.rasterizationSamples = m_renderTarget.getSampleCount();  // #MSAA
    m_gfxState.cmdApplyAllStates(cmd);
    m_gfxState.cmdSetViewportAndScissor(cmd, m_app->getViewportSize());
    m_gfxState.cmdBindShaders(cmd, {m_vertShader, m_fragShader});

    // Push the descriptor set (Buffer) for the frame information
    nvvk::WriteSetContainer writes;
    writes.append(m_dsetBind.getWriteSet(0), m_frameInfo);
    VkPushDescriptorSetInfo pushDescInfo{.sType      = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_INFO,
                                         .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                         .layout     = m_pipelineLayout,
                                         .descriptorWriteCount = writes.size(),
                                         .pDescriptorWrites    = writes.data()};
    vkCmdPushDescriptorSet2(cmd, &pushDescInfo);

    renderScene(cmd);
    vkCmdEndRendering(cmd);
  }

private:
  void createScene()
  {
    // Meshes
    m_meshes.emplace_back(nvutils::createConeMesh(0.05F));
    const int numInstances = 50;

    // Instances
    for(int i = 0; i < numInstances; i++)
    {
      nvutils::Node& n = m_nodes.emplace_back();
      n.mesh           = 0;
      n.material       = i;

      glm::mat4 mrot, mtrans;
      mrot = glm::rotate(glm::mat4(1), static_cast<float>(i) / static_cast<float>(numInstances) * glm::two_pi<float>(),
                         {0.0F, 0.0F, 1.0F});
      mtrans   = glm::translate(glm::mat4(1), {0, -0.5, 0});
      n.matrix = mrot * mtrans;
    }

    // Materials (colorful)
    for(int i = 0; i < numInstances; i++)
    {
      const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
      const glm::vec3 v    = static_cast<glm::vec3>(glm::sin(freq) * 0.5F + 0.5F);
      m_materials.push_back({glm::vec4(v, 1.0F)});
    }

    g_cameraManip->setClipPlanes({0.1F, 100.0F});
    g_cameraManip->setLookat({0.0F, 0.0F, 1.7F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  }

  void renderScene(VkCommandBuffer cmd)
  {
    NVVK_DBG_SCOPE(cmd);


    // Render all the instances
    const VkDeviceSize offsets{0};
    for(const nvutils::Node& node : m_nodes)
    {
      const PrimitiveMeshVk& m           = m_meshVk[node.mesh];
      auto                   num_indices = static_cast<uint32_t>(m_meshes[node.mesh].triangles.size() * 3);

      // Push constant information
      m_pushConst.transfo = node.localMatrix();
      m_pushConst.color   = m_materials[node.material].color;
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(shaderio::PushConstant), &m_pushConst);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m.vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cmd, num_indices, 1, 0, 0, 0);
    }
  }


  VkPushConstantRange getPushConstantRanges()
  {
    return {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(shaderio::PushConstant)};
  }

  void createPipeline()
  {
    std::vector<VkDescriptorPoolSize> poolSize;
    m_dsetBind.addBinding(shaderio::MsaaBinding::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                          VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    NVVK_CHECK(m_dsetBind.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT, &m_descriptorSetLayout));
    NVVK_DBG_NAME(m_descriptorSetLayout);

    const VkPushConstantRange pushConstantRanges = getPushConstantRanges();
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorSetLayout}, {pushConstantRanges}));
    NVVK_DBG_NAME(m_pipelineLayout);

    // Adjusting the vertex attributes, and bindings for the dynamic pipeline
    m_gfxState.vertexBindings   = {{.sType   = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                                    .stride  = sizeof(nvutils::PrimitiveVertex),
                                    .divisor = 1}};
    m_gfxState.vertexAttributes = {{.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                    .location = 0,
                                    .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                    .offset   = offsetof(nvutils::PrimitiveVertex, pos)},
                                   {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                    .location = 1,
                                    .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                    .offset   = offsetof(nvutils::PrimitiveVertex, nrm)}};
  }

  // This function is called to compile the shaders.
  void compileShaders()
  {
    nvutils::ScopedTimer timer("Compile Shaders");

    if(m_slangCompiler.compileFile("raster_msaa.slang"))
    {
      vkDestroyShaderEXT(m_device, m_vertShader, nullptr);
      vkDestroyShaderEXT(m_device, m_fragShader, nullptr);
      VkPushConstantRange   pushConstantRanges = getPushConstantRanges();
      VkShaderCreateInfoEXT shaderCreateInfo{
          .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
          .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
          .codeSize               = m_slangCompiler.getSpirvSize(),
          .pCode                  = m_slangCompiler.getSpirv(),
          .setLayoutCount         = 1,
          .pSetLayouts            = &m_descriptorSetLayout,
          .pushConstantRangeCount = 1,
          .pPushConstantRanges    = &pushConstantRanges,
      };
      shaderCreateInfo.pName     = "vertexMain";
      shaderCreateInfo.stage     = VK_SHADER_STAGE_VERTEX_BIT;
      shaderCreateInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
      NVVK_CHECK(vkCreateShadersEXT(m_device, 1U, &shaderCreateInfo, nullptr, &m_vertShader));
      NVVK_DBG_NAME(m_vertShader);
      shaderCreateInfo.pName     = "fragmentMain";
      shaderCreateInfo.stage     = VK_SHADER_STAGE_FRAGMENT_BIT;
      shaderCreateInfo.nextStage = 0;
      NVVK_CHECK(vkCreateShadersEXT(m_device, 1U, &shaderCreateInfo, nullptr, &m_fragShader));
      NVVK_DBG_NAME(m_fragShader);
    }
  }

  void createSceneBuffers()
  {
    nvvk::StagingUploader stagingUploader;
    stagingUploader.init(&m_alloc);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_meshVk.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& mesh = m_meshVk[i];
      NVVK_CHECK(m_alloc.createBuffer(mesh.vertices, std::span(m_meshes[i].vertices).size_bytes(), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
      NVVK_CHECK(m_alloc.createBuffer(mesh.indices, std::span(m_meshes[i].triangles).size_bytes(), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
      NVVK_DBG_NAME(mesh.vertices.buffer);
      NVVK_DBG_NAME(mesh.indices.buffer);

      NVVK_CHECK(stagingUploader.appendBuffer(mesh.vertices, 0, std::span(m_meshes[i].vertices)));
      NVVK_CHECK(stagingUploader.appendBuffer(mesh.indices, 0, std::span(m_meshes[i].triangles)));
    }
    stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    stagingUploader.deinit();

    NVVK_CHECK(m_alloc.createBuffer(m_frameInfo, sizeof(shaderio::FrameInfo),
                                    VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
    NVVK_DBG_NAME(m_frameInfo.buffer);
  }

  void onLastHeadlessFrame() override
  {
    // Save the displayed image (resolved single-sample when MSAA is active).
    m_app->saveImageToFile(m_renderTarget.getSampleImage(), m_renderTarget.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*     m_app{nullptr};   // The application instance
  nvvk::ResourceAllocator m_alloc;          // The resource allocator
  nvvk::RenderTarget      m_renderTarget;   // Offscreen MSAA target (color+depth) with color resolve
  nvapp::ImTexture        m_viewportImage;  // ImGui texture displaying the resolved image

  // #MSAA
  VkSampleCountFlagBits m_msaaSamples{VK_SAMPLE_COUNT_4_BIT};  // The MSAA samples

  VkFormat          m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat          m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue m_clearColor  = {{0.4F, 0.4F, 0.6F, 1.F}};      // Clear color
  VkDevice          m_device      = VK_NULL_HANDLE;                 // Convenient


  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk> m_meshVk;
  nvvk::Buffer                 m_frameInfo;

  // Data and setting
  struct Material
  {
    glm::vec4 color{1.F};
  };
  std::vector<nvutils::PrimitiveMesh> m_meshes;
  std::vector<nvutils::Node>          m_nodes;
  std::vector<Material>               m_materials;

  // Pipeline
  nvvk::GraphicsPipelineState m_gfxState;               // The graphics pipeline state
  nvvk::DescriptorBindings    m_dsetBind;               // The descriptor bindings
  VkDescriptorSetLayout       m_descriptorSetLayout{};  // The descriptor set layout
  VkPipelineLayout            m_pipelineLayout{};       // The pipeline layout


  // Shaders
  nvslang::SlangCompiler m_slangCompiler;  // The slang compiler
  VkShaderEXT            m_vertShader{};   // The vertex shader
  VkShaderEXT            m_fragShader{};   // The fragment shader


  shaderio::PushConstant m_pushConst{};  // Information sent to the shader
};

//////////////////////////////////////////////////////////////////////////
///
///
int main(int argc, char** argv)
{
  nvapp::Application           app;
  nvapp::ApplicationCreateInfo appInfo;

  std::string                projectName = nvutils::getExecutablePath().stem().string();
  nvutils::ParameterParser   cli(projectName);
  nvutils::ParameterRegistry reg;
  bool                       verbose = false;
  reg.add({"verbose", "Verbose output of the Vulkan context"}, &verbose);
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  // Setup the Vulkan instance
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  nvvk::ContextInitInfo vkSetup = {
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures}},
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Vulkan context creation
  nvvk::Context vkContext;
  vkSetup.verbose |= verbose;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Application setup
  appInfo.name           = fmt::format("{} ({})", projectName, "Slang");
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  app.init(appInfo);

  // Add all application elements
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  elemCamera->setCameraManipulator(g_cameraManip);

  app.addElement(elemCamera);
  app.addElement(std::make_shared<Msaa>());

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
