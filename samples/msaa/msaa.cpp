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

 This sample renders in a Multi-sampled image and resolved it in the
 common G-Buffer

*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <vulkan/vulkan_core.h>
#include <imgui.h>

#include "common/vk_context.hpp"
#include "imgui/imgui_camera_widget.h"
#include "nvh/primitives.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvk/extensions_vk.hpp"

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#if USE_HLSL
#include "_autogen/raster_vertexMain.spirv.h"
#include "_autogen/raster_fragmentMain.spirv.h"
const auto& vert_shd = std::vector<uint8_t>{std::begin(raster_vertexMain), std::end(raster_vertexMain)};
const auto& frag_shd = std::vector<uint8_t>{std::begin(raster_fragmentMain), std::end(raster_fragmentMain)};
#elif USE_SLANG
#include "_autogen/raster_slang.h"
#else
#include "_autogen/raster.frag.glsl.h"
#include "_autogen/raster.vert.glsl.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert_glsl), std::end(raster_vert_glsl)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag_glsl), std::end(raster_frag_glsl)};
#endif  // USE_HLSL

#include <GLFW/glfw3.h>
#include "common/utils.hpp"


//////////////////////////////////////////////////////////////////////////
/// </summary> Display an image on a quad.
class Msaa : public nvvkhl::IAppElement
{
public:
  Msaa() { createScene(); }
  ~Msaa() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil       = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
    m_alloc       = std::make_unique<nvvk::ResourceAllocatorDma>(m_device, app->getPhysicalDevice());
    m_dset        = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());


    createVkBuffers();
    createPipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

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
    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      close_app = true;
    }
    if(close_app)
    {
      m_app->close();
    }
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    createGbuffers({width, height});
    createMsaaBuffers({width, height});  // #MSAA
  }

  void onUIRender() override
  {
    if(!m_gBuffers)
      return;

    {  // Setting menu
      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();

      // #MSAA
      ImGui::Separator();
      ImGui::Text("MSAA Settings");
      VkImageFormatProperties image_format_properties;
      vkGetPhysicalDeviceImageFormatProperties(m_app->getPhysicalDevice(), m_gBuffers->getColorFormat(),
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

        vkDeviceWaitIdle(m_device);  // Flushing the graphic pipeline

        // The graphic pipeline contains MSAA information
        vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        m_dset->deinit();
        m_dset->init(m_device);
        createMsaaBuffers(m_viewSize);
        createPipeline();
      }

      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }

    {  // Window Title
      static float dirty_timer = 0.0F;
      dirty_timer += ImGui::GetIO().DeltaTime;
      if(dirty_timer > 1.0F)  // Refresh every seconds
      {
        std::array<char, 256> buf{};

        const int ret = snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms", PROJECT_NAME,
                                 static_cast<int>(m_viewSize.x), static_cast<int>(m_viewSize.y),
                                 static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
        assert(ret > 0);
        glfwSetWindowTitle(m_app->getWindowHandle(), buf.data());
        dirty_timer = 0;
      }
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    if(!m_gBuffers)
      return;

    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    const float view_aspect_ratio = m_viewSize.x / m_viewSize.y;
    glm::vec3   eye;
    glm::vec3   center;
    glm::vec3   up;
    CameraManip.getLookat(eye, center, up);

    // Update Frame buffer uniform buffer
    DH::FrameInfo    finfo{};
    const glm::vec2& clip = CameraManip.getClipPlanes();
    finfo.view            = CameraManip.getMatrix();
    finfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);
    finfo.proj[1][1] *= -1;
    finfo.camPos = eye;
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);
    nvvk::memoryBarrier(cmd);


    // #MSAA
    if((m_msaaSamples & VK_SAMPLE_COUNT_1_BIT) == VK_SAMPLE_COUNT_1_BIT)
    {  // Not using MSAA
      nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView()},
                                       m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                       VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor, {1.0F, 0});
      r_info.pStencilAttachment = nullptr;
      vkCmdBeginRendering(cmd, &r_info);
    }
    else
    {  // Using MSAA image and resolving to the G-Buffer
      nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_msaaColorIView}, m_msaaDepthIView,
                                       VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor, {1.0F, 0});
      r_info.colorAttachments[0].resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
      r_info.colorAttachments[0].resolveImageView   = m_gBuffers->getColorImageView();  // Resolving MSAA to offscreen
      r_info.colorAttachments[0].resolveMode        = VK_RESOLVE_MODE_AVERAGE_BIT;
      r_info.pStencilAttachment                     = nullptr;
      vkCmdBeginRendering(cmd, &r_info);
    }

    renderScene(cmd);

    vkCmdEndRendering(cmd);
    // Make sure it is finished
    const VkImageMemoryBarrier image_memory_barrier =
        nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(), VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &image_memory_barrier);
  }

private:
  void createScene()
  {
    // Meshes
    m_meshes.emplace_back(nvh::createConeMesh(0.05F));
    const int num_instances = 50;

    // Instances
    for(int i = 0; i < num_instances; i++)
    {
      nvh::Node& n = m_nodes.emplace_back();
      n.mesh       = 0;
      n.material   = i;

      glm::mat4 mrot, mtrans;
      mrot = glm::rotate(glm::mat4(1), static_cast<float>(i) / static_cast<float>(num_instances) * glm::two_pi<float>(),
                         {0.0F, 0.0F, 1.0F});
      mtrans   = glm::translate(glm::mat4(1), {0, -0.5, 0});
      n.matrix = mrot * mtrans;
    }

    // Materials (colorful)
    for(int i = 0; i < num_instances; i++)
    {
      const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
      const glm::vec3 v    = static_cast<glm::vec3>(glm::sin(freq) * 0.5F + 0.5F);
      m_materials.push_back({glm::vec4(v, 1.0F)});
    }


    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({0.0F, 0.0F, 1.7F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  }

  void renderScene(VkCommandBuffer cmd)
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    m_app->setViewport(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_dset->getSets(m_frame), 0, nullptr);
    const VkDeviceSize offsets{0};
    for(const nvh::Node& n : m_nodes)
    {
      const PrimitiveMeshVk& m           = m_meshVk[n.mesh];
      auto                   num_indices = static_cast<uint32_t>(m_meshes[n.mesh].triangles.size() * 3);

      // Push constant information
      m_pushConst.transfo = n.localMatrix();
      m_pushConst.color   = m_materials[n.material].color;
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(DH::PushConstant), &m_pushConst);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m.vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cmd, num_indices, 1, 0, 0, 0);
    }
  }

  void createPipeline()
  {
    m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    m_dset->initLayout();
    m_dset->initPool(2);  // two frames - allow to change on the fly

    // Writing to descriptors
    const VkDescriptorBufferInfo      dbi_unif{m_frameInfo.buffer, 0, VK_WHOLE_SIZE};
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_dset->makeWrite(0, 0, &dbi_unif));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    const VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                      sizeof(DH::PushConstant)};

    VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    create_info.pushConstantRangeCount = 1;
    create_info.pPushConstantRanges    = &push_constant_ranges;
    create_info.setLayoutCount         = 1;
    create_info.pSetLayouts            = &m_dset->getLayout();
    vkCreatePipelineLayout(m_device, &create_info, nullptr, &m_pipelineLayout);

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    // Creating the Pipeline
    nvvk::GraphicsPipelineState pstate;
    pstate.multisampleState.rasterizationSamples = m_msaaSamples;  // #MSAA
    pstate.addBindingDescriptions({{0, sizeof(nvh::PrimitiveVertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, p))},  // Position
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, n))},  // Normal
    });
    nvvk::GraphicsPipelineGenerator pgen(m_device, m_pipelineLayout, prend_info, pstate);
#if USE_SLANG
    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &rasterSlang[0], sizeof(rasterSlang));
    pgen.addShader(shaderModule, VK_SHADER_STAGE_VERTEX_BIT, "vertexMain");
    pgen.addShader(shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain");
#else
    pgen.addShader(vert_shd, VK_SHADER_STAGE_VERTEX_BIT, USE_HLSL ? "vertexMain" : "main");
    pgen.addShader(frag_shd, VK_SHADER_STAGE_FRAGMENT_BIT, USE_HLSL ? "fragmentMain" : "main");
#endif

    m_graphicsPipeline = pgen.createPipeline();
    m_dutil->setObjectName(m_graphicsPipeline, "Graphics");
    pgen.clearShaders();
#if USE_SLANG
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#endif
  }

  void createGbuffers(const glm::vec2& size)
  {
    m_viewSize = size;
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                   VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                   m_colorFormat, m_depthFormat);
  }

  void createMsaaBuffers(const glm::vec2& size)
  {
    m_alloc->destroy(m_msaaColor);
    m_alloc->destroy(m_msaaDepth);
    vkDestroyImageView(m_device, m_msaaColorIView, nullptr);
    vkDestroyImageView(m_device, m_msaaDepthIView, nullptr);

    // Default create image info
    VkImageCreateInfo create_info = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    create_info.imageType         = VK_IMAGE_TYPE_2D;
    create_info.samples           = m_msaaSamples;  // #MSAA
    create_info.mipLevels         = 1;
    create_info.arrayLayers       = 1;
    create_info.extent.width      = static_cast<uint32_t>(size.x);
    create_info.extent.height     = static_cast<uint32_t>(size.y);
    create_info.extent.depth      = 1;

    // Creating color
    {
      create_info.format = m_gBuffers->getColorFormat();
      create_info.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // #MSAA - Optimization
      m_msaaColor = m_alloc->createImage(create_info);
      m_dutil->setObjectName(m_msaaColor.image, "msaaColor");
      const VkImageViewCreateInfo iv_info = nvvk::makeImageViewCreateInfo(m_msaaColor.image, create_info);
      vkCreateImageView(m_device, &iv_info, nullptr, &m_msaaColorIView);
    }

    // Creating the depth buffer
    {
      create_info.format = m_gBuffers->getDepthFormat();
      create_info.usage  = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

      m_msaaDepth = m_alloc->createImage(create_info);
      m_dutil->setObjectName(m_msaaDepth.image, "msaaDepth");
      VkImageViewCreateInfo iv_info       = nvvk::makeImageViewCreateInfo(m_msaaDepth.image, create_info);
      iv_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      vkCreateImageView(m_device, &iv_info, nullptr, &m_msaaDepthIView);
    }

    // Setting the image layout for both color and depth
    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();

      nvvk::cmdBarrierImageLayout(cmd, m_msaaColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      nvvk::cmdBarrierImageLayout(cmd, m_msaaDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
      m_app->submitAndWaitTempCmdBuffer(cmd);
    }
  }

  void createVkBuffers()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_meshVk.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_meshVk[i];
      m.vertices         = m_alloc->createBuffer(cmd, m_meshes[i].vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m.indices          = m_alloc->createBuffer(cmd, m_meshes[i].triangles, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);
    }

    m_frameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_frameInfo.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    for(PrimitiveMeshVk& m : m_meshVk)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_frameInfo);

    m_dset->deinit();
    m_gBuffers.reset();

    // #MSAA
    m_alloc->destroy(m_msaaColor);
    m_alloc->destroy(m_msaaDepth);
    vkDestroyImageView(m_device, m_msaaColorIView, nullptr);
    vkDestroyImageView(m_device, m_msaaDepthIView, nullptr);
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers->getColorImage(), m_gBuffers->getSize(),
                           nvh::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                        m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>            m_dutil;
  std::shared_ptr<nvvk::ResourceAllocatorDma> m_alloc;

  // #MSAA
  nvvk::Image           m_msaaColor;
  nvvk::Image           m_msaaDepth;
  VkImageView           m_msaaColorIView{VK_NULL_HANDLE};
  VkImageView           m_msaaDepthIView{VK_NULL_HANDLE};
  VkSampleCountFlagBits m_msaaSamples{VK_SAMPLE_COUNT_4_BIT};

  glm::vec2                        m_viewSize    = {0, 0};
  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.4F, 0.4F, 0.6F, 1.F}};      // Clear color
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;                            // Descriptor set

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
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;
  std::vector<Material>           m_materials;

  // Pipeline
  DH::PushConstant m_pushConst{};                        // Information sent to the shader
  VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  int              m_frame{0};
};

//////////////////////////////////////////////////////////////////////////
///
///
int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo appInfo;

  nvh::CommandLineParser cli(PROJECT_NAME);
  cli.addArgument({"--headless"}, &appInfo.headless, "Run in headless mode");
  cli.parse(argc, argv);

  VkContextSettings vkSetup;
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});

  // Vulkan context creation
  VulkanContext vkContext(vkSetup);
  if(!vkContext.isValid())
    std::exit(0);

  load_VK_EXTENSIONS(vkContext.getInstance(), vkGetInstanceProcAddr, vkContext.getDevice(), vkGetDeviceProcAddr);  // Loading the Vulkan extension pointers

  // Application setup
  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appInfo);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<Msaa>());

  app->run();
  app.reset();
  vkContext.deinit();

  return test->errorCode();
}
