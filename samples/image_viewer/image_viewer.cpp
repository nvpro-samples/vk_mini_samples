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
 
 This sample shows how to load and display an image.
 - Render to a GBuffer and displayed using ImGui
 - The image is applied as a texture on a quad.
 - It is possible to change the sampling filters on the fly (using 2 sets) (see m_frame)
 - Zoom and pan the image under the cursor

*/
//////////////////////////////////////////////////////////////////////////
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#include <array>
#include <imgui.h>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvk/extensions_vk.hpp"
#include "common/vk_context.hpp"


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


// Texture wrapper class which load an image
struct SampleTexture
{
  SampleTexture(VkDevice device, uint32_t queueIndex, nvvkhl::AllocVma* a)
      : m_device(device)
      , m_queueIndex(queueIndex)
      , m_alloc(a)
  {
    m_size = {1, 1};
    std::array<uint8_t, 4> data{255, 255, 0, 255};
    create(4, data.data());
  }

  SampleTexture(VkDevice device, uint32_t queueIndex, nvvkhl::AllocVma* a, const std::string& filename)
      : m_device(device)
      , m_queueIndex(queueIndex)
      , m_alloc(a)
  {
    int      w        = 0;
    int      h        = 0;
    int      comp     = 0;
    int      req_comp = 4;
    stbi_uc* data     = stbi_load(filename.c_str(), &w, &h, &comp, req_comp);
    if((data != nullptr) && w > 1 && h > 1)
    {
      m_size = {static_cast<uint32_t>(w), static_cast<uint32_t>(h)};
      create(w * h * req_comp, data);
      stbi_image_free(data);
    }
  }

  ~SampleTexture()
  {  // Destroying in next frame, avoid deleting while using
    m_alloc->destroy(const_cast<nvvk::Texture&>(m_texture));
  }

  // Create the image, the sampler and the image view + generate the mipmap level for all
  void create(uint32_t bufsize, void* data)
  {
    const VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    const VkFormat            format    = VK_FORMAT_R8G8B8A8_UNORM;
    const VkImageCreateInfo create_info = nvvk::makeImage2DCreateInfo(m_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

    nvvk::CommandPool cpool(m_device, m_queueIndex);
    VkCommandBuffer   cmd = cpool.createCommandBuffer();
    m_texture             = m_alloc->createTexture(cmd, bufsize, data, create_info, sampler_info);
    nvvk::cmdGenerateMipmaps(cmd, m_texture.image, format, m_size, create_info.mipLevels);
    cpool.submitAndWait(cmd);
  }

  void               setSampler(const VkSampler& sampler) { m_texture.descriptor.sampler = sampler; }
  [[nodiscard]] bool isValid() const { return m_texture.image != nullptr; }
  [[nodiscard]] const VkDescriptorImageInfo& descriptor() const { return m_texture.descriptor; }
  [[nodiscard]] const VkExtent2D&            getSize() const { return m_size; }
  [[nodiscard]] float getAspect() const { return static_cast<float>(m_size.width) / static_cast<float>(m_size.height); }

private:
  VkDevice          m_device{};
  uint32_t          m_queueIndex{0};
  VkExtent2D        m_size{0, 0};
  nvvk::Texture     m_texture;
  nvvkhl::AllocVma* m_alloc{nullptr};
};

//////////////////////////////////////////////////////////////////////////
/// </summary> Display an image on a quad.
class ImageViewer : public nvvkhl::IAppElement
{
public:
  ImageViewer() = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
    m_alloc = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    });  // Allocator
    m_dset  = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // Find image file
    const std::vector<std::string> default_search_paths = {".", "..", "../..", "../../.."};
    const std::string              img_file = nvh::findFile(R"(media/fruit.jpg)", default_search_paths, true);
    assert(!img_file.empty());
    m_texture = std::make_shared<SampleTexture>(m_app->getDevice(), m_app->getQueue(0).familyIndex, m_alloc.get(), img_file);
    assert(m_texture->isValid());

    createSamplers();
    createPipeline();
    createVkBuffers();
    updateTexture();
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

  void onResize(uint32_t width, uint32_t height) override { createGbuffers({width, height}); }

  void onUIRender() override
  {
    // Setting menu
    {
      ImGui::Begin("Settings");
      ImGui::SliderFloat("Zoom", &m_zoom, 0.01F, 2.0F, nullptr, ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat2("Pan", &m_pan.x, -1.F, 1.0F);

      {  // Sampling filters
        static int mode   = 0;
        bool       change = false;
        change |= ImGui::RadioButton("Nearest", &mode, 0);
        ImGui::SameLine();
        change |= ImGui::RadioButton("Linear", &mode, 1);
        if(change)
        {
          m_texture->setSampler(m_samplers[mode]);
          updateTexture();
        }
      }
      if(ImGui::Button("Reset"))
      {
        m_zoom = 1;
        m_pan  = {0, 0};
      }
      ImGui::SameLine();
      if(ImGui::Button("1:1"))
      {
        m_zoom = static_cast<float>(m_texture->getSize().width) / static_cast<float>(m_viewSize.x);
        m_pan  = {0, 0};
      }

      ImGui::End();
    }

    //-------------------------
    // Rendering Viewport
    {
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Get size of current viewport
      const glm::vec2 size = ImGui::GetContentRegionAvail();

      // Deal with mouse interaction only if the window has focus
      if(ImGui::IsWindowHovered(ImGuiFocusedFlags_RootWindow))
      {
        const ImGuiIO& io = ImGui::GetIO();

        glm::vec2       mouse_pos = ImGui::GetMousePos();               // Current mouse pos in window
        const glm::vec2 corner    = ImGui::GetCursorScreenPos();        // Corner of the viewport
        mouse_pos                 = (mouse_pos - corner) - size / 2.F;  // Mouse pos relative to center of viewport
        const glm::vec2 pan       = mouse_pos * (2.F / m_zoom) / size;  // Position in image space before zoom

        // Change zoom on mouse wheel
        if(io.MouseWheel > 0)
        {
          m_zoom *= 1.1F;
        }
        if(io.MouseWheel < 0)
        {
          m_zoom /= 1.1F;
        }

        const glm::vec2 pan2 = mouse_pos * (2.F / m_zoom) / size;  // Position in image space after zoom
        m_pan += pan2 - pan;  // Re-adjust panning (making zoom relative to mouse cursor)

        const glm::vec2 drag = ImGui::GetMouseDragDelta(0, 0);  // Get the amount of mouse drag
        ImGui::ResetMouseDragDelta();                           // We want static move
        m_pan += drag * (2.F / m_zoom) / size;                  // Drag in image space
      }

      // Display the G-Buffer image
      if(m_gBuffers)
        ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }

    // Window Title
    {
      static float dirty_timer = 0.0F;
      dirty_timer += ImGui::GetIO().DeltaTime;
      if(dirty_timer > 1.0F)  // Refresh every seconds
      {
        std::array<char, 256> buf{};
        snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms", PROJECT_NAME, static_cast<int>(m_viewSize.x),
                 static_cast<int>(m_viewSize.y), static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
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

    // Adjusting the aspect ratio of the image
    const float img_aspect_ratio  = m_texture->getAspect();
    const float view_aspect_ratio = m_viewSize.x / m_viewSize.y;

    m_pushConst.scale = {1.0F, 1.0F};
    if(img_aspect_ratio > view_aspect_ratio)
    {
      if(img_aspect_ratio <= 1)
      {
        m_pushConst.scale.x = view_aspect_ratio / img_aspect_ratio;
      }
      else
      {
        m_pushConst.scale.y = view_aspect_ratio / img_aspect_ratio;
      }
    }
    else
    {
      if(view_aspect_ratio <= 1)
      {
        m_pushConst.scale.y = img_aspect_ratio / view_aspect_ratio;
      }
      else
      {
        m_pushConst.scale.x = img_aspect_ratio / view_aspect_ratio;
      }
    }

    // Applying the zoom and pan
    const glm::mat4 ortho = glm::ortho(-1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F);
    const glm::mat4 scale = glm::scale(glm::mat4(1), glm::vec3(m_zoom, m_zoom, 0));
    const glm::mat4 trans = glm::translate(glm::mat4(1), glm::vec3(m_pan.x, m_pan.y, 0));
    m_pushConst.transfo   = ortho * scale * trans;

    // Drawing the quad in a G-Buffer
    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView()},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdBeginRendering(cmd, &r_info);
    {
      const VkDeviceSize offsets{0};
      m_app->setViewport(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant), &m_pushConst);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_dset->getSets(m_frame), 0, nullptr);
      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
      vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
  }

private:
  struct Vertex
  {
    glm::vec2 pos;
    glm::vec2 uv;
  };

  struct PushConstant
  {
    glm::mat4 transfo{1};
    glm::vec2 scale{1};
  };

  void createPipeline()
  {
    m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_dset->initLayout();
    m_dset->initPool(2);  // two frames - allow to change textures on the fly
    m_dutil->setObjectName(m_dset->getLayout(), "Texture");


    const VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant)};

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
    pstate.addBindingDescriptions({{0, sizeof(Vertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, pos))},  // Position
        {1, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, uv))},   // Color
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

  void updateTexture()
  {
    m_frame = (m_frame + 1) % 2;
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_dset->makeWrite(m_frame, 0, &m_texture->descriptor()));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void createGbuffers(const glm::vec2& size)
  {
    m_viewSize = size;
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                   VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                   m_colorFormat, m_depthFormat);
  }

  void createVkBuffers()
  {
    // Quad with UV coordinates
    const std::vector<uint16_t> indices = {0, 2, 1, 2, 0, 3};
    std::vector<Vertex>         vertices(4);
    vertices[0] = {{-1.0F, -1.0F}, {0.0F, 0.0F}};
    vertices[1] = {{1.0F, -1.0F}, {1.0F, 0.0F}};
    vertices[2] = {{1.0F, 1.0F}, {1.0F, 1.0F}};
    vertices[3] = {{-1.0F, 1.0F}, {0.0F, 1.0F}};

    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      m_vertices          = m_alloc->createBuffer(cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m_indices           = m_alloc->createBuffer(cmd, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_dutil->DBG_NAME(m_vertices.buffer);
      m_dutil->DBG_NAME(m_indices.buffer);
      m_app->submitAndWaitTempCmdBuffer(cmd);
    }
  }

  void createSamplers()
  {  // Create two samplers, one nearest, one linear
    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter  = VK_FILTER_NEAREST;
    sampler_info.minFilter  = VK_FILTER_NEAREST;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

    m_samplers.emplace_back(m_alloc->acquireSampler(sampler_info));
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    m_samplers.emplace_back(m_alloc->acquireSampler(sampler_info));
  }


  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    m_alloc->destroy(m_vertices);
    m_alloc->destroy(m_indices);
    m_vertices = {};
    m_indices  = {};

    m_texture.reset();
    m_dset->deinit();
    m_gBuffers.reset();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers->getColorImage(), m_gBuffers->getSize(),
                           nvh::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*              m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>  m_dutil;
  std::shared_ptr<nvvkhl::AllocVma> m_alloc;

  glm::vec2                        m_viewSize{0, 0};
  VkFormat                         m_colorFormat{VK_FORMAT_R8G8B8A8_UNORM};       // Color format of the image
  VkFormat                         m_depthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor{{0, 0, 0, 1}};                    // Clear color
  VkDevice                         m_device{VK_NULL_HANDLE};                      // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                    // G-Buffers: color + depth
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;                           // Descriptor set

  // Resources
  nvvk::Buffer           m_vertices;  // Buffer of the vertices
  nvvk::Buffer           m_indices;   // Buffer of the indices
  std::vector<VkSampler> m_samplers;

  // Data and setting
  float                          m_zoom{1};
  glm::vec2                      m_pan{0, 0};
  std::shared_ptr<SampleTexture> m_texture;  // Loaded image and displayed

  // Pipeline
  PushConstant     m_pushConst;                          // Information sent to the shader
  VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  int              m_frame{0};
};


//////////////////////////////////////////////////////////////////////////
/// </summary>
/// <param name="argc"></param>
/// <param name="argv"></param>
/// <returns></returns>
int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo appInfo;

  nvh::CommandLineParser cli(PROJECT_NAME);
  cli.addArgument({"--headless"}, &appInfo.headless, "Run in headless mode");
  cli.parse(argc, argv);

  // Vulkan creation context information
  VkContextSettings vkSetup;
  if(!appInfo.headless)
  {
    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  // Creation of the Vulkan context
  VulkanContext vkContext(vkSetup);
  if(!vkContext.isValid())
    std::exit(0);

  load_VK_EXTENSIONS(vkContext.getInstance(), vkGetInstanceProcAddr, vkContext.getDevice(), vkGetDeviceProcAddr);  // Loading the Vulkan extension pointers

  // Setting up the application
  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appInfo);

  // Create a view/render
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);
  app->addElement(test);
  app->addElement(std::make_shared<ImageViewer>());

  app->run();
  app.reset();
  vkContext.deinit();

  return test->errorCode();
}
