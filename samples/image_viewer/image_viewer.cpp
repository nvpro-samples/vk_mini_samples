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

/*

 This sample shows how to load and display an image.
 - Render to a GBuffer and displayed using ImGui
 - The image is applied as a texture on a quad.
 - It is possible to change the sampling filters on the fly, with proper flagging of descriptor layout.
 - Zoom and pan the image under the cursor

*/

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION  // Implementation of the image loading library

#include <array>

#include <GLFW/glfw3.h>
#undef APIENTRY


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb/stb_image.h>

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on
#include <imgui/imgui.h>


#include <nvapp/application.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/mipmaps.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>

#include "common/utils.hpp"

// Our compiled shaders
#include "_autogen/image_viewer.frag.glsl.h"
#include "_autogen/image_viewer.slang.h"
#include "_autogen/image_viewer.vert.glsl.h"


// Texture wrapper class which load an image
struct SampleTexture
{
  SampleTexture(nvvk::ResourceAllocator* alloc)
      : m_alloc(alloc)
  {
    m_device = m_alloc->getDevice();
  }

  ~SampleTexture() { m_alloc->destroyImage(const_cast<nvvk::Image&>(m_image)); }

  void createFromFile(VkCommandBuffer cmd, nvvk::StagingUploader& staging, const std::filesystem::path& filename)
  {
    int      w, h, comp = 0;
    stbi_uc* data = stbi_load(filename.string().c_str(), &w, &h, &comp, 4);
    if((data != nullptr) && w > 1 && h > 1)
    {
      create(cmd, staging, {uint32_t(w), uint32_t(h)}, std::span<uint8_t>(data, w * h * 4));
      stbi_image_free(data);
    }
  }

  // Create the image, the sampler and the image view + generate the mipmap level for all
  void create(VkCommandBuffer cmd, nvvk::StagingUploader& uploader, VkExtent2D size, const std::span<uint8_t>& data)
  {
    m_size = size;

    const VkFormat      format      = VK_FORMAT_R8G8B8A8_UNORM;
    const VkImageLayout imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkImageCreateInfo createInfo = DEFAULT_VkImageCreateInfo;
    createInfo.mipLevels         = nvvk::mipLevels(m_size);
    createInfo.extent            = {m_size.width, m_size.height, 1};
    createInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    createInfo.format            = format;

    NVVK_CHECK(m_alloc->createImage(m_image, createInfo, DEFAULT_VkImageViewCreateInfo));
    NVVK_DBG_NAME(m_image.image);
    NVVK_DBG_NAME(m_image.descriptor.imageView);
    NVVK_CHECK(uploader.appendImage(m_image, data, imageLayout));

    // run copy prior mipmaps
    uploader.cmdUploadAppended(cmd);

    nvvk::cmdGenerateMipmaps(cmd, m_image.image, m_size, createInfo.mipLevels);
  }

  void               setSampler(const VkSampler& sampler) { m_image.descriptor.sampler = sampler; }
  [[nodiscard]] bool isValid() const { return m_image.image != nullptr; }
  [[nodiscard]] const VkDescriptorImageInfo& descriptor() const { return m_image.descriptor; }
  [[nodiscard]] const VkExtent2D&            getSize() const { return m_size; }
  [[nodiscard]] float getAspect() const { return static_cast<float>(m_size.width) / static_cast<float>(m_size.height); }

private:
  VkDevice                 m_device{};
  nvvk::ResourceAllocator* m_alloc{nullptr};
  VkExtent2D               m_size{0, 0};
  nvvk::Image              m_image;
};


struct ImageViewerSettings
{
  float     zoom = {1};
  glm::vec2 pan  = {0, 0};
} g_imageViewerSettings;


//////////////////////////////////////////////////////////////////////////
/// </summary> Display an image on a quad.
class ImageViewer : public nvapp::IAppElement
{
public:
  ImageViewer() = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    // Creating the allocator
    m_alloc.init({
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });

    // set up staging
    m_stagingUploader.init(&m_alloc, true);

    // Acquiring the sampler which will be used for displaying the GBuffer and the texture
    m_samplerPool.init(app->getDevice());
    createSamplers();

    // Creating the G-Buffer, a single color attachment, no depth-stencil
    VkSampler linearSampler;
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);
    m_gBuffers.init({.allocator      = &m_alloc,
                     .colorFormats   = {VK_FORMAT_R8G8B8A8_UNORM},
                     .imageSampler   = linearSampler,
                     .descriptorPool = m_app->getTextureDescriptorPool()});


    // Find image file and create the texture
    const std::filesystem::path imageFilename = nvutils::findFile("fruit.jpg", nvsamples::getResourcesDirs());
    assert(!imageFilename.empty());
    m_texture           = std::make_shared<SampleTexture>(&m_alloc);
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_texture->createFromFile(cmd, m_stagingUploader, imageFilename);
    m_texture->setSampler(m_samplers[0]);  // Default to nearest
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();
    assert(m_texture->isValid());

    createPipeline();
    createVkBuffers();  // The geometry is a simple quad
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);
    vkDestroyShaderEXT(m_device, m_vertexShader, nullptr);
    vkDestroyShaderEXT(m_device, m_fragmentShader, nullptr);

    m_alloc.destroyBuffer(m_vertices);
    m_alloc.destroyBuffer(m_indices);

    m_vertices = {};
    m_indices  = {};

    m_stagingUploader.deinit();
    m_samplerPool.deinit();
    m_texture.reset();
    m_gBuffers.deinit();
    m_alloc.deinit();
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

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { NVVK_CHECK(m_gBuffers.update(cmd, size)); }

  void onUIRender() override
  {
    // Setting menu
    {
      ImGui::Begin("Settings");
      ImGui::SliderFloat("Zoom", &g_imageViewerSettings.zoom, 0.01F, 2.0F, nullptr, ImGuiSliderFlags_Logarithmic);
      ImGui::SliderFloat2("Pan", &g_imageViewerSettings.pan.x, -1.F, 1.0F);

      {  // Sampling filters
        static int mode   = 0;
        bool       change = false;
        change |= ImGui::RadioButton("Nearest", &mode, 0);
        ImGui::SameLine();
        change |= ImGui::RadioButton("Linear", &mode, 1);
        if(change)
        {
          m_texture->setSampler(m_samplers[mode]);
        }
      }
      if(ImGui::Button("Reset"))
      {
        g_imageViewerSettings.zoom = 1;
        g_imageViewerSettings.pan  = {0, 0};
      }
      ImGui::SameLine();
      if(ImGui::Button("1:1"))
      {
        g_imageViewerSettings.zoom =
            static_cast<float>(m_texture->getSize().width) / static_cast<float>(m_gBuffers.getSize().width);
        g_imageViewerSettings.pan = {0, 0};
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

        glm::vec2       mousePos = ImGui::GetMousePos();              // Current mouse pos in window
        const glm::vec2 corner   = ImGui::GetCursorScreenPos();       // Corner of the viewport
        mousePos                 = (mousePos - corner) - size / 2.F;  // Mouse pos relative to center of viewport
        const glm::vec2 pan = mousePos * (2.F / g_imageViewerSettings.zoom) / size;  // Position in image space before zoom

        // Change zoom on mouse wheel
        if(io.MouseWheel > 0)
        {
          g_imageViewerSettings.zoom *= 1.1F;
        }
        if(io.MouseWheel < 0)
        {
          g_imageViewerSettings.zoom /= 1.1F;
        }

        const glm::vec2 pan2 = mousePos * (2.F / g_imageViewerSettings.zoom) / size;  // Position in image space after zoom
        g_imageViewerSettings.pan += pan2 - pan;  // Re-adjust panning (making zoom relative to mouse cursor)

        const glm::vec2 drag = ImGui::GetMouseDragDelta(0, 0);                          // Get the amount of mouse drag
        ImGui::ResetMouseDragDelta();                                                   // We want static move
        g_imageViewerSettings.pan += drag * (2.F / g_imageViewerSettings.zoom) / size;  // Drag in image space
      }

      // Display the G-Buffer image
      ImGui::Image(ImTextureID(m_gBuffers.getDescriptorSet()), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }

    // Window Title
    {
      static float dirtyTimer = 0.0F;
      dirtyTimer += ImGui::GetIO().DeltaTime;
      if(dirtyTimer > 1.0F)  // Refresh every seconds
      {
        std::array<char, 256> buf{};
        snprintf(buf.data(), buf.size(), "%s %dx%d | %d FPS / %.3fms", nvutils::getExecutablePath().stem().string().c_str(),
                 static_cast<int>(m_gBuffers.getSize().width), static_cast<int>(m_gBuffers.getSize().height),
                 static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
        glfwSetWindowTitle(m_app->getWindowHandle(), buf.data());
        dirtyTimer = 0;
      }
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);

    // Adjusting the aspect ratio of the image
    const float imgAspectRatio  = m_texture->getAspect();
    const float viewAspectRatio = m_gBuffers.getAspectRatio();

    m_pushConst.scale = {1.0F, 1.0F};

    bool  isImgWider = imgAspectRatio > viewAspectRatio;
    float ratio      = isImgWider ? viewAspectRatio / imgAspectRatio : imgAspectRatio / viewAspectRatio;

    // If aspect ratio <= 1, scale x for wider images and y for taller ones
    bool scale_x = (isImgWider ? imgAspectRatio : viewAspectRatio) <= 1;
    if(scale_x)
    {
      m_pushConst.scale.x = ratio;
    }
    else
    {
      m_pushConst.scale.y = ratio;
    }

    // Applying the zoom and pan
    const glm::mat4 ortho = glm::ortho(-1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F);
    const glm::mat4 scale = glm::scale(glm::mat4(1), glm::vec3(g_imageViewerSettings.zoom, g_imageViewerSettings.zoom, 0));
    const glm::mat4 trans =
        glm::translate(glm::mat4(1), glm::vec3(g_imageViewerSettings.pan.x, g_imageViewerSettings.pan.y, 0));
    m_pushConst.transfo = ortho * scale * trans;

    // Drawing the quad in a G-Buffer
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView();

    // Create the rendering info
    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea           = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;

    nvvk::cmdImageMemoryBarrier(cmd, {.image        = m_gBuffers.getColorImage(),
                                      .oldLayout    = VK_IMAGE_LAYOUT_GENERAL,
                                      .newLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                      .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                      .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT});

    vkCmdBeginRendering(cmd, &renderingInfo);
    {
      const VkDeviceSize offsets[] = {0};

      m_dynamicPipeline.cmdApplyAllStates(cmd);
      m_dynamicPipeline.cmdSetViewportAndScissor(cmd, m_app->getViewportSize());
      m_dynamicPipeline.cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_fragmentShader});

      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant), &m_pushConst);

      updateTexture(cmd);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, offsets);
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
      vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);

    nvvk::cmdImageMemoryBarrier(cmd, {.image        = m_gBuffers.getColorImage(),
                                      .oldLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                      .newLayout    = VK_IMAGE_LAYOUT_GENERAL,
                                      .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                      .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT});
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
    m_descBind.addBinding(
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT});
    NVVK_CHECK(m_descBind.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT, &m_descriptorSetLayout));
    NVVK_DBG_NAME(m_descriptorSetLayout);

    // Pipeline layout
    const VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorSetLayout}, {pushConstantRanges}));
    NVVK_DBG_NAME(m_pipelineLayout);

    m_dynamicPipeline.vertexBindings = {
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .stride = sizeof(Vertex), .divisor = 1}};
    m_dynamicPipeline.vertexAttributes = {{.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                           .location = 0,
                                           .format   = VK_FORMAT_R32G32_SFLOAT,
                                           .offset   = offsetof(Vertex, pos)},
                                          {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                           .location = 1,
                                           .format   = VK_FORMAT_R32G32_SFLOAT,
                                           .offset   = offsetof(Vertex, uv)}};

    // Creating the shaders
    VkShaderCreateInfoEXT shaderInfo{
        .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRanges,
    };
#if USE_SLANG
    shaderInfo.codeSize  = image_viewer_slang_sizeInBytes;
    shaderInfo.pCode     = image_viewer_slang;
    shaderInfo.pName     = "vertexMain";
    shaderInfo.stage     = VK_SHADER_STAGE_VERTEX_BIT;
    shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
    vkCreateShadersEXT(m_app->getDevice(), 1U, &shaderInfo, nullptr, &m_vertexShader);
    NVVK_DBG_NAME(m_vertexShader);
    shaderInfo.pName     = "fragmentMain";
    shaderInfo.stage     = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderInfo.nextStage = 0;
    vkCreateShadersEXT(m_app->getDevice(), 1U, &shaderInfo, nullptr, &m_fragmentShader);
    NVVK_DBG_NAME(m_fragmentShader);
#else
    shaderInfo.pName    = "main";
    shaderInfo.codeSize = std::span(image_viewer_vert_glsl).size_bytes();
    shaderInfo.pCode    = std::span(image_viewer_vert_glsl).data();
    shaderInfo.stage    = VK_SHADER_STAGE_VERTEX_BIT;
    vkCreateShadersEXT(m_app->getDevice(), 1U, &shaderInfo, nullptr, &m_vertexShader);
    NVVK_DBG_NAME(m_vertexShader);
    shaderInfo.pName    = "main";
    shaderInfo.stage    = VK_SHADER_STAGE_FRAGMENT_BIT;
    shaderInfo.codeSize = std::span(image_viewer_frag_glsl).size_bytes();
    shaderInfo.pCode    = std::span(image_viewer_frag_glsl).data();
    vkCreateShadersEXT(m_app->getDevice(), 1U, &shaderInfo, nullptr, &m_fragmentShader);
    NVVK_DBG_NAME(m_fragmentShader);
#endif
  }

  // Push the image descriptor, such that it can be used in shader
  void updateTexture(VkCommandBuffer cmd)
  {
    nvvk::WriteSetContainer container;
    container.append(m_descBind.getWriteSet(0), m_texture->descriptor());

    VkPushDescriptorSetInfo pushInfo{.sType                = VK_STRUCTURE_TYPE_PUSH_DESCRIPTOR_SET_INFO,
                                     .stageFlags           = VK_SHADER_STAGE_ALL_GRAPHICS,
                                     .layout               = m_pipelineLayout,
                                     .descriptorWriteCount = container.size(),
                                     .pDescriptorWrites    = container.data()};

    vkCmdPushDescriptorSet2(cmd, &pushInfo);
  }

  // Creating the geometry and pushing it to the GPU
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
      assert(m_stagingUploader.isAppendedEmpty());
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      NVVK_CHECK(m_alloc.createBuffer(m_vertices, std::span(vertices).size_bytes(), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
      NVVK_CHECK(m_alloc.createBuffer(m_indices, std::span(indices).size_bytes(), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
      NVVK_DBG_NAME(m_vertices.buffer);
      NVVK_DBG_NAME(m_indices.buffer);
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_vertices, 0, std::span(vertices)));
      NVVK_CHECK(m_stagingUploader.appendBuffer(m_indices, 0, std::span(indices)));
      m_stagingUploader.cmdUploadAppended(cmd);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      m_stagingUploader.releaseStaging();
    }
  }

  // Create two samplers, one nearest, one linear
  void createSamplers()
  {
    m_samplers.resize(2);

    VkSamplerCreateInfo sampler_info{
        .sType      = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter  = VK_FILTER_NEAREST,
        .minFilter  = VK_FILTER_NEAREST,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
    };

    NVVK_CHECK(m_samplerPool.acquireSampler(m_samplers[0], sampler_info));
    NVVK_DBG_NAME(m_samplers[0]);

    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    NVVK_CHECK(m_samplerPool.acquireSampler(m_samplers[1], sampler_info));
    NVVK_DBG_NAME(m_samplers[1]);
  }

  // Saving the buffer to disk
  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*     m_app{};
  nvvk::ResourceAllocator m_alloc;
  nvvk::StagingUploader   m_stagingUploader{};
  nvvk::SamplerPool       m_samplerPool;
  nvvk::GBuffer           m_gBuffers;  // G-Buffers: color

  VkDevice m_device{};  // Convenient

  // Resources
  nvvk::Buffer           m_vertices;  // Buffer of the vertices
  nvvk::Buffer           m_indices;   // Buffer of the indices
  std::vector<VkSampler> m_samplers;

  // Data and setting
  PushConstant                   m_pushConst;  // Information sent to the shader
  std::shared_ptr<SampleTexture> m_texture;    // Loaded image and displayed

  // Pipeline
  VkDescriptorSetLayout       m_descriptorSetLayout{};  // Descriptor set layout
  VkPipelineLayout            m_pipelineLayout{};       // The description of the pipeline
  nvvk::GraphicsPipelineState m_dynamicPipeline;
  nvvk::DescriptorBindings    m_descBind;

  // Shaders
  VkShaderEXT m_vertexShader{};
  VkShaderEXT m_fragmentShader{};
};


//////////////////////////////////////////////////////////////////////////
///
int main(int argc, char** argv)
{
  nvapp::Application           app;        // Main application
  nvapp::ApplicationCreateInfo appInfo;    // Base application information
  nvvk::ContextInitInfo        vkSetup;    // Vulkan context information
  nvvk::Context                vkContext;  // Vulkan context

  // Parsing the command line
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  reg.add({"zoom", "Zoom in image"}, &g_imageViewerSettings.zoom);
  reg.addVector({"pan", "Pan in image"}, &g_imageViewerSettings.pan);
  reg.addVector({"size", "Window size"}, &appInfo.windowSize);
  cli.add(reg);
  cli.parse(argc, argv);

  // Vulkan creation context information
  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dStateFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  vkSetup = {.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
             .deviceExtensions   = {
                 {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
                 {VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, &dStateFeatures},
                 {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
             }};

  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Creation of the Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Setting up the application
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();
  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);

  // Create the application and add the image viewer sample
  app.init(appInfo);
  app.addElement(std::make_shared<ImageViewer>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
