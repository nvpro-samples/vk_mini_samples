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

/*

 This sample shows how to load and display an image.
 - Render to a render target and displayed using ImGui
 - The image is applied as a texture on a quad.
 - Texturing uses VK_EXT_descriptor_heap bindless-style: nvvk::DescriptorHeap + shaders that
   index layout(descriptor_heap) / Slang DescriptorHandle with spvDescriptorHeapEXT. Push data
   carries transform/scale and samplerIdx (read in the shader to choose a sampler heap slot).
 - Zoom and pan the image under the cursor

*/

#define USE_SLANG true
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION  // Implementation of the image loading library

#include <array>
#include <cstddef>

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
#include <fmt/format.h>

#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptor_heap.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/mipmaps.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>

#include <nvapp/imgui_texture.hpp>

#include "common/utils.hpp"

// Our compiled shaders
#include "_autogen/image_viewer.frag.glsl.h"
#include "_autogen/image_viewer.slang.h"
#include "_autogen/image_viewer.vert.glsl.h"


// Texture wrapper class which load an image
struct SampleTexture
{
  explicit SampleTexture(nvvk::ResourceAllocator* alloc)
      : m_alloc(alloc)
  {
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

  [[nodiscard]] bool              isValid() const { return m_image.image != VK_NULL_HANDLE; }
  [[nodiscard]] VkImage           getImage() const { return m_image.image; }
  [[nodiscard]] VkFormat          getFormat() const { return m_image.format; }
  [[nodiscard]] const VkExtent2D& getSize() const { return m_size; }
  [[nodiscard]] float getAspect() const { return static_cast<float>(m_size.width) / static_cast<float>(m_size.height); }

private:
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

    // Allocator: buffer device address required for descriptor heap buffers
    m_alloc.init({
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });

    m_stagingUploader.init(&m_alloc, true);

    m_samplerPool.init(app->getDevice());

    // Offscreen render target: color only (no depth)
    NVVK_CHECK(m_renderTarget.init({
        .alloc        = &m_alloc,
        .colorFormats = {VK_FORMAT_R8G8B8A8_UNORM},
        .debugName    = "ImageViewer",
    }));

    // No buffers needed in this sample's resource heap (0 buffer count).
    NVVK_CHECK(m_heap.init(app->getPhysicalDevice(), app->getDevice()));

    const VkBufferUsageFlags2 heapUsage       = nvvk::DescriptorHeap::getRequiredBufferUsage();
    VkDeviceSize              samplerBufSize  = m_heap.setupSamplerHeap(2);
    VkDeviceSize              resourceBufSize = m_heap.setupResourceHeap(1, 0);  // 1 image + 0 buffer
    NVVK_CHECK(m_alloc.createBuffer(m_samplerHeapBuffer, samplerBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO, {},
                                    m_heap.getSamplerHeapAlignment()));
    NVVK_CHECK(m_alloc.createBuffer(m_resourceHeapBuffer, resourceBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                    m_heap.getResourceHeapAlignment()));
    NVVK_DBG_NAME(m_samplerHeapBuffer.buffer);
    NVVK_DBG_NAME(m_resourceHeapBuffer.buffer);

    const std::filesystem::path imageFilename = nvutils::findFile("fruit.jpg", nvsamples::getResourcesDirs());
    assert(!imageFilename.empty());
    m_texture           = std::make_shared<SampleTexture>(&m_alloc);
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_texture->createFromFile(cmd, m_stagingUploader, imageFilename);
    assert(m_texture->isValid());

    // --- Method A: staging upload (sampler heap is device-local only) ---
    // appendBufferMapping returns a writable pointer into the staging buffer;
    // vkWriteSamplerDescriptorsEXT writes land there directly (zero intermediate copy).
    void* smpMapping = nullptr;
    NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_samplerHeapBuffer, 0, m_heap.getSamplerHeapSize(), smpMapping));

    VkSamplerCreateInfo nearestSamplerCI{
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = VK_FILTER_NEAREST,
        .minFilter    = VK_FILTER_NEAREST,
        .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    m_nearestSamplerIdx = m_heap.acquireSamplerDescriptor(nearestSamplerCI, smpMapping);

    VkSamplerCreateInfo linearSamplerCI{
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = VK_FILTER_LINEAR,
        .minFilter    = VK_FILTER_LINEAR,
        .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
    };
    m_linearSamplerIdx = m_heap.acquireSamplerDescriptor(linearSamplerCI, smpMapping);

    m_stagingUploader.cmdUploadAppended(cmd);

    // --- Method B: persistently mapped buffer (resource heap is host-visible) ---
    // vkWriteResourceDescriptorsEXT writes land directly in device-visible memory;
    // no staging upload needed at all.
    NVVK_CHECK(m_heap.writeSampledImageDescriptor(0, m_texture->getImage(), m_texture->getFormat(),
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_resourceHeapBuffer.mapping));

    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    createPipeline();
    createVkBuffers();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    vkDestroyShaderEXT(m_device, m_vertexShader, nullptr);
    vkDestroyShaderEXT(m_device, m_fragmentShader, nullptr);
    m_vertexShader   = VK_NULL_HANDLE;
    m_fragmentShader = VK_NULL_HANDLE;

    m_heap.releaseSamplerDescriptor(m_nearestSamplerIdx);
    m_heap.releaseSamplerDescriptor(m_linearSamplerIdx);
    m_alloc.destroyBuffer(m_samplerHeapBuffer);
    m_alloc.destroyBuffer(m_resourceHeapBuffer);

    m_samplerHeapBuffer  = {};
    m_resourceHeapBuffer = {};
    m_heap.deinit();

    m_alloc.destroyBuffer(m_vertices);
    m_alloc.destroyBuffer(m_indices);

    m_vertices = {};
    m_indices  = {};

    m_stagingUploader.deinit();
    m_samplerPool.deinit();
    m_texture.reset();
    m_viewportImage.deinit();
    m_renderTarget.deinit();
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

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    NVVK_CHECK(m_renderTarget.update(cmd, size));
    m_viewportImage.update(m_renderTarget.getUiImageView());
  }

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
          m_samplerIdx = (mode == 0) ? m_nearestSamplerIdx : m_linearSamplerIdx;
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
            static_cast<float>(m_texture->getSize().width) / static_cast<float>(m_renderTarget.getSize().width);
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

      // Display the rendered image
      ImGui::Image(m_viewportImage, ImGui::GetContentRegionAvail());

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
                 static_cast<int>(m_renderTarget.getSize().width), static_cast<int>(m_renderTarget.getSize().height),
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
    const float viewAspectRatio = m_renderTarget.getAspectRatio();

    m_pushData.scale = {1.0F, 1.0F};

    bool  isImgWider = imgAspectRatio > viewAspectRatio;
    float ratio      = isImgWider ? viewAspectRatio / imgAspectRatio : imgAspectRatio / viewAspectRatio;

    bool scale_x = (isImgWider ? imgAspectRatio : viewAspectRatio) <= 1;
    if(scale_x)
    {
      m_pushData.scale.x = ratio;
    }
    else
    {
      m_pushData.scale.y = ratio;
    }

    const glm::mat4 ortho = glm::ortho(-1.0F, 1.0F, -1.0F, 1.0F, -1.0F, 1.0F);
    const glm::mat4 scale = glm::scale(glm::mat4(1), glm::vec3(g_imageViewerSettings.zoom, g_imageViewerSettings.zoom, 0));
    const glm::mat4 trans =
        glm::translate(glm::mat4(1), glm::vec3(g_imageViewerSettings.pan.x, g_imageViewerSettings.pan.y, 0));
    m_pushData.transfo    = ortho * scale * trans;
    m_pushData.samplerIdx = m_samplerIdx;

    // Drawing the quad in the render target. The render target keeps its images
    // in VK_IMAGE_LAYOUT_GENERAL, so no layout transitions are needed to render
    // into it or to sample it afterwards.
    nvvk::RenderTargetState rtState;
    m_renderTarget.fillState(rtState);
    nvvk::RenderTargetState::AttachmentOps ops{};  // default: clear+store on color & depth, don't care on stencil
    rtState.cmdBeginRendering(cmd, ops);

    {
      const VkDeviceSize offsets[] = {0};

      m_dynamicPipeline.cmdApplyAllStates(cmd);
      m_dynamicPipeline.cmdSetViewportAndScissor(cmd, m_renderTarget.getSize());
      m_dynamicPipeline.cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_fragmentShader});

      m_heap.cmdBindHeaps(cmd, m_samplerHeapBuffer.address, m_resourceHeapBuffer.address);

      VkPushDataInfoEXT pushInfo{.sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
                                 .offset = 0,
                                 .data   = {.address = &m_pushData, .size = sizeof(PushData)}};
      vkCmdPushDataEXT(cmd, &pushInfo);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, offsets);
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
      vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
  }

private:
  struct Vertex
  {
    glm::vec2 pos;
    glm::vec2 uv;
  };

  // Matches shader push_constant / [[vk::push_constant]]. samplerIdx is read in the fragment
  // stage (GLSL: flat varying; Slang: FragmentInput.samplerIdx) to index the sampler heap.
  struct PushData
  {
    glm::mat4 transfo{1};
    glm::vec2 scale{1};
    uint32_t  samplerIdx{};
  };
  static_assert(sizeof(PushData) == 76, "PushData must match shader push layout");

  void createPipeline()
  {
    // Bindless descriptor heap: SPIR-V uses SPV_EXT_descriptor_heap directly; no
    // VkShaderDescriptorSetAndBindingMappingInfoEXT (per vk_mini_samples descriptor_heap bindless path).
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

    VkShaderCreateFlagsEXT flags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT | VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT;

    std::array<VkShaderCreateInfoEXT, 2> createInfos{};
    createInfos[0].sType     = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
    createInfos[0].pNext     = nullptr;
    createInfos[0].flags     = flags;
    createInfos[0].stage     = VK_SHADER_STAGE_VERTEX_BIT;
    createInfos[0].nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
    createInfos[0].codeType  = VK_SHADER_CODE_TYPE_SPIRV_EXT;
#if USE_SLANG
    createInfos[0].codeSize = image_viewer_slang_sizeInBytes;
    createInfos[0].pCode    = image_viewer_slang;
    createInfos[0].pName    = "vertexMain";
#else
    createInfos[0].codeSize = std::span(image_viewer_vert_glsl).size_bytes();
    createInfos[0].pCode    = std::span(image_viewer_vert_glsl).data();
    createInfos[0].pName    = "main";
#endif

    createInfos[1].sType     = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
    createInfos[1].pNext     = nullptr;
    createInfos[1].flags     = flags;
    createInfos[1].stage     = VK_SHADER_STAGE_FRAGMENT_BIT;
    createInfos[1].nextStage = 0;
    createInfos[1].codeType  = VK_SHADER_CODE_TYPE_SPIRV_EXT;
#if USE_SLANG
    createInfos[1].codeSize = image_viewer_slang_sizeInBytes;
    createInfos[1].pCode    = image_viewer_slang;
    createInfos[1].pName    = "fragmentMain";
#else
    createInfos[1].codeSize = std::span(image_viewer_frag_glsl).size_bytes();
    createInfos[1].pCode    = std::span(image_viewer_frag_glsl).data();
    createInfos[1].pName    = "main";
#endif

    std::array<VkShaderEXT, 2> shaders{};
    NVVK_CHECK(vkCreateShadersEXT(m_device, static_cast<uint32_t>(createInfos.size()), createInfos.data(), nullptr,
                                  shaders.data()));
    m_vertexShader   = shaders[0];
    m_fragmentShader = shaders[1];
    NVVK_DBG_NAME(m_vertexShader);
    NVVK_DBG_NAME(m_fragmentShader);
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

  // Saving the buffer to disk
  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_renderTarget.getColorImage(), m_renderTarget.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*     m_app{};
  nvvk::ResourceAllocator m_alloc;
  nvvk::DescriptorHeap    m_heap{};
  nvvk::StagingUploader   m_stagingUploader{};
  nvvk::SamplerPool       m_samplerPool;
  nvvk::RenderTarget      m_renderTarget;   // Offscreen render target: color only
  nvapp::ImTexture        m_viewportImage;  // ImGui texture for the render target color image

  VkDevice m_device{};

  nvvk::Buffer m_samplerHeapBuffer{};
  nvvk::Buffer m_resourceHeapBuffer{};
  nvvk::Buffer m_vertices;
  nvvk::Buffer m_indices;

  PushData                       m_pushData{};
  uint32_t                       m_samplerIdx{};
  uint32_t                       m_nearestSamplerIdx{};
  uint32_t                       m_linearSamplerIdx{};
  std::shared_ptr<SampleTexture> m_texture;

  nvvk::GraphicsPipelineState m_dynamicPipeline;

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
  bool                       verbose = false;
  reg.add({"verbose", "Verbose output of the Vulkan context"}, &verbose);
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  reg.add({"zoom", "Zoom in image"}, &g_imageViewerSettings.zoom);
  reg.addVector({"pan", "Pan in image"}, &g_imageViewerSettings.pan);
  reg.addVector({"size", "Window size"}, &appInfo.windowSize);
  cli.add(reg);
  cli.parse(argc, argv);

  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dStateFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  VkPhysicalDeviceDescriptorHeapFeaturesEXT heapFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT};
  heapFeatures.descriptorHeap = VK_TRUE;
  VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untypedPtrFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR};
  untypedPtrFeatures.shaderUntypedPointers = VK_TRUE;

  vkSetup = {.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
             .deviceExtensions   = {
                 {VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, &dStateFeatures},
                 {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
                 {VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, &heapFeatures},
                 {VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, &untypedPtrFeatures},
             }};

  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Creation of the Vulkan context
  vkSetup.verbose |= verbose;
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
  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);

  // Create the application and add the image viewer sample
  app.init(appInfo);
  app.addElement(std::make_shared<ImageViewer>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
