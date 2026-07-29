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

 This sample renders many simple polygons with the builtin camera.

 It demonstrates VK_EXT_descriptor_heap on a *classic graphics pipeline* (VkPipeline),
 without shader objects. The per-frame camera UBO is reached through a bound descriptor
 heap instead of a descriptor set: the pipeline opts in with
 VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT (the pipeline-path counterpart of the
 shader-object flag VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT). A descriptor-heap pipeline
 uses NO VkPipelineLayout (the spec requires layout == VK_NULL_HANDLE), so push data is
 sent with vkCmdPushDataEXT rather than vkCmdPushConstants. vkCmdBindPipeline and the rest
 of the raster setup are unchanged. See the `descriptor_heap` sample for the concept in depth.

*/
//////////////////////////////////////////////////////////////////////////

#define USE_SLANG true
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

#include "_autogen/polygons_raster.frag.glsl.h"
#include "_autogen/polygons_raster.slang.h"  // Pre-compiled shader
#include "_autogen/polygons_raster.vert.glsl.h"

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/camera.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/primitives.hpp>
#include <nvvk/buffer_suballocator.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptor_heap.hpp>  // nvvk::DescriptorHeap (bindless resources)
#include <nvapp/imgui_texture.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>

// The camera for the scene
std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};


//////////////////////////////////////////////////////////////////////////
/// Shows many simple polygons
class SimplePolygons : public nvapp::IAppElement
{
public:
  SimplePolygons()           = default;
  ~SimplePolygons() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    m_allocator = std::make_unique<nvvk::ResourceAllocator>();
    NVVK_CHECK(m_allocator->init(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    }));  // Allocator

    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());

    // Offscreen render target: one color attachment + depth (depth supports transfer readback for picking)
    NVVK_CHECK(m_renderTarget.init({
        .alloc        = m_allocator.get(),
        .colorFormats = {m_colorFormat},
        .depthFormat  = m_depthFormat,
        .depthUsage   = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .debugName = "SimplePolygons",
    }));

    createScene();
    createVkBuffers();
    createPipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);

    for(PrimitiveMeshVk& m : m_meshVk)
    {
      m_meshAllocator.subFree(m.vertices);
      m_meshAllocator.subFree(m.indices);
    }
    m_meshAllocator.deinit();

    m_allocator->destroyBuffer(m_frameInfo);
    m_allocator->destroyBuffer(m_pixelBuffer);
    m_allocator->destroyBuffer(m_resourceHeapBuffer);

    m_heap.deinit();

    m_viewportImage.deinit();
    m_renderTarget.deinit();
    m_allocator->deinit();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    NVVK_CHECK(m_renderTarget.update(cmd, size));
    m_viewportImage.update(m_renderTarget.getUiImageView());
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      nvgui::CameraWidget(g_cameraManip);
      ImGui::Separator();
      ImGui::Text("Double-click an object to focus the view");
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Deal with mouse interaction only if the window has focus
      if(ImGui::IsWindowHovered(ImGuiFocusedFlags_RootWindow) && ImGui::IsMouseDoubleClicked(0))
      {
        rasterPicking();
      }

      // Display the rendered image
      ImGui::Image(m_viewportImage, ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    // Update Frame buffer uniform buffer
    shaderio::FrameInfo finfo{};
    finfo.view   = g_cameraManip->getViewMatrix();
    finfo.proj   = g_cameraManip->getPerspectiveMatrix();
    finfo.camPos = g_cameraManip->getEye();
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

    // Build the dynamic-rendering state from the render target. The render target
    // keeps its images in VK_IMAGE_LAYOUT_GENERAL, so no layout transitions are
    // needed to render into it or to sample it afterwards.
    nvvk::RenderTargetState rtState;
    m_renderTarget.fillState(rtState);
    rtState.colorAttachments[0].clearValue = {m_clearColor};
    rtState.depthAttachment.clearValue     = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    // Start the rendering
    nvvk::RenderTargetState::AttachmentOps ops{};  // default: clear+store on color & depth, don't care on stencil
    rtState.cmdBeginRendering(cmd, ops);

    m_graphicState.cmdSetViewportAndScissor(cmd, m_renderTarget.getSize());

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    // Bind the resource descriptor heap in place of vkCmdBindDescriptorSets. This sample has no
    // sampler heap, so the sampler-heap address is 0. Shaders then index resources by heap slot.
    m_heap.cmdBindHeaps(cmd, 0, m_resourceHeapBuffer.address);

    VkBuffer lastVertexBuffer = {};
    VkBuffer lastIndexBuffer  = {};

    for(const nvutils::Node& node : m_nodes)
    {
      PrimitiveMeshVk& mesh = m_meshVk[node.mesh];
      // Push constant information. A descriptor-heap pipeline has no VkPipelineLayout, so push data
      // is delivered with vkCmdPushDataEXT instead of vkCmdPushConstants.
      m_pushConst.transfo = node.localMatrix();
      m_pushConst.color   = m_materials[node.material].color;
      const VkPushDataInfoEXT pushInfo{.sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
                                       .offset = 0,
                                       .data   = {.address = &m_pushConst, .size = sizeof(shaderio::PushConstant)}};
      vkCmdPushDataEXT(cmd, &pushInfo);

      nvvk::BufferRange vertexBufferRange = m_meshAllocator.subRange(mesh.vertices);
      nvvk::BufferRange indexBufferRange  = m_meshAllocator.subRange(mesh.indices);

      // Due to sub allocation we can do less binds.
      // Note we bind the full buffers here and use the sub allocation offsets in the draw call
      if(vertexBufferRange.buffer != lastVertexBuffer)
      {
        const VkDeviceSize vertexOffset = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBufferRange.buffer, &vertexOffset);
        lastVertexBuffer = vertexBufferRange.buffer;
      }
      if(indexBufferRange.buffer != lastIndexBuffer)
      {
        vkCmdBindIndexBuffer(cmd, indexBufferRange.buffer, 0, VK_INDEX_TYPE_UINT32);
        lastIndexBuffer = indexBufferRange.buffer;
      }

      uint32_t numIndices = static_cast<uint32_t>(m_meshes[node.mesh].triangles.size() * 3);

      // we adjust firstIndex and vertexOffset according to the sub allocation offsets
      uint32_t firstIndex   = static_cast<uint32_t>(indexBufferRange.offset / sizeof(uint32_t));
      int32_t  vertexOffset = static_cast<int32_t>(vertexBufferRange.offset / sizeof(nvutils::PrimitiveVertex));

      vkCmdDrawIndexed(cmd, numIndices, 1, firstIndex, vertexOffset, 0);
    }
    vkCmdEndRendering(cmd);
  }

private:
  void createScene()
  {
    // Meshes
    m_meshes.emplace_back(nvutils::createSphereMesh(0.5F, 3));
    m_meshes.emplace_back(nvutils::createSphereUv(0.5F, 30, 30));
    m_meshes.emplace_back(nvutils::createCube(1.0F, 1.0F, 1.0F));
    m_meshes.emplace_back(nvutils::createTetrahedron());
    m_meshes.emplace_back(nvutils::createOctahedron());
    m_meshes.emplace_back(nvutils::createIcosahedron());
    m_meshes.emplace_back(nvutils::createConeMesh(0.5F, 1.0F, 32));
    m_meshes.emplace_back(nvutils::createTorusMesh(0.5F, 0.25F, 32, 16));

    const int num_meshes = static_cast<int>(m_meshes.size());

    // Materials (colorful)
    for(int i = 0; i < num_meshes; i++)
    {
      const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
      const glm::vec3 v    = static_cast<glm::vec3>(sin(freq) * 0.5F + 0.5F);
      m_materials.push_back({glm::vec4(v, 1)});
    }

    // Instances
    int   elemPerRow = (int)std::sqrt(num_meshes) + 1;
    int   elemPerCol = (num_meshes - 1) / elemPerRow + 1;
    float spacing    = 2.0f;
    for(int i = 0; i < num_meshes; i++)
    {
      nvutils::Node& n = m_nodes.emplace_back();
      n.mesh           = i;
      n.material       = i;
      n.translation.x  = (i % elemPerRow - elemPerRow / 2.F + 0.5F) * spacing;
      n.translation.y  = 0.f;
      n.translation.z  = (i / elemPerRow - elemPerCol / 2.F + 0.5F) * spacing;
    }

    g_cameraManip->setClipPlanes({0.1F, 100.0F});
    g_cameraManip->setLookat({4.5F, 4.5F, 2.5F}, {0.F, 0.F, 0.F}, {0.0F, 1.0F, 0.0F});
  }

  void createPipeline()
  {
    // Descriptor-heap pipelines use NO VkPipelineLayout at all: the spec requires layout ==
    // VK_NULL_HANDLE when VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT is set (see
    // VUID-VkGraphicsPipelineCreateInfo-flags-11311). Resources are reached through the bound heap,
    // and push data is sent later with vkCmdPushDataEXT.

    // Creating the Pipeline
    m_graphicState.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    m_graphicState.vertexBindings              = {{.sType   = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                                                   .stride  = sizeof(nvutils::PrimitiveVertex),
                                                   .divisor = 1}};
    m_graphicState.vertexAttributes            = {{.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                                   .location = 0,
                                                   .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                                   .offset   = offsetof(nvutils::PrimitiveVertex, pos)},
                                                  {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                                   .location = 1,
                                                   .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                                   .offset   = offsetof(nvutils::PrimitiveVertex, nrm)}};

    // Helper to create the graphic pipeline
    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = VK_NULL_HANDLE;  // required by the descriptor-heap flag
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;

    // Opt the pipeline into descriptor-heap access. This is the pipeline-path equivalent of the
    // shader-object flag VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT; the creator chains it into the
    // VkGraphicsPipelineCreateInfo for us.
    creator.flags2 = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;


    // Adding the shaders to the pipeline
    std::array<VkShaderModule, 2> shaderModules{};
#if USE_SLANG
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", polygons_raster_slang);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", polygons_raster_slang);
#else
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", polygons_raster_vert_glsl);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", polygons_raster_frag_glsl);
#endif

    NVVK_CHECK(creator.createGraphicsPipeline(m_device, nullptr, m_graphicState, &m_pipeline));
    NVVK_DBG_NAME(m_pipeline);
  }

  void createVkBuffers()
  {
    // We allocate several tiny meshes and therefore make use of the BufferSubAllocator class
    // to avoid many small VkBuffers.

    nvvk::BufferSubAllocator::InitInfo bufferSubInitInfo;
    bufferSubInitInfo.resourceAllocator = m_allocator.get();
    bufferSubInitInfo.debugName         = "meshAllocator";
    bufferSubInitInfo.usageFlags        = VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT;
    bufferSubInitInfo.blockSize         = 1 * 1024 * 1024;  // this scene is very small 1 MB is known to be enough

    NVVK_CHECK(m_meshAllocator.init(bufferSubInitInfo));

    nvvk::StagingUploader uploader;
    uploader.init(m_allocator.get());

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_meshVk.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_meshVk[i];
      NVVK_CHECK(m_meshAllocator.subAllocate(m.vertices, std::span(m_meshes[i].vertices).size_bytes(),
                                             uint32_t(sizeof(nvutils::PrimitiveVertex))));
      NVVK_CHECK(m_meshAllocator.subAllocate(m.indices, std::span(m_meshes[i].triangles).size_bytes()));
      NVVK_CHECK(uploader.appendBufferRange(m_meshAllocator.subRange(m.vertices), std::span(m_meshes[i].vertices)));
      NVVK_CHECK(uploader.appendBufferRange(m_meshAllocator.subRange(m.indices), std::span(m_meshes[i].triangles)));
    }

    // The heap descriptor references the buffer by device address, so it needs SHADER_DEVICE_ADDRESS.
    NVVK_CHECK(m_allocator->createBuffer(m_frameInfo, sizeof(shaderio::FrameInfo),
                                         VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                         VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
    NVVK_DBG_NAME(m_frameInfo.buffer);

    NVVK_CHECK(m_allocator->createBuffer(m_pixelBuffer, sizeof(float) * 4, VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
    NVVK_DBG_NAME(m_pixelBuffer.buffer);

    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();

    // Descriptor heap: size a resource heap for this sample's buffer slots (no images), back it with
    // a host-visible mapped buffer, and register the per-frame UBO into slot kHeapBufFrameInfo. The
    // shader then reaches the UBO through the heap instead of a descriptor set.
    NVVK_CHECK(m_heap.init(m_app->getPhysicalDevice(), m_device));
    const VkDeviceSize resourceBufSize = m_heap.setupResourceHeap(0, shaderio::kHeapBufCount);
    constexpr auto     heapUsage       = nvvk::DescriptorHeap::getRequiredBufferUsage();
    NVVK_CHECK(m_allocator->createBuffer(m_resourceHeapBuffer, resourceBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO,
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                         m_heap.getResourceHeapAlignment()));
    NVVK_DBG_NAME(m_resourceHeapBuffer.buffer);

    NVVK_CHECK(m_heap.writeBufferDescriptor(shaderio::kHeapBufFrameInfo, m_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                            m_resourceHeapBuffer.mapping));
  }


  //--------------------------------------------------------------------------------------------------
  // Find the 3D position under the mouse cursor and set the camera interest to this position
  //
  void rasterPicking()
  {
    glm::vec2       mouse_pos = ImGui::GetMousePos();         // Current mouse pos in window
    const glm::vec2 corner    = ImGui::GetCursorScreenPos();  // Corner of the viewport
    mouse_pos                 = mouse_pos - corner;           // Mouse pos relative to center of viewport

    const glm::mat4 view = g_cameraManip->getViewMatrix();
    glm::mat4       proj = g_cameraManip->getPerspectiveMatrix();

    // Find the distance under the cursor
    const float d = getDepth(static_cast<int>(mouse_pos.x), static_cast<int>(mouse_pos.y));

    if(d < 1.0F)  // Ignore infinite
    {
      glm::vec4       win_norm = {0, 0, m_renderTarget.getSize().width, m_renderTarget.getSize().height};
      const glm::vec3 hit_pos  = glm::unProjectZO({mouse_pos.x, mouse_pos.y, d}, view, proj, win_norm);

      // Set the interest position
      glm::dvec3 eye, center, up;
      g_cameraManip->getLookat(eye, center, up);
      g_cameraManip->setLookat(eye, hit_pos, up, false);
    }
  }

  //--------------------------------------------------------------------------------------------------
  // Read the depth buffer at the X,Y coordinates
  // Note: depth format is VK_FORMAT_D32_SFLOAT
  //
  float getDepth(int x, int y)
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    // Transit the depth buffer image in eTransferSrcOptimal
    nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.getDepthImage(),
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

    // Copy the pixel under the cursor
    VkBufferImageCopy copy_region{};
    copy_region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    copy_region.imageOffset      = {x, y, 0};
    copy_region.imageExtent      = {1, 1, 1};
    vkCmdCopyImageToBuffer(cmd, m_renderTarget.getDepthImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_pixelBuffer.buffer, 1, &copy_region);

    // Put back the depth buffer as  it was
    nvvk::cmdImageMemoryBarrier(cmd, {m_renderTarget.getDepthImage(),
                                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
    m_app->submitAndWaitTempCmdBuffer(cmd);


    // Grab the value
    float    value{1.0F};
    void*    mapped      = m_pixelBuffer.mapping;
    VkFormat depthFormat = m_renderTarget.getDepthFormat();
    switch(depthFormat)
    {
      case VK_FORMAT_X8_D24_UNORM_PACK32:
      case VK_FORMAT_D24_UNORM_S8_UINT: {
        uint32_t ivalue{0};
        memcpy(&ivalue, mapped, sizeof(uint32_t));
        const uint32_t mask = (1 << 24) - 1;
        ivalue              = ivalue & mask;
        value               = float(ivalue) / float(mask);
      }
      break;
      case VK_FORMAT_D32_SFLOAT: {
        memcpy(&value, mapped, sizeof(float));
      }
      break;
      case VK_FORMAT_D16_UNORM: {
        uint16_t ivalue{0};
        memcpy(&ivalue, mapped, sizeof(uint16_t));
        value = float(ivalue) / float(0xFFFF);
      }
      break;
      default:
        assert(!"Wrong Format");
    }

    return value;
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_renderTarget.getColorImage(), m_renderTarget.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*                      m_app{nullptr};
  std::shared_ptr<nvvk::ResourceAllocator> m_allocator;

  VkFormat           m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat           m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue  m_clearColor  = {{0.3F, 0.3F, 0.3F, 1.0F}};     // Clear color
  VkDevice           m_device      = VK_NULL_HANDLE;                 // Convenient
  nvvk::RenderTarget m_renderTarget;                                 // Offscreen render target: color + depth
  nvapp::ImTexture   m_viewportImage;                                // ImGui texture for the render target color image

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::BufferSubAllocation vertices;  // BufferSubAllocation of the vertices
    nvvk::BufferSubAllocation indices;   // BufferSubAllocation of the indices
  };
  std::vector<PrimitiveMeshVk> m_meshVk;
  nvvk::BufferSubAllocator     m_meshAllocator;

  nvvk::Buffer m_frameInfo;
  nvvk::Buffer m_pixelBuffer;


  // Data and setting
  struct Material
  {
    glm::vec4 color{1.F};
  };
  std::vector<nvutils::PrimitiveMesh> m_meshes;     // All primitive meshes
  std::vector<nvutils::Node>          m_nodes;      // All nodes/instances
  std::vector<Material>               m_materials;  // All materials

  // Pipeline
  shaderio::PushConstant      m_pushConst{};   // Information sent to the shader
  nvvk::GraphicsPipelineState m_graphicState;  // State of the graphic pipeline

  VkPipeline m_pipeline{};  // Graphic pipeline (no VkPipelineLayout: descriptor-heap pipeline)

  // Descriptor heap (bindless): replaces the descriptor pool/layout/set
  nvvk::DescriptorHeap m_heap{};                // Resource heap for this sample's buffer(s)
  nvvk::Buffer         m_resourceHeapBuffer{};  // Host-visible buffer backing the resource heap
};

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  bool                       verbose = false;
  reg.add({"verbose", "Verbose output of the Vulkan context"}, &verbose);
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  nvvk::ContextInitInfo vkSetup;
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
  vkSetup.instanceExtensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

  // Descriptor heap on a classic pipeline: only the heap extension (and the untyped-pointer
  // extension it relies on) are required. Note there is intentionally no VK_EXT_shader_object or
  // VK_EXT_extended_dynamic_state3 here -- this sample keeps a traditional VkPipeline.
  VkPhysicalDeviceDescriptorHeapFeaturesEXT heapFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT};
  VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untypedPtrFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR};
  vkSetup.deviceExtensions.insert(vkSetup.deviceExtensions.begin(),
                                  {
                                      {VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, &heapFeatures},
                                      {VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, &untypedPtrFeatures},
                                  });

  // Adding validation layers
  nvvk::ValidationSettings vvlInfo{};
  vvlInfo.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();

  // Create Vulkan context
  nvvk::Context vkContext;
  vkSetup.verbose |= verbose;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    std::exit(0);
  }

  // Application setup
  appInfo.name           = fmt::format("{} ({})", nvutils::getExecutablePath().stem().string(), SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Camera manipulator (global)
  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(g_cameraManip);

  // Add all application elements
  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app.addElement(std::make_shared<SimplePolygons>());

  app.run();
  app.deinit();

  vkContext.deinit();

  return 0;
}
