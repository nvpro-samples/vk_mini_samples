/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*

 This sample loads a KTX image with the `nv_ktx::KTXImage`
 
 - The library requires to also link with libzstd_static, zlibstatic and basisu
   This can be found in the CMakeLists.txt

 Note that KTX can be linear or sRgb, since our framebuffer aren't set as sRGB, we also
 need a tonemapper. We could have simply computed the result value in the 
 fragment shader, but it is more interesting to see that a second pass 
 can be added, therefore making it more flexible. 




*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "fileformats/nv_ktx.h"
#include "imgui/imgui_camera_widget.h"
#include "nvh/fileoperations.hpp"
#include "nvh/primitives.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"

#include "_autogen/raster.frag.h"
#include "_autogen/raster.vert.h"
#include "shaders/device_host.h"


//--
static std::string g_img_file = R"(media/fruit.ktx2)";
//--

constexpr bool g_use_tm_compute = true;

// Texture wrapper class which load an KTX image
struct TextureKtx
{

  TextureKtx(nvvk::Context* c, nvvkhl::AllocVma* a, const std::string& filename)
      : m_ctx(c)
      , m_alloc(a)
  {
    nv_ktx::KTXImage           ktx_image;
    const nv_ktx::ReadSettings ktx_read_settings;
    nv_ktx::ErrorWithText      maybe_error = ktx_image.readFromFile(filename.c_str(), ktx_read_settings);
    if(maybe_error.has_value())
    {
      LOGE("KTX Error: %s\n", maybe_error->c_str());
    }
    m_dutil = std::make_unique<nvvk::DebugUtil>(c->m_device);  // Debug utility

    // Check if format is supported
    VkImageFormatProperties prop{};
    vkGetPhysicalDeviceImageFormatProperties(c->m_physicalDevice, ktx_image.format, VK_IMAGE_TYPE_2D,
                                             VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, 0, &prop);
    assert(prop.maxResourceSize != 0);

    create(ktx_image);
  }

  ~TextureKtx()
  {  // Destroying in next frame, avoid deleting while using
    nvvkhl::Application::submitResourceFree([tex = m_texture, a = m_alloc]() {
      auto t = tex;
      a->destroy(t);
    });
  }

  // Create the image, the sampler and the image view + generate the mipmap level for all
  void create(nv_ktx::KTXImage& ktximage)
  {
    const VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    const VkFormat            format = ktximage.format;

    nvvk::CommandPool cpool(m_ctx->m_device, m_ctx->m_queueGCT.familyIndex);
    auto*             cmd = cpool.createCommandBuffer();


    auto              img_size        = VkExtent2D{ktximage.mip_0_width, ktximage.mip_0_height};
    VkImageCreateInfo img_create_info = nvvk::makeImage2DCreateInfo(img_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);
    img_create_info.mipLevels         = ktximage.num_mips;

    // Creating image level 0
    std::vector<char>& data         = ktximage.subresource();
    const VkDeviceSize buffer_size  = data.size();
    const nvvk::Image  result_image = m_alloc->createImage(cmd, buffer_size, data.data(), img_create_info);

    // Create all mip-levels
    nvvk::cmdBarrierImageLayout(cmd, result_image.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    auto* staging = m_alloc->getStaging();
    for(uint32_t mip = 1; mip < ktximage.num_mips; mip++)
    {
      img_create_info.extent.width  = std::max(1U, ktximage.mip_0_width >> mip);
      img_create_info.extent.height = std::max(1U, ktximage.mip_0_height >> mip);

      const VkOffset3D         offset{};
      VkImageSubresourceLayers subresource{};
      subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.layerCount = 1;
      subresource.mipLevel   = mip;

      std::vector<char>& mipresource = ktximage.subresource(mip, 0, 0);
      const VkDeviceSize buffer_size = mipresource.size();
      if(img_create_info.extent.width > 0 && img_create_info.extent.height > 0)
      {
        staging->cmdToImage(cmd, result_image.image, offset, img_create_info.extent, subresource, buffer_size,
                            mipresource.data());
      }
    }
    nvvk::cmdBarrierImageLayout(cmd, result_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Texture
    const VkImageViewCreateInfo iv_info = nvvk::makeImageViewCreateInfo(result_image.image, img_create_info);
    m_texture                           = m_alloc->createTexture(result_image, iv_info, sampler_info);
    m_dutil->DBG_NAME(m_texture.image);
    m_dutil->DBG_NAME(m_texture.descriptor.sampler);

    cpool.submitAndWait(cmd);
  }

  [[nodiscard]] bool                         valid() const { return m_texture.image != VK_NULL_HANDLE; }
  [[nodiscard]] const VkDescriptorImageInfo& descriptorImage() const { return m_texture.descriptor; }

private:
  nvvk::Context*                   m_ctx{nullptr};
  nvvkhl::AllocVma*                m_alloc{nullptr};
  std::unique_ptr<nvvk::DebugUtil> m_dutil;

  VkExtent2D    m_size{0, 0};
  nvvk::Texture m_texture;
};


//////////////////////////////////////////////////////////////////////////
class ImageKtx : public nvvkhl::IAppElement
{
public:
  ImageKtx()           = default;
  ~ImageKtx() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil      = std::make_unique<nvvk::DebugUtil>(m_device);                    // Debug utility
    m_alloc      = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());  // Allocator
    m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(m_app->getContext().get(), m_alloc.get());
    m_dset       = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // Find image file
    const std::vector<std::string> default_search_paths = {".", "..", "../..", "../../.."};
    const std::string              img_file             = nvh::findFile(g_img_file, default_search_paths, true);
    assert(!img_file.empty());
    m_texture = std::make_shared<TextureKtx>(m_app->getContext().get(), m_alloc.get(), img_file);
    assert(m_texture->valid());

    createScene();
    createVkBuffers();
    createPipeline();

    if(g_use_tm_compute)
    {
      m_tonemapper->createComputePipeline();
    }
    else
    {
      m_tonemapper->createGraphicPipeline(m_gBuffers->getColorFormat(0), m_gBuffers->getDepthFormat());
    }
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onUIMenu() override
  {
    static bool close_app{false};
    static bool show_demo{false};
    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
      {
        close_app = true;
      }
      if(ImGui::MenuItem("Show Demo"))
      {
        show_demo = true;
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
    if(show_demo)
    {
      ImGui::ShowDemoWindow(&show_demo);
    }
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    createGbuffers({width, height});

    // Tonemapper is using GBuffer1 as input
    if(g_use_tm_compute)
    {
      m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(1), m_gBuffers->getDescriptorImageInfo(0));
    }
    else
    {
      m_tonemapper->updateGraphicDescriptorSets(m_gBuffers->getDescriptorImageInfo(1));
    }
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();
      if(ImGui::CollapsingHeader("Tonemapper", ImGuiTreeNodeFlags_DefaultOpen))
      {
        m_tonemapper->onUI();
      }
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer0 image
      if(m_gBuffers)
        ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    if(!m_gBuffers)
      return;

    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    const float view_aspect_ratio = m_viewSize.x / m_viewSize.y;

    // Update Frame buffer uniform buffer
    FrameInfo            finfo{};
    const nvmath::vec2f& clip = CameraManip.getClipPlanes();
    finfo.view                = CameraManip.getMatrix();
    finfo.proj                = nvmath::perspectiveVK(CameraManip.getFov(), view_aspect_ratio, clip.x, clip.y);
    finfo.camPos              = CameraManip.getEye();
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(FrameInfo), &finfo);

    renderScene(cmd);  // Render to GBuffer-1
    renderPost(cmd);   // Use GBuffer-1 and render to GBuffer-0
  }


private:
  void createScene()
  {
    m_meshes.emplace_back(nvh::createSphereUv());
    m_materials.push_back({vec4(1)});
    nvh::Node& n = m_nodes.emplace_back();
    n.mesh       = 0;
    n.material   = 0;

    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({0.0F, 0.0F, 1.5F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});

    // Set clear color in sRgb space
    nvmath::vec3f c = toLinear({0.3F, 0.3F, 0.3F});
    memcpy(m_clearColor.float32, &c.x, sizeof(vec3));
  }


  void renderScene(VkCommandBuffer cmd)
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    // Drawing the scene in GBuffer-1
    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView(1)},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmd, &r_info);
    m_app->setViewport(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_dset->getSets(m_frame), 0, nullptr);
    const VkDeviceSize offsets{0};
    for(const nvh::Node& n : m_nodes)
    {
      const PrimitiveMeshVk& m = m_meshVk[n.mesh];
      // Push constant information
      m_pushConst.transfo = n.localMatrix();
      m_pushConst.color   = m_materials[n.material].color;
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(PushConstant), &m_pushConst);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m.vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      auto num_indices = static_cast<uint32_t>(m_meshes[n.mesh].triangles.size() * 3);
      vkCmdDrawIndexed(cmd, num_indices, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);
  }

  void renderPost(VkCommandBuffer cmd)
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    if(g_use_tm_compute)
    {
      // Compute
      auto size = VkExtent2D{static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)};
      m_tonemapper->runCompute(cmd, size);
    }
    else
    {
      // Graphic

      // Tonemapping GBuffer-1 to GBuffer-0
      nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView(0)},
                                       m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                       VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
      r_info.pStencilAttachment = nullptr;

      vkCmdBeginRendering(cmd, &r_info);
      m_tonemapper->runGraphic(cmd);
      vkCmdEndRendering(cmd);
    }
  }

  void createPipeline()
  {
    m_dset->addBinding(BKtxFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    m_dset->addBinding(BKtxTex, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_dset->initLayout();
    m_dset->initPool(2);  // two frames - allow to change on the fly

    // Writing to descriptors
    const VkDescriptorBufferInfo      dbi_unif{m_frameInfo.buffer, 0, VK_WHOLE_SIZE};
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_dset->makeWrite(0, 0, &dbi_unif));
    writes.emplace_back(m_dset->makeWrite(0, BKtxTex, &m_texture->descriptorImage()));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    const VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                      sizeof(PushConstant)};

    VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    create_info.pushConstantRangeCount = 1;
    create_info.pPushConstantRanges    = &push_constant_ranges;
    create_info.setLayoutCount         = 1;
    create_info.pSetLayouts            = &m_dset->getLayout();
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &create_info, nullptr, &m_pipelineLayout));

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_srgbFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    // Creating the Pipeline
    nvvk::GraphicsPipelineState pstate;
    pstate.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    pstate.addBindingDescriptions({{0, sizeof(nvh::PrimitiveVertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, p))},  // Position
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, n))},  // Normal
        {2, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, t))},     // TexCoord
    });

    nvvk::GraphicsPipelineGenerator pgen(m_device, m_pipelineLayout, prend_info, pstate);
    pgen.addShader(std::vector<uint32_t>{std::begin(raster_vert), std::end(raster_vert)}, VK_SHADER_STAGE_VERTEX_BIT);
    pgen.addShader(std::vector<uint32_t>{std::begin(raster_frag), std::end(raster_frag)}, VK_SHADER_STAGE_FRAGMENT_BIT);

    m_graphicsPipeline = pgen.createPipeline();
    m_dutil->setObjectName(m_graphicsPipeline, "Graphics");
    pgen.clearShaders();
  }

  void createGbuffers(const VkExtent2D& size)
  {
    m_viewSize = {size.width, size.height};
    // Create two color GBuffers: 0-final result, 1-scene in sRGB
    const std::vector<VkFormat> color_buffers = {m_colorFormat, m_srgbFormat};
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), size, color_buffers, m_depthFormat);
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

    m_frameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_frameInfo.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    m_texture.reset();

    for(PrimitiveMeshVk& m : m_meshVk)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_frameInfo);

    m_dset->deinit();
    m_gBuffers.reset();
    m_tonemapper.reset();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*              m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>  m_dutil;
  std::shared_ptr<nvvkhl::AllocVma> m_alloc;

  nvmath::vec2f                    m_viewSize    = {0, 0};
  VkFormat                         m_colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the image
  VkFormat                         m_srgbFormat  = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.0F, 0.0F, 0.0F, 1.0F}};     // Clear color
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk> m_meshVk;
  nvvk::Buffer                 m_frameInfo;

  std::vector<VkSampler> m_samplers;

  // Data and setting
  struct Material
  {
    vec4 color{1.F};
  };
  std::vector<nvh::PrimitiveMesh>                m_meshes;
  std::vector<nvh::Node>                         m_nodes;
  std::vector<Material>                          m_materials;
  std::shared_ptr<TextureKtx>                    m_texture;
  std::unique_ptr<nvvk::DescriptorSetContainer>  m_dset;  // Descriptor set
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;


  // Pipeline
  PushConstant     m_pushConst{};                        // Information sent to the shader
  VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  int              m_frame{0};
};


//////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = PROJECT_NAME " Example";
  spec.vSync            = true;
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  spec.vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<ImageKtx>());

  app->run();
  app.reset();

  return test->errorCode();
}
