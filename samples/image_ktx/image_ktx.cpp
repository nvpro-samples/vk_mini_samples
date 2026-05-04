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

 This sample loads a KTX image with the `nv_ktx::KTXImage`
 
 - The library requires to also link with libzstd_static, zlibstatic and basisu
   This can be found in the CMakeLists.txt

 Note that KTX can be linear or sRgb, since our framebuffer aren't set as sRGB, we also
 need a tonemapper. We could have simply computed the result value in the 
 fragment shader, but it is more interesting to see that a second pass 
 can be added, therefore making it more flexible. 




*/
//////////////////////////////////////////////////////////////////////////
#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#include <imgui/imgui.h>

#include <array>
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>


#define VMA_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION


// Pre-compiled shaders
#include "_autogen/image_ktx.frag.glsl.h"
#include "_autogen/image_ktx.slang.h"
#include "_autogen/image_ktx.vert.glsl.h"
#include "_autogen/tonemapper.slang.h"

#include "nvshaders/tonemap_functions.h.slang"
#include "shaders/shaderio.h"  // Shared between host and device

#include "common/utils.hpp"


#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/tonemapper.hpp>
#include <nvimageformats/nv_ktx.h>
#include <nvshaders_host/tonemapper.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/primitives.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/descriptor_heap.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include "nvvk/validation_settings.hpp"

std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};

//--
static std::string g_imgFile = "fruit.ktx2";
//--

// Texture wrapper class which load an KTX image
struct TextureKtx
{

  TextureKtx(VkDevice device, VkPhysicalDevice physicalDevice, nvvk::QueueInfo queueInfo, nvvk::ResourceAllocator* allocator)
      : m_device(device)
      , m_physicalDevice(physicalDevice)
      , m_queueInfo(queueInfo)
      , m_alloc(allocator)
  {
  }

  ~TextureKtx()
  {  // Destroying in next frame, avoid deleting while using
    m_alloc->destroyImage(m_image);
  }

  void create(VkCommandBuffer cmd, nvvk::StagingUploader& uploader, const std::filesystem::path& filename)
  {
    nv_ktx::KTXImage           ktx_image;
    const nv_ktx::ReadSettings ktx_read_settings;
    std::ifstream              ktx_file(filename, std::ios::binary);
    nv_ktx::ErrorWithText      maybe_error = ktx_image.readFromStream(ktx_file, ktx_read_settings);
    if(maybe_error.has_value())
    {
      LOGE("KTX Error: %s\n", maybe_error->c_str());
      return;
    }

    // Check if format is supported
    VkImageFormatProperties prop{};
    vkGetPhysicalDeviceImageFormatProperties(m_physicalDevice, ktx_image.format, VK_IMAGE_TYPE_2D,
                                             VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, 0, &prop);
    assert(prop.maxResourceSize != 0);

    create(cmd, uploader, ktx_image);
  }


  // Create the image, the sampler and the image view + generate the mipmap level for all
  void create(VkCommandBuffer cmd, nvvk::StagingUploader& uploader, nv_ktx::KTXImage& ktximage)
  {
    m_format = ktximage.format;


    auto              img_size   = VkExtent2D{ktximage.mip_0_width, ktximage.mip_0_height};
    VkImageCreateInfo createInfo = DEFAULT_VkImageCreateInfo;
    createInfo.mipLevels         = ktximage.num_mips;
    createInfo.extent            = {img_size.width, img_size.height, 1};
    createInfo.format            = ktximage.format;
    createInfo.usage             = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;


    // Creating image level 0
    const VkOffset3D         offset{};
    std::vector<char>&       data = ktximage.subresource();
    VkImageSubresourceLayers subresource{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1};
    NVVK_CHECK(m_alloc->createImage(m_image, createInfo, DEFAULT_VkImageViewCreateInfo));
    NVVK_CHECK(uploader.appendImage(m_image, std::span(data), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL));

    for(uint32_t mip = 1; mip < ktximage.num_mips; mip++)
    {
      createInfo.extent.width  = std::max(1U, ktximage.mip_0_width >> mip);
      createInfo.extent.height = std::max(1U, ktximage.mip_0_height >> mip);

      subresource.mipLevel = mip;

      std::vector<char>& mipresource = ktximage.subresource(mip, 0, 0);
      const VkDeviceSize buffer_size = mipresource.size();
      if(createInfo.extent.width > 0 && createInfo.extent.height > 0)
      {
        uploader.appendImageSub(m_image, offset, createInfo.extent, subresource, buffer_size, mipresource.data(),
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      }
    }

    NVVK_DBG_NAME(m_image.image);
    NVVK_DBG_NAME(m_image.descriptor.imageView);
    uploader.cmdUploadAppended(cmd);

    // Transition image layout to shader read
    m_image.descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    nvvk::cmdImageMemoryBarrier(cmd, {m_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, m_image.descriptor.imageLayout});
  }

  [[nodiscard]] bool                         valid() const { return m_image.image != VK_NULL_HANDLE; }
  [[nodiscard]] const VkDescriptorImageInfo& descriptorImage() const { return m_image.descriptor; }
  void setSampler(const VkSampler& sampler) { m_image.descriptor.sampler = sampler; }


  nvvk::QueueInfo          m_queueInfo{};
  nvvk::ResourceAllocator* m_alloc{};
  nvvk::Image              m_image;
  VkFormat                 m_format{};
  VkDevice                 m_device{};
  VkPhysicalDevice         m_physicalDevice{};
  VkExtent2D               m_size{0, 0};

private:
};


//////////////////////////////////////////////////////////////////////////
class ImageKtx : public nvapp::IAppElement
{

  enum GBufferTargets
  {
    eTonemapped = 0,
    eRendered   = 1,
  };


public:
  ImageKtx()           = default;
  ~ImageKtx() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    m_alloc.init(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    });  // Allocator

    {
      auto code = std::span<const uint32_t>(tonemapper_slang);
      m_tonemapper.init(&m_alloc, code);
    }

    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    // Creating the G-Buffer, a single color attachment, no depth-stencil
    VkSampler linearSampler{};
    m_samplerPool.init(m_device);
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);
    m_gBuffers.init({.allocator      = &m_alloc,
                     .colorFormats   = {m_colorFormat, m_srgbFormat},
                     .depthFormat    = m_depthFormat,
                     .imageSampler   = linearSampler,
                     .descriptorPool = m_app->getTextureDescriptorPool()});


    // Find image file
    const std::vector<std::filesystem::path> defaultSearchPaths = nvsamples::getResourcesDirs();
    const std::filesystem::path              imageFile          = nvutils::findFile(g_imgFile, defaultSearchPaths);
    assert(!imageFile.empty());

    m_texture = std::make_shared<TextureKtx>(m_app->getDevice(), m_app->getPhysicalDevice(), app->getQueue(0), &m_alloc);
    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      m_app->createTempCmdBuffer();
      nvvk::StagingUploader uploader;
      uploader.init(&m_alloc, true);
      m_texture->create(cmd, uploader, imageFile);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      uploader.deinit();
      assert(m_texture->valid());
      m_texture->setSampler(linearSampler);  // Default to nearest
    }

    // Descriptor heap
    m_heap.init(app->getPhysicalDevice(), app->getDevice());
    VkDeviceSize   samplerBufSize  = m_heap.setupSamplerHeap(1);
    VkDeviceSize   resourceBufSize = m_heap.setupResourceHeap(2);  // slot 0: KTX image, slot 1: FrameInfo UBO
    constexpr auto heapUsage       = nvvk::DescriptorHeap::getRequiredBufferUsage();

    NVVK_CHECK(m_alloc.createBuffer(m_samplerHeapBuffer, samplerBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                    m_heap.getSamplerHeapAlignment()));
    NVVK_CHECK(m_alloc.createBuffer(m_resourceHeapBuffer, resourceBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                    m_heap.getResourceHeapAlignment()));

    m_heapSamplerIdx = m_heap.acquireSamplerDescriptor(DEFAULT_VkSamplerCreateInfo, m_samplerHeapBuffer.mapping);
    assert(m_heapSamplerIdx != ~0u);
    NVVK_CHECK(m_heap.writeImageDescriptor(0, m_texture->m_image, m_resourceHeapBuffer.mapping));


    createScene();
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

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { NVVK_CHECK(m_gBuffers.update(cmd, size)); }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      nvgui::CameraWidget(g_cameraManip);
      if(ImGui::CollapsingHeader("Tonemapper", ImGuiTreeNodeFlags_DefaultOpen))
      {
        nvgui::tonemapperWidget(m_tonemapperData);
      }
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer0 image
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(GBufferTargets::eTonemapped), ImGui::GetContentRegionAvail());

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
    // Barrier to make sure the information is transfered
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);


    renderScene(cmd);  // Render to GBuffer-1
    renderPost(cmd);   // Use GBuffer-1 and render to GBuffer-0
  }


private:
  void createScene()
  {
    m_meshes.emplace_back(nvutils::createSphereUv());
    m_materials.push_back({glm::vec4(1)});
    nvutils::Node& node = m_nodes.emplace_back();
    node.mesh           = 0;
    node.material       = 0;

    g_cameraManip->setClipPlanes({0.1F, 100.0F});
    g_cameraManip->setLookat({0.0F, 0.0F, 1.5F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});

    // Set clear color in sRgb space
    glm::vec3 c = shaderio::toLinear(glm::vec3{.3F, 0.3F, 0.3F});
    memcpy(m_clearColor.float32, &c.x, sizeof(glm::vec3));
  }


  void renderScene(VkCommandBuffer cmd)
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    // Drawing the scene in GBuffer-1
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView(GBufferTargets::eRendered);
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


    // Transition GBuffer images to be used as attachments
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(GBufferTargets::eRendered), VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getDepthImage(),
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                      {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
    vkCmdBeginRendering(cmd, &renderingInfo);

    nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_app->getViewportSize());

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_heapPipeline);

    m_heap.cmdBindHeaps(cmd, m_samplerHeapBuffer.address, m_resourceHeapBuffer.address);

    const VkDeviceSize offsets{0};
    for(const nvutils::Node& node : m_nodes)
    {
      const PrimitiveMeshVk& m = m_meshVk[node.mesh];
      // Push constant information
      m_pushConst.transfo = node.localMatrix();
      m_pushConst.color   = m_materials[node.material].color;

      VkPushDataInfoEXT pushInfo{.sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
                                 .offset = 0,
                                 .data   = {.address = &m_pushConst, .size = sizeof(shaderio::PushConstant)}};
      vkCmdPushDataEXT(cmd, &pushInfo);


      vkCmdBindVertexBuffers(cmd, 0, 1, &m.vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      auto num_indices = static_cast<uint32_t>(m_meshes[node.mesh].triangles.size() * 3);
      vkCmdDrawIndexed(cmd, num_indices, 1, 0, 0, 0);
    }

    vkCmdEndRendering(cmd);
    // Transition GBuffer images to be used as textures
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(GBufferTargets::eRendered),
                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getDepthImage(),
                                      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
  }

  void renderPost(VkCommandBuffer cmd)
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    m_tonemapper.runCompute(cmd, m_gBuffers.getSize(), m_tonemapperData,
                            m_gBuffers.getDescriptorImageInfo(GBufferTargets::eRendered),
                            m_gBuffers.getDescriptorImageInfo(GBufferTargets::eTonemapped));
  }

  void createPipeline()
  {
    // Creating the Pipeline
    nvvk::GraphicsPipelineState graphicState;
    graphicState.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    graphicState.vertexBindings              = {{
                     .sType   = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                     .stride  = sizeof(nvutils::PrimitiveVertex),
                     .divisor = 1,
    }};
    graphicState.vertexAttributes            = {
        {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                    .location = 0,
                    .format   = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset   = static_cast<uint32_t>(offsetof(nvutils::PrimitiveVertex, pos))},
        {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                    .location = 1,
                    .format   = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset   = static_cast<uint32_t>(offsetof(nvutils::PrimitiveVertex, nrm))},
        {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                    .location = 2,
                    .format   = VK_FORMAT_R32G32_SFLOAT,
                    .offset   = static_cast<uint32_t>(offsetof(nvutils::PrimitiveVertex, tex))},
    };

    // Create a pipeline that sources its descriptors from the descriptor
    // heap instead of classic descriptor sets. The shader SPIR-V is reused as-is;
    // VkShaderDescriptorSetAndBindingMappingInfoEXT tells the driver where each
    // (set, binding) lives in the heap.
    //
    // Heap layout used by this sample:
    //   sampler heap:  slot 0 = linear sampler
    //   resource heap: slot 0 = KTX texture image (for BKtxTex)
    //                  slot 1 = FrameInfo uniform buffer (for BKtxFrameInfo)
    //

    // Write the FrameInfo uniform buffer descriptor into the resource heap at
    // slot 1. The KTX image was already written at slot 0 in onAttach.
    NVVK_CHECK(m_heap.writeBufferDescriptor(1, m_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, m_resourceHeapBuffer.mapping));

    // --- Set/binding -> heap mapping ---
    // Binding 0 (FrameInfo uniform buffer) -> resource heap slot 1
    const uint32_t resourceStride = static_cast<uint32_t>(m_heap.resourceDescriptorStride());
    const uint32_t samplerStride  = static_cast<uint32_t>(m_heap.samplerDescriptorSize());

    VkDescriptorSetAndBindingMappingEXT uboMapping{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
    uboMapping.descriptorSet                             = 0;
    uboMapping.firstBinding                              = shaderio::BKtxFrameInfo;
    uboMapping.bindingCount                              = 1;
    uboMapping.resourceMask                              = VK_SPIRV_RESOURCE_TYPE_UNIFORM_BUFFER_BIT_EXT;
    uboMapping.source                                    = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    uboMapping.sourceData.constantOffset                 = {};
    uboMapping.sourceData.constantOffset.heapOffset      = 1 * resourceStride;
    uboMapping.sourceData.constantOffset.heapArrayStride = resourceStride;

    // Binding 1 (sampler2D) is a combined image sampler. The image side lives
    // in the resource heap (slot 0); the sampler side lives in the sampler
    // heap (slot 0). HEAP_WITH_CONSTANT_OFFSET carries both offsets at once.
    VkDescriptorSetAndBindingMappingEXT texMapping{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
    texMapping.descriptorSet = 0;
    texMapping.firstBinding  = shaderio::BKtxTex;
    texMapping.bindingCount  = 1;
    texMapping.resourceMask  = VK_SPIRV_RESOURCE_TYPE_COMBINED_SAMPLED_IMAGE_BIT_EXT
                              | VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT | VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
    texMapping.source                                      = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
    texMapping.sourceData.constantOffset                   = {};
    texMapping.sourceData.constantOffset.heapOffset        = 0 * resourceStride;
    texMapping.sourceData.constantOffset.heapArrayStride   = resourceStride;
    texMapping.sourceData.constantOffset.samplerHeapOffset = 0 * samplerStride;
    texMapping.sourceData.constantOffset.samplerHeapArrayStride = samplerStride;

    const std::array<VkDescriptorSetAndBindingMappingEXT, 2> mappings = {uboMapping, texMapping};

    VkShaderDescriptorSetAndBindingMappingInfoEXT mappingInfo{VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT};
    mappingInfo.mappingCount = static_cast<uint32_t>(mappings.size());
    mappingInfo.pMappings    = mappings.data();

    // --- Manual shader stages so we can attach the mapping info via pNext ---
    // GraphicsPipelineCreator::addShader() doesn't let us pNext-chain mapping
    // info, so we build VkPipelineShaderStageCreateInfo ourselves and hand
    // them to the creator through pipelineInfo.pStages.
#if USE_SLANG
    VkShaderModuleCreateInfo vertModule{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, &mappingInfo, 0,
                                        sizeof(image_ktx_slang), image_ktx_slang};
    VkShaderModuleCreateInfo fragModule{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, &mappingInfo, 0,
                                        sizeof(image_ktx_slang), image_ktx_slang};
    const char*              vertEntry = "vertexMain";
    const char*              fragEntry = "fragmentMain";
#else
    VkShaderModuleCreateInfo vertModule{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, &mappingInfo, 0,
                                        sizeof(image_ktx_vert_glsl), image_ktx_vert_glsl};
    VkShaderModuleCreateInfo fragModule{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, &mappingInfo, 0,
                                        sizeof(image_ktx_frag_glsl), image_ktx_frag_glsl};
    const char*              vertEntry = "main";
    const char*              fragEntry = "main";
#endif

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].pNext = &vertModule;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].pName = vertEntry;
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].pNext = &fragModule;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].pName = fragEntry;

    nvvk::GraphicsPipelineCreator heapCreator;
    heapCreator.pipelineInfo.layout     = VK_NULL_HANDLE;  // required by VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT
    heapCreator.pipelineInfo.pStages    = stages.data();
    heapCreator.pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    heapCreator.colorFormats            = {m_srgbFormat};
    heapCreator.renderingState.depthAttachmentFormat = m_depthFormat;
    heapCreator.flags2                               = VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT;

    NVVK_CHECK(heapCreator.createGraphicsPipeline(m_device, nullptr, graphicState, &m_heapPipeline));
    NVVK_DBG_NAME(m_heapPipeline);
  }

  void createVkBuffers()
  {
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);

    m_meshVk.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_meshVk[i];
      m_alloc.createBuffer(m.vertices, std::span(m_meshes[i].vertices).size_bytes(), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT);
      m_alloc.createBuffer(m.indices, std::span(m_meshes[i].triangles).size_bytes(), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT);
      uploader.appendBuffer(m.vertices, 0, std::span(m_meshes[i].vertices));
      uploader.appendBuffer(m.indices, 0, std::span(m_meshes[i].triangles));
      NVVK_DBG_NAME(m.vertices.buffer);
      NVVK_DBG_NAME(m.indices.buffer);
    }

    m_alloc.createBuffer(m_frameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT);
    NVVK_DBG_NAME(m_frameInfo.buffer);
    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();
  }


  void destroyResources()
  {
    vkDestroyPipeline(m_device, m_heapPipeline, nullptr);

    m_texture.reset();

    for(PrimitiveMeshVk& m : m_meshVk)
    {
      m_alloc.destroyBuffer(m.vertices);
      m_alloc.destroyBuffer(m.indices);
    }
    m_alloc.destroyBuffer(m_frameInfo);

    if(m_heapSamplerIdx != ~0u)
    {
      m_heap.releaseSamplerDescriptor(m_heapSamplerIdx);
      m_heapSamplerIdx = ~0u;
    }
    m_heap.deinit();
    m_alloc.destroyBuffer(m_samplerHeapBuffer);
    m_alloc.destroyBuffer(m_resourceHeapBuffer);

    m_gBuffers.deinit();
    m_tonemapper.deinit();
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
  nvapp::Application*      m_app{nullptr};
  nvvk::ResourceAllocator  m_alloc;
  nvvk::GBuffer            m_gBuffers;  // G-Buffers: color + depth
  nvvk::SamplerPool        m_samplerPool;
  nvshaders::Tonemapper    m_tonemapper{};
  shaderio::TonemapperData m_tonemapperData;

  VkFormat          m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image (tonemapper)
  VkFormat          m_srgbFormat  = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the image (rendering)
  VkFormat          m_depthFormat = VK_FORMAT_UNDEFINED;            // Depth format of the depth buffer
  VkClearColorValue m_clearColor  = {};                             // Clear color
  VkDevice          m_device      = VK_NULL_HANDLE;                 // Convenient

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
    glm::vec4 color{1.F};
  };
  std::vector<nvutils::PrimitiveMesh> m_meshes;
  std::vector<nvutils::Node>          m_nodes;
  std::vector<Material>               m_materials;
  std::shared_ptr<TextureKtx>         m_texture;


  // Pipeline
  shaderio::PushConstant m_pushConst{};  // Information sent to the shader

  // Descriptor-heap pipeline. Uses the same SPIR-V as the classic path, with set/binding declarations remapped to heap offsets via
  // VkShaderDescriptorSetAndBindingMappingInfoEXT at pipeline creation.
  VkPipeline m_heapPipeline = VK_NULL_HANDLE;

  nvvk::DescriptorHeap m_heap{};
  nvvk::Buffer         m_samplerHeapBuffer{};
  nvvk::Buffer         m_resourceHeapBuffer{};
  uint32_t             m_heapSamplerIdx{~0u};

  int m_frame{0};
};


//////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;  // Base application information

  // Command line parsing
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  // Setting up what's needed for the Vulkan context creation
  nvvk::ContextInitInfo vkSetup;
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  VkPhysicalDeviceDescriptorHeapFeaturesEXT heapFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT};
  heapFeatures.descriptorHeap = VK_TRUE;
  VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untypedPtrFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR};

  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures});
  vkSetup.deviceExtensions.push_back({VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, &heapFeatures});
  vkSetup.deviceExtensions.push_back({VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, &untypedPtrFeatures});

  nvvk::ValidationSettings validation{};
  vkSetup.instanceCreateInfoExt = validation.buildPNextChain();

  // Creation of the Vulkan context
  nvvk::Context vkContext;  // Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // How we want the application
  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);


  // Add all application elements
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  elemCamera->setCameraManipulator(g_cameraManip);
  app.addElement(elemCamera);

  app.addElement(std::make_shared<ImageKtx>());

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
