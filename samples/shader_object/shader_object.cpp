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

 This sample creates a 3D cube and render using the builtin camera

*/
//////////////////////////////////////////////////////////////////////////

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION

#include <glm/glm.hpp>


namespace shaderio {
using namespace glm;
#include "shaders/shaderio.h"  // Shared between host and device
}  // namespace shaderio

#include "_autogen/shader_object.frag.glsl.h"
#include "_autogen/shader_object.slang.h"
#include "_autogen/shader_object.vert.glsl.h"


#include "common/utils.hpp"


#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/property_editor.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/primitives.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>

// The camera for the scene
std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};


//////////////////////////////////////////////////////////////////////////
/// </summary> Display an image on a quad.
class ShaderObject : public nvapp::IAppElement
{
  // Application settings
  struct Settings
  {
    float           lineWidth        = 1.0F;
    float           pointSize        = 1.0F;
    VkFrontFace     frontFace        = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkCompareOp     depthCompare     = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkBool32        depthTestEnable  = VK_TRUE;
    VkBool32        depthWriteEnable = VK_TRUE;
    VkPolygonMode   polygonMode      = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cullMode         = VK_CULL_MODE_BACK_BIT;
  };


public:
  ShaderObject()           = default;
  ~ShaderObject() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    m_alloc.init({
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });

    // The texture sampler to use
    m_samplerPool.init(m_device);
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // Initialization of the G-Buffers we want use
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    m_gBuffers.init({.allocator      = &m_alloc,
                     .colorFormats   = {VK_FORMAT_R8G8B8A8_UNORM},
                     .depthFormat    = m_depthFormat,
                     .imageSampler   = linearSampler,
                     .descriptorPool = m_app->getTextureDescriptorPool()});

    createScene();
    createSceneDataBuffers();
    createFrameInfoBuffer();
    createDescriptorSet();
    createShaderObjects();  // #SHADER_OBJECT
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { NVVK_CHECK(m_gBuffers.update(cmd, size)); }

  void onUIRender() override
  {
    {  // Setting menu
      namespace PE = nvgui::PropertyEditor;

      ImGui::Begin("Settings");
      nvgui::CameraWidget(g_cameraManip);

      ImGui::Text("Menger Sponge");
      {
        static int   seed        = 1;
        static float probability = 0.7F;
        bool         update      = false;
        PE::begin();
        update |= PE::SliderFloat("Probability", &probability, 0, 1);
        update |= PE::SliderInt("Seed", &seed, 1, 100);
        if(update)
        {
          probability = (probability < 0.025F ? -1 : probability);
          vkDeviceWaitIdle(m_device);
          destroyPrimitiveMeshResources();
          createMesh(probability, seed);
          createSceneDataBuffers();
        }
        PE::end();
      }

      ImGui::Text("Dynamic Pipeline");
      PE::begin();
      PE::SliderFloat("Point Size", &m_settings.pointSize, 1, 10);
      PE::SliderFloat("Line Width", &m_settings.lineWidth, 1, 10);
      PE::Checkbox("Depth Test", (bool*)&m_settings.depthTestEnable);
      PE::Checkbox("Depth Write", (bool*)&m_settings.depthWriteEnable);
      {
        const char* items[] = {"Counter Clockwise", "Clockwise"};
        PE::Combo("Front Face", (int*)&m_settings.frontFace, items, IM_ARRAYSIZE(items));
      }
      {
        const char* items[] = {"None", "Front", "Back", "Front&Back"};
        PE::Combo("Cull Mode", (int*)&m_settings.cullMode, items, IM_ARRAYSIZE(items));
      }
      {
        const char* items[] = {"Fill", "Line", "Point"};
        PE::Combo("Polygon Mode", (int*)&m_settings.polygonMode, items, IM_ARRAYSIZE(items));
      }
      {
        const char* items[] = {"Never", "Less", "Equal", "Less or Equal", "Greater", "Not Equal", "Greater or Equal",
                               "Always"};
        PE::Combo("Depth Compare", (int*)&m_settings.depthCompare, items, IM_ARRAYSIZE(items));
      }
      PE::end();

      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());

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

    // Barrier to make sure the information is transferred
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

    // Update the descriptor set
    updateDescriptorSet(cmd);

    // #SHADER_OBJECT
    m_graphicState.cmdSetViewportAndScissor(cmd, m_gBuffers.getSize());
    m_graphicState.rasterizationState.lineWidth       = m_settings.lineWidth;
    m_graphicState.rasterizationState.cullMode        = m_settings.cullMode;
    m_graphicState.rasterizationState.polygonMode     = m_settings.polygonMode;
    m_graphicState.rasterizationState.frontFace       = m_settings.frontFace;
    m_graphicState.depthStencilState.depthCompareOp   = m_settings.depthCompare;
    m_graphicState.depthStencilState.depthTestEnable  = m_settings.depthTestEnable;
    m_graphicState.depthStencilState.depthWriteEnable = m_settings.depthWriteEnable;
    m_graphicState.cmdApplyAllStates(cmd);

    // Bind linked shaders
    m_graphicState.cmdBindShaders(cmd, {.vertex = m_shaders[0], .fragment = m_shaders[1]});

    // Rendering to GBuffer: attachment information
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView();
    colorAttachment.clearValue                = {.color = m_clearColor};
    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView                 = m_gBuffers.getDepthImageView();
    depthAttachment.clearValue                = {{{1.0F, 0}}};
    VkRenderingInfo renderingInfo             = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea                  = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount        = 1;
    renderingInfo.pColorAttachments           = &colorAttachment;
    renderingInfo.pDepthAttachment            = &depthAttachment;

    // Drawing the primitives in G-Buffer 0
    vkCmdBeginRendering(cmd, &renderingInfo);

    const VkDeviceSize     offsets{0};
    shaderio::PushConstant pushConst{.pointSize = m_settings.pointSize};  // Information sent to the shader

    for(const nvutils::Node& node : m_nodes)
    {
      const PrimitiveMeshVk& mesh       = m_meshVk[node.mesh];
      auto                   numIndices = static_cast<uint32_t>(m_meshes[node.mesh].triangles.size() * 3);

      // Push constant information per node
      pushConst.transfo = node.localMatrix();
      pushConst.color   = m_materials[node.material].color;
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(shaderio::PushConstant), &pushConst);

      vkCmdBindVertexBuffers(cmd, 0, 1, &mesh.vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, mesh.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cmd, numIndices, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
  }

private:
  void createScene()
  {
    createMesh(0.7F, 4080);                             // Meshes
    m_materials.push_back({.color = {.88, .88, .88}});  // Material
    m_nodes.push_back({.mesh = 0});                     // Instances

    g_cameraManip->setClipPlanes({0.01F, 100.0F});  // Default camera
    g_cameraManip->setLookat({-1.24282, 0.28388, 1.24613}, {-0.07462, -0.08036, -0.02502}, {0.00000, 1.00000, 0.00000});
  }

  void createMesh(float probability, int seed)
  {
    m_meshes                                = {};
    nvutils::PrimitiveMesh     cube         = nvutils::createCube();
    std::vector<nvutils::Node> mengerNodes  = nvutils::mengerSpongeNodes(MENGER_SUBDIV, probability, seed);
    nvutils::PrimitiveMesh     mengerSponge = nvutils::mergeNodes(mengerNodes, {cube});
    if(mengerSponge.triangles.empty())  // Don't allow empty result
      mengerSponge = cube;
    m_meshes.push_back(mengerSponge);
  }


  VkPushConstantRange getPushConstantRange()
  {
    return VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               .offset     = 0,
                               .size       = sizeof(shaderio::PushConstant)};
  }

  //-------------------------------------------------------------------------------------------------
  // Descriptor Set contains only the access to the Frame Buffer
  //
  void createDescriptorSet()
  {
    m_bindings.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);

    NVVK_CHECK(m_bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                    &m_descriptorSetLayout));
    NVVK_DBG_NAME(m_descriptorSetLayout);

    const VkPushConstantRange pushConstantRanges = getPushConstantRange();
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorSetLayout}, {pushConstantRanges}));
    NVVK_DBG_NAME(m_pipelineLayout);

    // For the dynamic graphic pipeline
    m_graphicState.vertexBindings   = {{.sType   = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                                        .stride  = sizeof(nvutils::PrimitiveVertex),
                                        .divisor = 1}};
    m_graphicState.vertexAttributes = {{.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                        .location = 0,
                                        .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                        .offset   = offsetof(nvutils::PrimitiveVertex, pos)},
                                       {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                        .location = 1,
                                        .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                        .offset   = offsetof(nvutils::PrimitiveVertex, nrm)}};
  }

  // Writing to descriptors
  void updateDescriptorSet(VkCommandBuffer cmd)
  {
    nvvk::WriteSetContainer writes;
    writes.append(m_bindings.getWriteSet(0), m_frameInfo);
    vkCmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, writes.data());
  }

  //-------------------------------------------------------------------------------------------------
  // Creating all Shader Objects
  // #SHADER_OBJECT
  void createShaderObjects()
  {
    VkPushConstantRange pushConstantRanges = getPushConstantRange();

    // Vertex
    std::vector<VkShaderCreateInfoEXT> shaderCreateInfos;
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT{
        .sType     = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .pNext     = NULL,
        .flags     = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
        .stage     = VK_SHADER_STAGE_VERTEX_BIT,
        .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .codeType  = VK_SHADER_CODE_TYPE_SPIRV_EXT,
#if(USE_SLANG)
        .codeSize = shader_object_slang_sizeInBytes,
        .pCode    = shader_object_slang,
        .pName    = "vertexMain",
#else
        .codeSize = std::span(shader_object_vert_glsl).size_bytes(),
        .pCode    = shader_object_vert_glsl,
        .pName    = "main",
#endif  // USE_SLANG
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorSetLayout,  // Descriptor set layout compatible with the shaders
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRanges,
        .pSpecializationInfo    = NULL,
    });

    // Fragment
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT{
        .sType     = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .pNext     = NULL,
        .flags     = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
        .stage     = VK_SHADER_STAGE_FRAGMENT_BIT,
        .nextStage = 0,
        .codeType  = VK_SHADER_CODE_TYPE_SPIRV_EXT,
#if(USE_SLANG)
        .codeSize = shader_object_slang_sizeInBytes,
        .pCode    = shader_object_slang,
        .pName    = "fragmentMain",
#else
        .codeSize = std::span(shader_object_frag_glsl).size_bytes(),
        .pCode    = shader_object_frag_glsl,
        .pName    = "main",
#endif  // USE_SLANG
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorSetLayout,  // Descriptor set layout compatible with the shaders
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRanges,
        .pSpecializationInfo    = NULL,
    });

    // Create the shaders
    NVVK_CHECK(vkCreateShadersEXT(m_device, 2, shaderCreateInfos.data(), NULL, m_shaders.data()));
  }

  //-------------------------------------------------------------------------------------------------
  // Creating the Vulkan buffers that are holding the data for:
  // - vertices and indices, one of each for each object
  //
  void createSceneDataBuffers()
  {
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);

    m_meshVk.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& mesh = m_meshVk[i];
      NVVK_CHECK(m_alloc.createBuffer(mesh.vertices, std::span(m_meshes[i].vertices).size_bytes(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
      NVVK_CHECK(m_alloc.createBuffer(mesh.indices, std::span(m_meshes[i].triangles).size_bytes(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(mesh.vertices, 0, std::span(m_meshes[i].vertices)));
      NVVK_CHECK(uploader.appendBuffer(mesh.indices, 0, std::span(m_meshes[i].triangles)));
      NVVK_DBG_NAME(mesh.vertices.buffer);
      NVVK_DBG_NAME(mesh.indices.buffer);
    }
    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();
  }

  //-------------------------------------------------------------------------------------------------
  // Creating the Vulkan buffer that is holding the data for Frame information
  // The frame info contains the camera and other information changing at each frame.
  void createFrameInfoBuffer()
  {
    NVVK_CHECK(m_alloc.createBuffer(m_frameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    NVVK_DBG_NAME(m_frameInfo.buffer);
  }

  void destroyPrimitiveMeshResources()
  {
    for(PrimitiveMeshVk& m : m_meshVk)
    {
      m_alloc.destroyBuffer(m.vertices);
      m_alloc.destroyBuffer(m.indices);
    }
    m_meshVk.clear();
  }

  void destroyResources()
  {
    // #SHADER_OBJECT
    for(auto shader : m_shaders)
      vkDestroyShaderEXT(m_device, shader, NULL);

    destroyPrimitiveMeshResources();

    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);

    m_alloc.destroyBuffer(m_frameInfo);

    m_samplerPool.deinit();
    m_gBuffers.deinit();
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
  nvapp::Application*     m_app{};
  nvvk::ResourceAllocator m_alloc;
  nvvk::GBuffer           m_gBuffers;       // G-Buffers: color + depth
  nvvk::SamplerPool       m_samplerPool{};  // The sampler pool, used to create a sampler for the texture

  VkFormat          m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;    // Color format of the image
  VkFormat          m_depthFormat = VK_FORMAT_UNDEFINED;         // Depth format of the depth buffer
  VkClearColorValue m_clearColor  = {{0.7F, 0.7F, 0.7F, 1.0F}};  // Clear color
  VkDevice          m_device      = VK_NULL_HANDLE;              // Convenient

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
    glm::vec3 color{1.F};
  };
  std::vector<nvutils::PrimitiveMesh> m_meshes;
  std::vector<nvutils::Node>          m_nodes;
  std::vector<Material>               m_materials;

  // Pipeline
  std::array<VkShaderEXT, 2>  m_shaders{};
  nvvk::GraphicsPipelineState m_graphicState;
  nvvk::DescriptorBindings    m_bindings;
  VkPipelineLayout            m_pipelineLayout{};       // The description of the pipeline
  VkDescriptorSetLayout       m_descriptorSetLayout{};  // Descriptor set layout

  Settings m_settings;
};


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvapp::Application           application;  // The application
  nvapp::ApplicationCreateInfo appInfo;      // Information to create the application
  nvvk::Context                vkContext;    // The Vulkan context
  nvvk::ContextInitInfo        vkSetup;      // Information to create the Vulkan context

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);


  // #SHADER_OBJECT
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  // Context setup
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature});

  // Validation layers settings
  nvvk::ValidationSettings vvlInfo{};
  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();

  // Create the Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    std::exit(1);
  }

  if(shaderObjFeature.shaderObject == VK_FALSE)
  {
    LOGE("ERROR: Shader Object is not supported");
    std::exit(1);
  }

  // Application information
  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  application.init(appInfo);


  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(g_cameraManip);
  application.addElement(elemCamera);
  application.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  application.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  application.addElement(std::make_shared<ShaderObject>());

  application.run();
  application.deinit();
  vkContext.deinit();

  return 0;
}
