/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */


/*

  Vulkan Example - Using the Memory Budget Extension
 
  This sample shows how to use the VK_EXT_memory_budget extension to get information about the
  memory budget of the GPU and how to use this information to manage the creation and deletion of
  objects in the scene.
 
  This sample creates a Menger Sponge, which is a fractal object, and creates it in parallel
  until the memory budget is exceeded. When the budget is exceeded, the sample deletes the objects
  that are not visible to the camera, and then creates new objects until the budget is exceeded
  again.

*/

#include <thread>

#define VMA_MEMORY_BUDGET 1
#define VMA_IMPLEMENTATION
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_nvml.hpp"
#include "nvvkhl/gbuffer.hpp"

#include "common/alloc_dma.hpp"


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
#include "_autogen/raster.frag.h"
#include "_autogen/raster.vert.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert), std::end(raster_vert)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag), std::end(raster_frag)};
#endif  // USE_HLSL

std::shared_ptr<nvvkhl::ElementBenchmarkParameters> g_benchmark;

static uint32_t SIZE_OF_MESH = 142'000'000;  // Approximation of the memory used by the average Menger Sponge

//--------------------------------------------------------------------------------------------------
// Convert a value to a metric formatted string (e.g. 1000 -> "1k")
//
template <typename T>
inline std::string metricFormatter(T value, const char* unit = "")
{
  static double      s_value[]  = {1000000000, 1000000, 1000, 1, 0.001, 0.000001, 0.000000001};
  static const char* s_prefix[] = {"G", "M", "k", "", "m", "u", "n"};

  if(value == 0)
    return fmt::format("0 {}", unit);

  for(int i = 0; i < 7; ++i)
  {
    if(std::fabs(value) >= s_value[i])
      return fmt::format("{:6.4g}{}{}", value / s_value[i], s_prefix[i], unit);
  }

  return fmt::format("{:6.4g}{}{}", value / s_value[6], s_prefix[6], unit);
}

// Fill an array with the nodes to create or delete
std::vector<int> fillArray(int start, int numElements, const std::vector<nvh::Node>& nodes, bool forward = true, bool add = true)
{
  std::vector<int> result(numElements);
  int              sizeArray = static_cast<int>(nodes.size());
  int              increment = forward ? 1 : -1;
  int              elemID    = 0;
  for(int i = 0; i < sizeArray; ++i)
  {
    int nodeID = (start + (i * increment)) % sizeArray;
    if(nodeID < 0)  // If nodeID is negative, add sizeArray to make it positive
      nodeID += sizeArray;
    if(add && nodes[nodeID].mesh == 0)
      result[elemID++] = nodeID;
    else if(!add && nodes[nodeID].mesh != 0)
      result[elemID++] = nodeID;

    if(elemID == numElements)  // If numElements are filled, exit the loop
      break;
  }
  return result;
}

template <typename T>
inline size_t getBaseTypeSize(const std::vector<T>& vec)
{
  using BaseType = typename std::remove_reference<T>::type;
  return sizeof(BaseType);
}

//--------------------------------------------------------------------------------------------------
// This class will create a Menger Sponge using a thread and create/delete meshes to stay within
// the memory budget
//
class MemoryBudget : public nvvkhl::IAppElement
{
  // Application settings
  struct Settings
  {
    int   statNumTriangles = 0;  // Number of triangles displayed
    int   statNumMesh      = 0;  // Number of meshes displayed
    float progress         = 0;  // Progress of the creation
    int   numMeshes        = 0;  // Number of created meshes
    int   firstIn          = 0;
    int   firstOut         = 0;
    bool  stopCreating     = false;
    float budgetPercentage = 0.8F;  // 80% of the budget
  };


public:
  MemoryBudget() { m_pushConst.pointSize = 1.0F; }
  ~MemoryBudget() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    m_dutil  = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility

    // #MEMORY_BUDGET
    VmaAllocatorCreateInfo allocator_info = {
        .flags                       = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT,
        .physicalDevice              = m_app->getContext()->m_physicalDevice,
        .device                      = m_app->getContext()->m_device,
        .preferredLargeHeapBlockSize = (128ull * 1024 * 1024),
        .instance                    = m_app->getContext()->m_instance,
        .vulkanApiVersion            = VK_API_VERSION_1_3,
    };
    //m_alloc = std::make_unique<nvvkhl::AllocVma>(allocator_info);  // Allocator
    m_alloc = std::make_unique<AllocDma>(m_app->getContext().get());  // Allocator
    m_dset  = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    createScene();
    createFrameInfoBuffer();
    createDescriptorSetPipeline();
    createShaderObjects();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override { createGbuffers({width, height}); }

  void onUIRender() override
  {
    if(!m_gBuffers)
      return;

    static bool  showApply      = false;
    static int   numMeshes      = 0;
    static float showOverBudget = 0;

    // VMA, but Vulkan method to get the budgets below
    // static uint64_t currentFrame = 0;
    // vmaSetCurrentFrameIndex(m_alloc->vma(), currentFrame);
    // VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    // vmaGetHeapBudgets(m_alloc->vma(), budgets);

    // Check if we are over budget, which is 80% of the budget by default
    // If it is the case, we will stop creating new objects and start deleting some
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = getMemoryBudget();

    double overbudgetUsage =
        (static_cast<double>(budget.heapUsage[0]) - (static_cast<double>(budget.heapBudget[0]) * m_settings.budgetPercentage));

    // #MEMORY_BUDGET
    if(m_settings.numMeshes != 0 && overbudgetUsage > 0)
    {
      m_settings.stopCreating = true;
      vkDeviceWaitIdle(m_device);
      int numObj = static_cast<int>(std::max(overbudgetUsage / double(SIZE_OF_MESH), 1.));
      LOGI(fmt::format("Over budget: {}\n", overbudgetUsage).c_str());
      std::vector<int> toDelete = fillArray(m_settings.firstOut, numObj, m_nodes, true, false);
      deleteMeshes(toDelete);
      numMeshes           = m_settings.numMeshes;
      m_settings.firstOut = (m_settings.firstOut + numObj) % static_cast<int>(m_nodes.size());
      showOverBudget      = 5;  // Display the message for 5 seconds
    }

    {  // Setting menu
      using PE = ImGuiH::PropertyEditor;

      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();

      bool showProgress = m_settings.progress > 0.0F && m_settings.progress < 0.99F;

      ImGui::Text("Settings");
      ImGui::BeginDisabled(showProgress);
      {
        PE::begin();
        PE::entry(
            "Budget Threshold",
            [&]() { return ImGui::SliderFloat("##T", &m_settings.budgetPercentage, 0.1f, 2, "%.3f"); },
            "Percentage of usage over budget.");

        if(!showApply)
          numMeshes = m_settings.numMeshes;

        if(PE::entry(
               "Number of elements",
               [&]() {
                 return ImGui::SliderInt("##Num", &numMeshes, 0, static_cast<int>(m_nodes.size()), "%d", ImGuiSliderFlags_AlwaysClamp);
               },
               "Number of visible element"))
        {
          showApply = (numMeshes != m_settings.numMeshes);
        }
        PE::end();

        if(showApply)
        {
          if(ImGui::Button("Apply"))
          {
            showApply = false;
            if(numMeshes > m_settings.numMeshes)
            {
              std::thread([&]() {
                int              numObj   = numMeshes - m_settings.numMeshes;
                std::vector<int> toCreate = fillArray(m_settings.firstIn, numObj, m_nodes, true, true);
                m_settings.stopCreating   = false;
                createMeshes(toCreate);  // Adding from another thread all objects
                numMeshes = m_settings.numMeshes;
              }).detach();
            }
            else
            {
              vkDeviceWaitIdle(m_device);
              std::vector<int> toDelete = fillArray(m_settings.firstIn, m_settings.numMeshes - numMeshes, m_nodes, false, false);
              deleteMeshes(toDelete);  // Adding from another thread all objects
              numMeshes = m_settings.numMeshes;
            }
          }
        }
      }
      ImGui::EndDisabled();

      ImGui::Text("Memory Budget");
      if(ImGui::BeginTable("Memory_Budget_usage", 3, ImGuiTableFlags_Borders))
      {
        ImGui::TableSetupColumn("---", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Device", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("System", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Budget");
        for(int i = 0; i < 2; i++)
        {
          ImGui::TableNextColumn();
          ImGui::Text("%s", metricFormatter(budget.heapBudget[i], "B").c_str());
        }
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("Usage");
        for(int i = 0; i < 2; i++)
        {
          ImGui::TableNextColumn();
          ImGui::Text("%s", metricFormatter(budget.heapUsage[i], "B").c_str());
        }
        ImGui::EndTable();
      }


      ImGui::Separator();
      ImGui::TextDisabled("Triangles: %s", metricFormatter(m_settings.statNumTriangles).c_str());
      // ImGui::TextDisabled("Meshes: %d", m_settings.statNumMesh);

      if(showProgress)
      {
        ImGui::Separator();
        ImGui::ProgressBar(m_settings.progress);
      }

      if(showOverBudget > 0)
      {
        ImGui::Separator();
        showOverBudget -= ImGui::GetIO().DeltaTime;
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "OVER BUDGET: Reducing Number of elements");
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
  }

  void onRender(VkCommandBuffer cmd) override
  {
    if(!m_gBuffers)
      return;

    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);
    const glm::vec2&                      clip = CameraManip.getClipPlanes();

    // Update Frame buffer uniform buffer
    DH::FrameInfo finfo{};
    finfo.view = CameraManip.getMatrix();
    finfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), m_gBuffers->getAspectRatio(), clip.x, clip.y);
    finfo.proj[1][1] *= -1;
    finfo.camPos = CameraManip.getEye();
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);

    // Drawing the primitives in G-Buffer 0
    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView(0)},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    renderAllMeshes(cmd, r_info);
  }


private:
  // #MEMORY_BUDGET
  VkPhysicalDeviceMemoryBudgetPropertiesEXT getMemoryBudget()
  {
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2         memProp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    memProp.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(m_app->getPhysicalDevice(), &memProp);

    return budget;
  }

  template <typename T>
  glm::vec3 integerToColor(T i)
  {
    const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
    return static_cast<glm::vec3>(sin(freq) * 0.5F + 0.5F);
  }

  void createScene()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    m_nodes = nvh::mengerSpongeNodes(2, 0.7F, 4080);
    m_meshes.resize(m_nodes.size() + 1);
    m_meshVk.resize(m_nodes.size() + 1);
    m_materials.resize(m_nodes.size() + 1);
    m_meshes[0]    = nvh::createCube();
    m_materials[0] = {.7f, .7f, .7f};

    {  // Create the buffer for the cube
      VkCommandBuffer  cmd = m_app->createTempCmdBuffer();
      PrimitiveMeshVk& m   = m_meshVk[0];
      m.vertices           = m_alloc->createBuffer(cmd, m_meshes[0].vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m.indices            = m_alloc->createBuffer(cmd, m_meshes[0].triangles, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_dutil->DBG_NAME(m.vertices.buffer);
      m_dutil->DBG_NAME(m.indices.buffer);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      m_alloc->finalizeAndReleaseStaging();
    }

    CameraManip.setClipPlanes({0.01F, 100.0F});  // Default camera
    CameraManip.setLookat({-1.24282, 0.28388, 1.24613}, {-0.07462, -0.08036, -0.02502}, {0.00000, 1.00000, 0.00000});

    // Find a good number of objects to create
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = getMemoryBudget();
    int numObj = static_cast<int>(budget.heapBudget[0] / SIZE_OF_MESH);  // Number of possible objects
    numObj     = (numObj * 65) / 100;                                    // 65% of the budget
    if(g_benchmark->config().testEnabled)
      numObj = 2;  // Only 2 objects for when test are enabled
    numObj = std::min(numObj, static_cast<int>(m_nodes.size()));

    // Create in another thread, all meshes
    std::thread([&, numObj]() {
      std::vector<int> toCreate = fillArray(m_settings.firstIn, numObj, m_nodes, true, true);
      createMeshes(toCreate);  // Adding from another thread all objects
    }).detach();
  }

  // Create the meshes in parallel
  void createMeshes(const std::vector<int>& toCreate)
  {
    LOGI("Creating %d meshes\n", static_cast<int>(toCreate.size()));
    nvvk::CommandPool cmd_pool(m_device, m_app->getQueueT().familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                               m_app->getQueueT().queue);

    float progressInc   = 1.0F / static_cast<float>(toCreate.size());
    m_settings.progress = 0.0F;

    for(int nodeID : toCreate)
    {
      if(m_settings.stopCreating)
        break;
      ++m_settings.numMeshes;
      m_settings.firstIn = (m_settings.firstIn + 1) % m_nodes.size();

      int meshID     = nodeID + 1;
      int materialID = nodeID + 1;
      m_settings.progress += progressInc;

      {  // Create the mesh & material
        m_meshes[meshID]        = (createMengerSpongeMesh(0.7F, nodeID));
        m_materials[materialID] = integerToColor(nodeID);
      }

      {  // Create the buffer for this mesh
        VkCommandBuffer cmdBuffer = cmd_pool.createCommandBuffer();

        PrimitiveMeshVk& meshVk = m_meshVk[meshID];
        meshVk.vertices = m_alloc->createBuffer(cmdBuffer, m_meshes[meshID].vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        meshVk.indices  = m_alloc->createBuffer(cmdBuffer, m_meshes[meshID].triangles, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        m_dutil->DBG_NAME_IDX(meshVk.vertices.buffer, meshID);
        m_dutil->DBG_NAME_IDX(meshVk.indices.buffer, meshID);
        cmd_pool.submitAndWait(cmdBuffer);
      }

      // Set the node to the new mesh
      m_nodes[nodeID].mesh     = meshID;
      m_nodes[nodeID].material = materialID;
    }
    m_alloc->finalizeAndReleaseStaging();  // The create buffer uses staging and at this point, we know the buffers are uploaded
    m_settings.progress = 1.0F;
  }

  // Create the meshes in parallel
  void deleteMeshes(std::vector<int>& toDestroy)
  {
    LOGI("Deleting %d meshes\n", static_cast<int>(toDestroy.size()));

    for(int nodeID : toDestroy)
    {
      --m_settings.numMeshes;

      // Reset the node
      m_nodes[nodeID].mesh     = 0;
      m_nodes[nodeID].material = 0;

      m_alloc->destroy(m_meshVk[nodeID + 1].indices);
      m_alloc->destroy(m_meshVk[nodeID + 1].vertices);
    }
  }

  // Create the Menger Sponge mesh
  nvh::PrimitiveMesh createMengerSpongeMesh(float probability, int seed)
  {
    nvh::PrimitiveMesh     cube         = nvh::createCube();
    std::vector<nvh::Node> mengerNodes  = nvh::mengerSpongeNodes(MENGER_SUBDIV, probability, seed);
    nvh::PrimitiveMesh     mengerSponge = nvh::mergeNodes(mengerNodes, {cube});
    if(mengerSponge.triangles.empty())  // Don't allow empty result
      mengerSponge = cube;
    return mengerSponge;
  }

  void renderAllMeshes(VkCommandBuffer cmd, const VkRenderingInfoKHR& renderingInfo)
  {
    vkCmdBeginRendering(cmd, &renderingInfo);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(), 0, 1, m_dset->getSets(), 0, nullptr);

    // #SHADER_OBJECT
    m_shaderObjPipeline.setViewportScissor(m_app->getViewportSize());
    m_shaderObjPipeline.cmdSetPipelineState(cmd);

    const VkShaderStageFlagBits stages[2]       = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
    const VkShaderStageFlagBits unusedStages[3] = {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                                   VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT};
    // Bind linked shaders
    vkCmdBindShadersEXT(cmd, 2, stages, m_shaders.data());
    vkCmdBindShadersEXT(cmd, 3, unusedStages, NULL);

    m_settings.statNumTriangles = 0;
    m_settings.statNumMesh      = 0;
    const VkDeviceSize offsets  = 0;
    bool               fillMode = true;
    for(const nvh::Node& n : m_nodes)
    {
      auto num_indices = static_cast<uint32_t>(m_meshes[n.mesh].triangles.size() * 3);
      m_settings.statNumTriangles += static_cast<int>(m_meshes[n.mesh].triangles.size());
      m_settings.statNumMesh += n.mesh > 0 ? 1 : 0;

      if(n.mesh == 0)
      {
        if(fillMode)
          vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_LINE);
        fillMode = false;
      }
      else
      {
        if(!fillMode)
          vkCmdSetPolygonModeEXT(cmd, VK_POLYGON_MODE_FILL);
        fillMode = true;
      }

      // Push constant information
      m_pushConst.transfo = n.localMatrix();
      m_pushConst.color   = m_materials[n.material];
      vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(DH::PushConstant), &m_pushConst);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m_meshVk[n.mesh].vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m_meshVk[n.mesh].indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cmd, num_indices, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
  }


  VkPushConstantRange getPushConstantRange()
  {
    return VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               .offset     = 0,
                               .size       = sizeof(DH::PushConstant)};
  }

  //-------------------------------------------------------------------------------------------------
  // Descriptor Set contains only the access to the Frame Buffer
  //
  void createDescriptorSetPipeline()
  {
    m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    m_dset->initLayout();
    m_dset->initPool(1);

    VkPushConstantRange push_constant_ranges = getPushConstantRange();
    m_dset->initPipeLayout(1, &push_constant_ranges);

    // Writing to descriptors
    const VkDescriptorBufferInfo      dbi_unif{m_frameInfo.buffer, 0, VK_WHOLE_SIZE};
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_dset->makeWrite(0, 0, &dbi_unif));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Creation of the dynamic graphic pipeline
    m_shaderObjPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    m_shaderObjPipeline.addBindingDescriptions({{0, sizeof(nvh::PrimitiveVertex)}});
    m_shaderObjPipeline.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, p))},  // Position
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, n))},  // Normal
    });
    m_shaderObjPipeline.update();
  }

  //-------------------------------------------------------------------------------------------------
  // Creating all Shader Objects
  // #SHADER_OBJECT
  void createShaderObjects()
  {
    VkPushConstantRange push_constant_ranges = getPushConstantRange();

    // Vertex
    std::vector<VkShaderCreateInfoEXT> shaderCreateInfos;
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT {
      .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .pNext = NULL, .flags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
      .stage = VK_SHADER_STAGE_VERTEX_BIT, .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT, .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
#if(USE_SLANG)
      .codeSize = sizeof(rasterSlang), .pCode = &rasterSlang[0], .pName = "vertexMain",
#else
        .codeSize = vert_shd.size() * getBaseTypeSize(vert_shd),
        .pCode    = vert_shd.data(),
        .pName    = USE_HLSL ? "vertexMain" : "main",
#endif  // USE_SLANG
      .setLayoutCount             = 1,
      .pSetLayouts                = &m_dset->getLayout(),  // Descriptor set layout compatible with the shaders
          .pushConstantRangeCount = 1, .pPushConstantRanges = &push_constant_ranges, .pSpecializationInfo = NULL,
    });

    // Fragment
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT {
      .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .pNext = NULL, .flags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .nextStage = 0, .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
#if(USE_SLANG)
      .codeSize = sizeof(rasterSlang), .pCode = &rasterSlang[0], .pName = "fragmentMain",
#else
        .codeSize = frag_shd.size() * getBaseTypeSize(frag_shd),
        .pCode    = frag_shd.data(),
        .pName    = USE_HLSL ? "fragmentMain" : "main",
#endif  // USE_SLANG
      .setLayoutCount             = 1,
      .pSetLayouts                = &m_dset->getLayout(),  // Descriptor set layout compatible with the shaders
          .pushConstantRangeCount = 1, .pPushConstantRanges = &push_constant_ranges, .pSpecializationInfo = NULL,
    });

    // Create the shaders
    NVVK_CHECK(vkCreateShadersEXT(m_device, 2, shaderCreateInfos.data(), NULL, m_shaders.data()));
  }

  //-------------------------------------------------------------------------------------------------
  // G-Buffers, a color and a depth, which are used for rendering. The result color will be displayed
  // and an image filling the ImGui Viewport window.
  void createGbuffers(const VkExtent2D& size)
  {
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), size, m_colorFormat, m_depthFormat);
  }

  //-------------------------------------------------------------------------------------------------
  // Creating the Vulkan buffer that is holding the data for Frame information
  // The frame info contains the camera and other information changing at each frame.
  void createFrameInfoBuffer()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_frameInfo         = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_dutil->DBG_NAME(m_frameInfo.buffer);
  }

  void destroyResources()
  {
    for(auto shader : m_shaders)
      vkDestroyShaderEXT(m_device, shader, NULL);

    for(PrimitiveMeshVk& m : m_meshVk)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_meshVk.clear();
    m_alloc->destroy(m_frameInfo);

    m_dset->deinit();
    m_gBuffers.reset();
  }


  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*             m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil> m_dutil;
  //std::shared_ptr<nvvkhl::AllocVma> m_alloc;
  std::shared_ptr<AllocDma> m_alloc;

  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.7F, 0.7F, 0.7F, 1.0F}};     // Clear color
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;                            // Descriptor set
  nvvk::GraphicShaderObjectPipeline             m_shaderObjPipeline;               // Shader Object pipeline

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk> m_meshVk;
  nvvk::Buffer                 m_frameInfo;

  // Data and setting
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;
  std::vector<glm::vec3>          m_materials;

  // Pipeline
  DH::PushConstant           m_pushConst{};  // Information sent to the shader
  std::array<VkShaderEXT, 2> m_shaders{};

  Settings m_settings;
};


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name  = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  spec.vSync = true;
  spec.vkSetup.setVersion(1, 3);

  // #MEMORY_BUDGET
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  spec.vkSetup.addDeviceExtension(VK_EXT_SHADER_OBJECT_EXTENSION_NAME, false, &shaderObjFeature);
  spec.vkSetup.addDeviceExtension(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, false);
  spec.vkSetup.addInstanceExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, false);

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  if(shaderObjFeature.shaderObject == VK_FALSE)
  {
    nvprintf("ERROR: Shader Object is not supported");
    std::exit(1);
  }

  // Create the test framework
  g_benchmark = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(g_benchmark);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());
  app->addElement(std::make_shared<nvvkhl::ElementNvml>(true));
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<MemoryBudget>());

  app->run();
  app.reset();

  return g_benchmark->errorCode();
}
