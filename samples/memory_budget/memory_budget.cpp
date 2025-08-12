/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION
#define VMA_MEMORY_BUDGET 1


#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include <glm/glm.hpp>

#include "shaders/shaderio.h"  // Shared between host and device


#include "_autogen/memory_budget.frag.glsl.h"
#include "_autogen/memory_budget.slang.h"
#include "_autogen/memory_budget.vert.glsl.h"

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgpu_monitor/elem_gpu_monitor.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/property_editor.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/primitives.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
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


static uint32_t SIZE_OF_MESH = 142'000'000;  // Approximation of the memory used by the average Menger Sponge

//--------------------------------------------------------------------------------------------------
// Converts numeric values to human-readable metric format (e.g., 1000 -> "1k")
// @param value - Numeric value to format
// @param unit - Optional unit string to append
// @return Formatted string with metric prefix
//--------------------------------------------------------------------------------------------------
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

//--------------------------------------------------------------------------------------------------
// Creates array of indices for mesh creation/deletion
// @param start - Starting index
// @param numElements - Number of elements to generate
// @param nodes - Reference to scene nodes
// @param forward - Direction of traversal
// @param add - True for creation, false for deletion
// @return Vector of indices
//--------------------------------------------------------------------------------------------------
static std::vector<int> fillArray(int start, int numElements, const std::vector<nvutils::Node>& nodes, bool forward = true, bool add = true)
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


//--------------------------------------------------------------------------------------------------
// MemoryBudget: Main class demonstrating Vulkan memory budget management
// - Creates and manages a Menger Sponge fractal scene
// - Monitors GPU memory usage using VK_EXT_memory_budget extension
// - Dynamically adds/removes scene objects to stay within memory budget
// - Uses shader objects for rendering
//--------------------------------------------------------------------------------------------------
class MemoryBudget : public nvapp::IAppElement
{
  // Application settings and statistics for memory budget management and scene state
  struct Settings
  {
    int   statNumTriangles = 0;      // Total number of triangles currently rendered
    int   statNumMesh      = 0;      // Total number of active meshes in the scene
    float progress         = 0;      // Progress indicator for mesh creation (0.0 to 1.0)
    int   numMeshes        = 0;      // Current count of created meshes
    int   firstIn          = 0;      // Index for next mesh to be created
    int   firstOut         = 0;      // Index for next mesh to be removed
    bool  stopCreating     = false;  // Flag to stop mesh creation thread
    float budgetPercentage = 0.8F;   // Target percentage of total GPU memory budget to use
  };


public:
  MemoryBudget() { m_pushConst.pointSize = 1.0F; }
  ~MemoryBudget() override = default;

  void setCamera(std::shared_ptr<nvutils::CameraManipulator> cameraManip) { m_cameraManip = cameraManip; }

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    // #MEMORY_BUDGET
    VmaAllocatorCreateInfo allocator_info = {
        .flags                       = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT,
        .physicalDevice              = m_app->getPhysicalDevice(),
        .device                      = m_app->getDevice(),
        .preferredLargeHeapBlockSize = (128ull * 1024 * 1024),
        .instance                    = m_app->getInstance(),
        .vulkanApiVersion            = VK_API_VERSION_1_4,
    };

    m_alloc.init(allocator_info);

    // Acquiring the sampler which will be used for displaying the GBuffer
    m_samplerPool.init(app->getDevice());
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // GBuffer
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    m_gBuffers.init({
        .allocator      = &m_alloc,
        .colorFormats   = {m_colorFormat},  // Only one GBuffer color attachment
        .depthFormat    = m_depthFormat,
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    });

    createScene();
    createFrameInfoBuffer();
    createDescriptorSetPipeline();
    createShaderObjects();
  }

  void onDetach() override
  {
    // Stop any running creation thread and wait for it to complete
    stopMeshCreationThread();

    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { NVVK_CHECK(m_gBuffers.update(cmd, size)); }

  void onUIRender() override
  {
    static bool  showApply      = false;
    static int   numMeshes      = 0;
    static float showOverBudget = 0;

    // VMA, but Vulkan method to get the budgets below
    static uint32_t currentFrame = 0;
    vmaSetCurrentFrameIndex(m_alloc, currentFrame);
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
    vmaGetHeapBudgets(m_alloc, budgets);

    // Check if we are over budget, which is 80% of the budget by default
    // If it is the case, we will stop creating new objects and start deleting some
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = getMemoryBudget();

    double overbudgetUsage =
        (static_cast<double>(budget.heapUsage[0]) - (static_cast<double>(budget.heapBudget[0]) * m_settings.budgetPercentage));

    // #MEMORY_BUDGET
    if(m_settings.numMeshes != 0 && overbudgetUsage > 0)
    {
      stopMeshCreationThread();
      vkDeviceWaitIdle(m_device);

      int numObj = static_cast<int>(std::max(overbudgetUsage / double(SIZE_OF_MESH), 1.));
      LOGI("Over budget: %f\n", overbudgetUsage);
      std::vector<int> toDelete = fillArray(m_settings.firstOut, numObj, m_nodes, true, false);
      deleteMeshes(toDelete);
      numMeshes           = m_settings.numMeshes;
      m_settings.firstOut = (m_settings.firstOut + numObj) % static_cast<int>(m_nodes.size());
      showOverBudget      = 5;  // Display the message for 5 seconds
    }

    {  // Setting menu
      namespace PE = nvgui::PropertyEditor;

      ImGui::Begin("Settings");
      nvgui::CameraWidget(m_cameraManip);

      bool showProgress = m_settings.progress > 0.0F && m_settings.progress < 0.99F;

      ImGui::Text("Settings");
      ImGui::BeginDisabled(showProgress);
      {
        PE::begin();
        PE::SliderFloat("Budget Threshold", &m_settings.budgetPercentage, 0.1f, 2, "%.3f", {}, "Percentage of usage over budget.");

        if(!showApply)
          numMeshes = m_settings.numMeshes;

        if(PE::SliderInt("Number of elements", &numMeshes, 0, static_cast<int>(m_nodes.size()), "%d",
                         ImGuiSliderFlags_AlwaysClamp, "Number of visible element"))
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
              int numObj = numMeshes - m_settings.numMeshes;
              startMeshCreationThread(numObj);
              numMeshes = m_settings.numMeshes;
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
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);

    // Update Frame buffer uniform buffer
    shaderio::FrameInfo finfo{};
    finfo.view   = m_cameraManip->getViewMatrix();
    finfo.proj   = m_cameraManip->getPerspectiveMatrix();
    finfo.camPos = m_cameraManip->getEye();
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);

    // Making sure the information is transfered
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);


    // Drawing the primitives in a G-Buffer
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView();
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


    renderAllMeshes(cmd, renderingInfo);
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
    nvutils::ScopedTimer st(__FUNCTION__);

    m_nodes = nvutils::mengerSpongeNodes(2, 0.7F, 4080);
    m_meshes.resize(m_nodes.size() + 1);
    m_meshVk.resize(m_nodes.size() + 1);
    m_materials.resize(m_nodes.size() + 1);
    m_meshes[0]    = nvutils::createCube();
    m_materials[0] = {.7f, .7f, .7f};

    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);

    {  // Create the buffer for the cube
      VkCommandBuffer  cmd = m_app->createTempCmdBuffer();
      PrimitiveMeshVk& m   = m_meshVk[0];
      NVVK_CHECK(m_alloc.createBuffer(m.vertices, std::span(m_meshes[0].vertices).size_bytes(), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
      NVVK_CHECK(m_alloc.createBuffer(m.indices, std::span(m_meshes[0].triangles).size_bytes(), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m.vertices, 0, std::span(m_meshes[0].vertices)));
      NVVK_CHECK(uploader.appendBuffer(m.indices, 0, std::span(m_meshes[0].triangles)));
      NVVK_DBG_NAME(m.vertices.buffer);
      NVVK_DBG_NAME(m.indices.buffer);
      uploader.cmdUploadAppended(cmd);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      uploader.deinit();
    }

    m_cameraManip->setClipPlanes({0.01F, 100.0F});  // Default camera
    m_cameraManip->setLookat({-1.24282, 0.28388, 1.24613}, {-0.07462, -0.08036, -0.02502}, {0.00000, 1.00000, 0.00000});

    // Find a good number of objects to create
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = getMemoryBudget();
    int numObj = static_cast<int>(budget.heapBudget[0] / SIZE_OF_MESH);  // Number of possible objects
    numObj     = (numObj * 65) / 100;                                    // 65% of the budget
    if(m_app->isHeadless())
      numObj = 2;  // Only 2 objects for when test are enabled
    numObj = std::min(numObj, static_cast<int>(m_nodes.size()));

    // Create in another thread, all meshes
    startMeshCreationThread(numObj);
  }

  //--------------------------------------------------------------------------------------------------
  // Creates scene objects in a separate thread to avoid blocking the main rendering thread
  // @param numObj - Number of new objects to create
  //--------------------------------------------------------------------------------------------------
  void startMeshCreationThread(int numObj)
  {
    std::unique_lock<std::mutex> lock(m_threadMutex);
    if(m_threadRunning)
    {
      // Wait for any existing thread to complete
      m_settings.stopCreating = true;
      m_threadCV.wait(lock, [this] { return !m_threadRunning; });
    }

    m_settings.stopCreating = false;
    m_threadRunning         = true;

    std::thread([this, numObj]() {
      std::vector<int> toCreate = fillArray(m_settings.firstIn, numObj, m_nodes, true, true);
      createMeshes(toCreate);  // Adding from another thread all objects

      // Signal thread completion
      {
        std::lock_guard<std::mutex> lock(m_threadMutex);
        m_threadRunning = false;
        m_threadCV.notify_one();
      }
    }).detach();
  }

  void stopMeshCreationThread()
  {
    std::unique_lock<std::mutex> lock(m_threadMutex);
    m_settings.stopCreating = true;
    if(m_threadRunning)
    {
      m_threadCV.wait(lock, [this] { return !m_threadRunning; });
    }
  }

  //--------------------------------------------------------------------------------------------------
  // Creates multiple mesh instances in parallel, monitoring memory budget
  // @param toCreate - Vector of node indices for meshes to be created
  //--------------------------------------------------------------------------------------------------
  void createMeshes(const std::vector<int>& toCreate)
  {
    LOGI("Creating %d meshes\n", static_cast<int>(toCreate.size()));


    VkCommandPool                 transientCmdPool;
    const VkCommandPoolCreateInfo commandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,  // Hint that commands will be short-lived
        .queueFamilyIndex = m_app->getQueue(1).familyIndex,
    };
    NVVK_CHECK(vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &transientCmdPool));
    NVVK_DBG_NAME(transientCmdPool);


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
        VkCommandBuffer cmd;
        nvvk::beginSingleTimeCommands(cmd, m_device, transientCmdPool);

        nvvk::StagingUploader uploader;
        uploader.init(&m_alloc);


        PrimitiveMeshVk& meshVk = m_meshVk[meshID];
        NVVK_CHECK(m_alloc.createBuffer(meshVk.vertices, std::span(m_meshes[meshID].vertices).size_bytes(),
                                        VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
        NVVK_CHECK(m_alloc.createBuffer(meshVk.indices, std::span(m_meshes[meshID].triangles).size_bytes(),
                                        VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
        NVVK_CHECK(uploader.appendBuffer(meshVk.vertices, 0, std::span(m_meshes[meshID].vertices)));
        NVVK_CHECK(uploader.appendBuffer(meshVk.indices, 0, std::span(m_meshes[meshID].triangles)));
        NVVK_DBG_NAME(meshVk.vertices.buffer);
        NVVK_DBG_NAME(meshVk.indices.buffer);

        uploader.cmdUploadAppended(cmd);
        nvvk::endSingleTimeCommands(cmd, m_device, transientCmdPool, m_app->getQueue(1).queue);
        uploader.deinit();
      }

      // Set the node to the new mesh
      m_nodes[nodeID].mesh     = meshID;
      m_nodes[nodeID].material = materialID;
    }

    m_settings.progress = 1.0F;

    vkDestroyCommandPool(m_device, transientCmdPool, nullptr);
  }

  //--------------------------------------------------------------------------------------------------
  // Removes meshes from the scene when memory budget is exceeded
  // @param toDestroy - Vector of node indices for meshes to be deleted
  //--------------------------------------------------------------------------------------------------
  void deleteMeshes(std::vector<int>& toDestroy)
  {
    LOGI("Deleting %d meshes\n", static_cast<int>(toDestroy.size()));

    for(int nodeID : toDestroy)
    {
      --m_settings.numMeshes;

      // Reset the node
      m_nodes[nodeID].mesh     = 0;
      m_nodes[nodeID].material = 0;

      m_alloc.destroyBuffer(m_meshVk[nodeID + 1].indices);
      m_alloc.destroyBuffer(m_meshVk[nodeID + 1].vertices);
    }
  }

  // Create the Menger Sponge mesh
  nvutils::PrimitiveMesh createMengerSpongeMesh(float probability, int seed)
  {
    nvutils::PrimitiveMesh     cube         = nvutils::createCube();
    std::vector<nvutils::Node> mengerNodes  = nvutils::mengerSpongeNodes(MENGER_SUBDIV, probability, seed);
    nvutils::PrimitiveMesh     mengerSponge = nvutils::mergeNodes(mengerNodes, {cube});
    if(mengerSponge.triangles.empty())  // Don't allow empty result
      mengerSponge = cube;
    return mengerSponge;
  }

  //--------------------------------------------------------------------------------------------------
  // Renders all active meshes in the scene using shader objects
  // @param cmd - Command buffer for recording render commands
  // @param renderingInfo - Vulkan rendering information structure
  //--------------------------------------------------------------------------------------------------
  void renderAllMeshes(VkCommandBuffer cmd, const VkRenderingInfoKHR& renderingInfo)
  {
    vkCmdBeginRendering(cmd, &renderingInfo);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorPack.sets[0], 0, nullptr);

    // #SHADER_OBJECT
    m_graphicState.cmdSetViewportAndScissor(cmd, m_app->getViewportSize());
    m_graphicState.cmdApplyAllStates(cmd);
    vkCmdSetLineRasterizationModeEXT(cmd, VK_LINE_RASTERIZATION_MODE_BRESENHAM);

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
    for(const nvutils::Node& n : m_nodes)
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
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(shaderio::PushConstant), &m_pushConst);

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
                               .size       = sizeof(shaderio::PushConstant)};
  }

  //-------------------------------------------------------------------------------------------------
  // Descriptor Set contains only the access to the Frame Buffer
  //
  void createDescriptorSetPipeline()
  {
    m_descriptorPack.bindings.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    NVVK_CHECK(m_descriptorPack.initFromBindings(m_device, 1));
    NVVK_DBG_NAME(m_descriptorPack.layout);
    NVVK_DBG_NAME(m_descriptorPack.pool);
    NVVK_DBG_NAME(m_descriptorPack.sets[0]);

    const VkPushConstantRange push_constant_ranges = getPushConstantRange();
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorPack.layout}, {push_constant_ranges}));
    NVVK_DBG_NAME(m_pipelineLayout);


    // Writing to descriptors
    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descriptorPack.bindings.getWriteSet(0, m_descriptorPack.sets[0]), m_frameInfo);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

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
  }

  //-------------------------------------------------------------------------------------------------
  // Creating all Shader Objects
  // #SHADER_OBJECT
  void createShaderObjects()
  {
    VkPushConstantRange push_constant_ranges = getPushConstantRange();

    // Vertex
    std::vector<VkShaderCreateInfoEXT> shaderCreateInfos;
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT{
        .sType     = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .pNext     = NULL,
        .flags     = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
        .stage     = VK_SHADER_STAGE_VERTEX_BIT,
        .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .codeType  = VK_SHADER_CODE_TYPE_SPIRV_EXT,
#if (USE_SLANG)
        .codeSize = memory_budget_slang_sizeInBytes,
        .pCode    = memory_budget_slang,
        .pName    = "vertexMain",
#else
        .codeSize = std::span(memory_budget_vert_glsl).size_bytes(),
        .pCode    = std::span(memory_budget_vert_glsl).data(),
        .pName    = "main",
#endif  // USE_SLANG
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorPack.layout,  // Descriptor set layout compatible with the shaders
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant_ranges,
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
#if (USE_SLANG)
        .codeSize = memory_budget_slang_sizeInBytes,
        .pCode    = memory_budget_slang,
        .pName    = "fragmentMain",
#else
        .codeSize = std::span(memory_budget_frag_glsl).size_bytes(),
        .pCode    = std::span(memory_budget_frag_glsl).data(),
        .pName    = "main",
#endif  // USE_SLANG
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorPack.layout,  // Descriptor set layout compatible with the shaders
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant_ranges,
        .pSpecializationInfo    = NULL,
    });

    // Create the shaders
    NVVK_CHECK(vkCreateShadersEXT(m_device, 2, shaderCreateInfos.data(), NULL, m_shaders.data()));
  }

  //-------------------------------------------------------------------------------------------------
  // Creating the Vulkan buffer that is holding the data for Frame information
  // The frame info contains the camera and other information changing at each frame.
  void createFrameInfoBuffer()
  {
    NVVK_CHECK(m_alloc.createBuffer(m_frameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
    NVVK_DBG_NAME(m_frameInfo.buffer);
  }

  void destroyResources()
  {
    for(auto shader : m_shaders)
      vkDestroyShaderEXT(m_device, shader, NULL);

    for(PrimitiveMeshVk& m : m_meshVk)
    {
      m_alloc.destroyBuffer(m.vertices);
      m_alloc.destroyBuffer(m.indices);
    }
    m_meshVk.clear();
    m_alloc.destroyBuffer(m_frameInfo);


    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    m_descriptorPack.deinit();

    m_gBuffers.deinit();
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
  nvapp::Application* m_app{nullptr};

  VkFormat          m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat          m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue m_clearColor  = {{0.7F, 0.7F, 0.7F, 1.0F}};     // Clear color
  VkDevice          m_device      = VK_NULL_HANDLE;                 // Convenient

  // Pipeline
  VkPipelineLayout     m_pipelineLayout{};  // Pipeline layout
  nvvk::DescriptorPack m_descriptorPack{};  // Descriptor bindings, layout, pool, and set

  nvvk::GraphicsPipelineState m_graphicState;  // State of the graphic pipeline

  // Frame information
  nvvk::Buffer               m_frameInfo;
  shaderio::PushConstant     m_pushConst{};  // Information sent to the shader
  std::array<VkShaderEXT, 2> m_shaders{};

  nvvk::SamplerPool m_samplerPool{};  // The sampler pool, used to create a sampler for the texture

  // Sample settings
  Settings m_settings;

  // Synchronization primitives for mesh creation thread
  std::mutex              m_threadMutex;
  std::condition_variable m_threadCV;
  bool                    m_threadRunning = false;

  // Resource management
  nvvk::ResourceAllocator m_alloc;     // Vulkan memory allocator with budget tracking
  nvvk::GBuffer           m_gBuffers;  // G-Buffer for deferred rendering

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Vertex buffer containing position and normal data
    nvvk::Buffer indices;   // Index buffer for triangle indices
  };

  // Scene data
  std::vector<PrimitiveMeshVk>        m_meshVk;     // Vulkan buffers for each mesh
  std::vector<nvutils::PrimitiveMesh> m_meshes;     // CPU-side mesh data
  std::vector<nvutils::Node>          m_nodes;      // Scene graph nodes
  std::vector<glm::vec3>              m_materials;  // Material colors for each mesh

  std::shared_ptr<nvutils::CameraManipulator> m_cameraManip{};
};


//--------------------------------------------------------------------------------------------------
// Main application entry point
// - Sets up Vulkan context with memory budget and shader object extensions
// - Creates application window and UI elements
// - Initializes scene and starts render loop
//--------------------------------------------------------------------------------------------------
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;  // Base application information

  // Command line parsing
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  // Vulkan context creation
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  nvvk::ContextInitInfo vkSetup = {
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions =
          {
              {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature},
              {VK_EXT_MEMORY_BUDGET_EXTENSION_NAME},
          },
      .queues     = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_TRANSFER_BIT},
      .apiVersion = VK_API_VERSION_1_4,
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }


  // Creation of the Vulkan context
  nvvk::Context vkContext;  // Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Application setup information
  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  if(shaderObjFeature.shaderObject == VK_FALSE)
  {
    LOGE("ERROR: Shader Object is not supported");
    std::exit(1);
  }

  auto cameraManip        = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera         = std::make_shared<nvapp::ElementCamera>();
  auto memoryBudgetSample = std::make_shared<MemoryBudget>();
  elemCamera->setCameraManipulator(cameraManip);
  memoryBudgetSample->setCamera(cameraManip);

  // Add all application elements
  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<nvgpu_monitor::ElementGpuMonitor>(true));
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app.addElement(memoryBudgetSample);

  app.run();

  app.deinit();
  vkContext.deinit();

  return 0;
}
