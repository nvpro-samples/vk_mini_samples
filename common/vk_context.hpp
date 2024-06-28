#include <vulkan/vulkan_core.h>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <unordered_set>
#include "nvvk/error_vk.hpp"
#include "nvh/nvprint.hpp"

//--------------------------------------------------------------------------------------------------
// CATCHING VULKAN ERRORS
//--------------------------------------------------------------------------------------------------
static VKAPI_ATTR VkBool32 VKAPI_CALL VkContextDebugReport(VkDebugUtilsMessageSeverityFlagBitsEXT      messageSeverity,
                                                           VkDebugUtilsMessageTypeFlagsEXT             messageType,
                                                           const VkDebugUtilsMessengerCallbackDataEXT* callbackData,
                                                           void*                                       userData)
{
  if(messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
  {
    // To ignore specific message, insert it to settings.ignoreDbgMessages
    auto ignoredMsg = reinterpret_cast<std::unordered_set<uint32_t>*>(userData);
    if(ignoredMsg->find(callbackData->messageIdNumber) != ignoredMsg->end())
      return VK_FALSE;
    fprintf(stderr, "%s\n", callbackData->pMessage);
#if defined(_MSVC_LANG)
    __debugbreak();
#elif defined(LINUX)
    raise(SIGTRAP);
#endif
  }
  return VK_FALSE;
}


// Vulkan Queue information
struct QueueInfo
{
  uint32_t familyIndex;
  uint32_t queueIndex;
  VkQueue  queue;
};

// Struct to hold an extension and its corresponding feature
struct ExtensionFeaturePair
{
  const char* extensionName = nullptr;
  void*       feature       = nullptr;  // [optional] Pointer to the feature structure for the extension
};

// Specific to the creation of Vulkan context
struct VkContextSettings
{
  std::vector<const char*> instanceExtensions = {};  // Instance extensions: VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
  std::vector<ExtensionFeaturePair> deviceExtensions = {};  // Device extensions: {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}, {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature}, {OTHER}}
  std::vector<VkQueueFlags>    queues = {VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT,
                                         VK_QUEUE_COMPUTE_BIT, VK_QUEUE_TRANSFER_BIT};  // All desired queues, first is always GTC
  std::unordered_set<uint32_t> ignoreDbgMessages;   // Ignore debug messages: 0x901f59ec
  void*       instanceCreateInfoExt = nullptr;      // Instance create info extension (ex: VkLayerSettingsCreateInfoEXT)
  const char* applicationName       = "No Engine";  // Application name
  uint32_t    apiVersion            = VK_API_VERSION_1_3;  // Vulkan API version
  VkAllocationCallbacks* alloc      = nullptr;             // Allocation callbacks
  bool enableAllFeatures            = true;  // If true, pull all capability of `features` from the physical device
#if NDEBUG
  bool enableValidationLayers = false;  // Disable validation layers in release
#else
  bool enableValidationLayers = true;  // Enable validation layers
#endif
};

inline std::vector<VkExtensionProperties> getDeviceExtensions(VkPhysicalDevice physicalDevice)
{
  uint32_t                           count;
  std::vector<VkExtensionProperties> extensionProperties;
  NVVK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr));
  extensionProperties.resize(count);
  NVVK_CHECK(vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensionProperties.data()));
  extensionProperties.resize(std::min(extensionProperties.size(), size_t(count)));
  return extensionProperties;
}


//--------------------------------------------------------------------------------------------------
// Simple class to handle the Vulkan context creation
class VkContext
{
public:
  VkContext(const VkContextSettings& settings = VkContextSettings())
      : m_settings(settings)
  {
    createInstance();
    selectPhysicalDevice();
    createDevice();
  }
  ~VkContext() { cleanup(); }

  VkInstance             getInstance() const { return m_instance; }
  VkDevice               getDevice() const { return m_device; }
  VkPhysicalDevice       getPhysicalDevice() const { return m_physicalDevice; }
  QueueInfo              getQueueInfo(uint32_t index) const { return m_queueInfos[index]; }
  std::vector<QueueInfo> getQueueInfos() const { return m_queueInfos; }

private:
  VkContextSettings m_settings{};  // What was used to create the information

  VkInstance       m_instance{};
  VkDevice         m_device{};
  VkPhysicalDevice m_physicalDevice{};

  // For device creation
  VkPhysicalDeviceFeatures2        m_deviceFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDeviceVulkan11Features features11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
  VkPhysicalDeviceVulkan12Features features12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
  VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};

  // For Queue creation
  std::vector<VkQueueFlags>            m_desiredQueues{};
  std::vector<VkDeviceQueueCreateInfo> m_queueCreateInfos{};
  std::vector<QueueInfo>               m_queueInfos{};
  std::vector<std::vector<float>>      m_queuePriorities{};  // Store priorities here

  // Callback for debug messages
  VkDebugUtilsMessengerEXT m_dbgMessenger = VK_NULL_HANDLE;

  void createInstance()
  {
    VkApplicationInfo appInfo{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = m_settings.applicationName,
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName        = "My Engine",
        .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion         = m_settings.apiVersion,
    };

    std::vector<const char*> layers;
    if(m_settings.enableValidationLayers)
    {
      layers.push_back("VK_LAYER_KHRONOS_validation");
      m_settings.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkInstanceCreateInfo createInfo{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext                   = m_settings.instanceCreateInfoExt,
        .pApplicationInfo        = &appInfo,
        .enabledLayerCount       = uint32_t(layers.size()),
        .ppEnabledLayerNames     = layers.data(),
        .enabledExtensionCount   = uint32_t(m_settings.instanceExtensions.size()),
        .ppEnabledExtensionNames = m_settings.instanceExtensions.data(),
    };

    VkResult result = vkCreateInstance(&createInfo, m_settings.alloc, &m_instance);
    if(result != VK_SUCCESS)
      assert(!"failed to create instance!");

    if(m_settings.enableValidationLayers)
    {
      auto vkCreateDebugUtilsMessengerEXT =
          (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
      assert(vkCreateDebugUtilsMessengerEXT != nullptr);
      VkDebugUtilsMessengerCreateInfoEXT dbg_messenger_create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
      dbg_messenger_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT       // For debug printf
                                                  | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT  // GPU info, bug
                                                  | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;   // Invalid usage
      dbg_messenger_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT      // Violation of spec
                                              | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;  // Non-optimal use
      dbg_messenger_create_info.pfnUserCallback = VkContextDebugReport;
      dbg_messenger_create_info.pUserData       = &m_settings.ignoreDbgMessages;
      NVVK_CHECK(vkCreateDebugUtilsMessengerEXT(m_instance, &dbg_messenger_create_info, nullptr, &m_dbgMessenger));
    }
  }

  void selectPhysicalDevice()
  {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    if(deviceCount == 0)
      assert(!"Failed to find GPUs with Vulkan support!");
    std::vector<VkPhysicalDevice> gpus(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, gpus.data());

    // Find the discrete GPU if present, or use first one available.
    m_physicalDevice = gpus[0];
    for(VkPhysicalDevice& device : gpus)
    {
      VkPhysicalDeviceProperties properties;
      vkGetPhysicalDeviceProperties(device, &properties);
      if(properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        m_physicalDevice = device;
    }

    // Query the physical device features
    m_deviceFeatures.pNext = &features11;
    if(m_settings.apiVersion >= VK_API_VERSION_1_2)
      features11.pNext = &features12;
    if(m_settings.apiVersion >= VK_API_VERSION_1_3)
      features12.pNext = &features13;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &m_deviceFeatures);
  }

  void createDevice()
  {
    // Find the queues that we need
    m_desiredQueues = m_settings.queues;
    findQueueFamilies();

    // Filter the available extensions otherwise the device creation will fail
    m_settings.deviceExtensions = filterAvailableExtensions(getDeviceExtensions(m_physicalDevice), m_settings.deviceExtensions);

    // Chain all custom features to the pNext chain of m_deviceFeatures
    for(auto extension : m_settings.deviceExtensions)
    {
      if(extension.feature)
        prependFeatures(reinterpret_cast<VkBaseOutStructure*>(&m_deviceFeatures),
                        reinterpret_cast<VkBaseOutStructure*>(extension.feature));
    }
    // Activate features on request
    if(m_settings.enableAllFeatures)
      vkGetPhysicalDeviceFeatures2(m_physicalDevice, &m_deviceFeatures);

    // List of extensions to enable
    std::vector<const char*> enabledExtensions;
    for(const auto& ext : m_settings.deviceExtensions)
      enabledExtensions.push_back(ext.extensionName);
    VkDeviceCreateInfo createInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &m_deviceFeatures,
        .queueCreateInfoCount    = uint32_t(m_queueCreateInfos.size()),
        .pQueueCreateInfos       = m_queueCreateInfos.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(enabledExtensions.size()),
        .ppEnabledExtensionNames = enabledExtensions.data(),
    };

    if(vkCreateDevice(m_physicalDevice, &createInfo, m_settings.alloc, &m_device) != VK_SUCCESS)
      assert(!"failed to create logical device!");

    for(auto& queue : m_queueInfos)
      vkGetDeviceQueue(m_device, queue.familyIndex, queue.queueIndex, &queue.queue);
  }

  void prependFeatures(VkBaseOutStructure* baseStruct, VkBaseOutStructure* prependStruct)
  {
    // Traverse the pNext chain of prependStruct to find the last element
    VkBaseOutStructure* lastPrepend = prependStruct;
    while(lastPrepend->pNext)
    {
      lastPrepend = reinterpret_cast<VkBaseOutStructure*>(lastPrepend->pNext);
    }
    lastPrepend->pNext = baseStruct->pNext;  // Append the original pNext chain of baseStruct to the end of prependStruct's chain
    baseStruct->pNext = prependStruct;  // Prepend the prependStruct to the baseStruct
  }

  void findQueueFamilies()
  {
    uint32_t queueFamilyCount;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueFamilies.data());

    std::unordered_map<uint32_t, uint32_t> queueFamilyUsage;
    for(uint32_t i = 0; i < queueFamilyCount; ++i)
    {
      queueFamilyUsage[i] = 0;
    }

    for(size_t i = 0; i < m_desiredQueues.size(); ++i)
    {
      bool found = false;
      for(uint32_t j = 0; j < queueFamilyCount; ++j)
      {
        // Check for an exact match and unused queue family
        // Avoid queue family with VK_QUEUE_GRAPHICS_BIT if not needed
        if((queueFamilies[j].queueFlags & m_desiredQueues[i]) == m_desiredQueues[i] && queueFamilyUsage[j] == 0
           && ((m_desiredQueues[i] & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
        {
          m_queueInfos.push_back({j, queueFamilyUsage[j]});
          queueFamilyUsage[j]++;
          found = true;
          break;
        }
      }

      if(!found)
      {
        for(uint32_t j = 0; j < queueFamilyCount; ++j)
        {
          // Check for an exact match and allow reuse if queue count not exceeded
          // Avoid queue family with VK_QUEUE_GRAPHICS_BIT if not needed
          if((queueFamilies[j].queueFlags & m_desiredQueues[i]) == m_desiredQueues[i]
             && queueFamilyUsage[j] < queueFamilies[j].queueCount
             && ((m_desiredQueues[i] & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
          {
            m_queueInfos.push_back({j, queueFamilyUsage[j]});
            queueFamilyUsage[j]++;
            found = true;
            break;
          }
        }
      }

      if(!found)
      {
        for(uint32_t j = 0; j < queueFamilyCount; ++j)
        {
          // Check for a partial match and allow reuse if queue count not exceeded
          // Avoid queue family with VK_QUEUE_GRAPHICS_BIT if not needed
          if((queueFamilies[j].queueFlags & m_desiredQueues[i]) && queueFamilyUsage[j] < queueFamilies[j].queueCount
             && ((m_desiredQueues[i] & VK_QUEUE_GRAPHICS_BIT) || !(queueFamilies[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
          {
            m_queueInfos.push_back({j, queueFamilyUsage[j]});
            queueFamilyUsage[j]++;
            found = true;
            break;
          }
        }
      }

      if(!found)
      {
        for(uint32_t j = 0; j < queueFamilyCount; ++j)
        {
          // Check for a partial match and allow reuse if queue count not exceeded
          if((queueFamilies[j].queueFlags & m_desiredQueues[i]) && queueFamilyUsage[j] < queueFamilies[j].queueCount)
          {
            m_queueInfos.push_back({j, queueFamilyUsage[j]});
            queueFamilyUsage[j]++;
            found = true;
            break;
          }
        }
      }

      if(!found)
      {
        // If no suitable queue family is found, assert a failure
        assert(!"Failed to find a suitable queue family!");
      }
    }

    for(const auto& usage : queueFamilyUsage)
    {
      if(usage.second > 0)
      {
        m_queuePriorities.emplace_back(usage.second, 1.0f);  // Same priority for all queues in a family
        m_queueCreateInfos.push_back({VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, nullptr, 0, usage.first, usage.second,
                                      m_queuePriorities.back().data()});
      }
    }
  }

  std::vector<ExtensionFeaturePair> filterAvailableExtensions(const std::vector<VkExtensionProperties>& availableExtensions,
                                                              const std::vector<ExtensionFeaturePair>& desiredExtensions)
  {
    std::unordered_map<std::string, bool> availableExtensionsMap;
    for(const auto& ext : availableExtensions)
      availableExtensionsMap[ext.extensionName] = true;

    std::vector<ExtensionFeaturePair> filteredExtensions;
    for(const auto& desiredExtension : desiredExtensions)
    {
      if(availableExtensionsMap.find(desiredExtension.extensionName) != availableExtensionsMap.end())
      {
        filteredExtensions.push_back(desiredExtension);
      }
      else
      {
        LOGE("Extension not available: %s\n", desiredExtension.extensionName);
      }
    }

    return filteredExtensions;
  }

  void cleanup()
  {
    vkDestroyDevice(m_device, m_settings.alloc);
    if(m_dbgMessenger)
    {
      auto vkDestroyDebugUtilsMessengerEXT =
          (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
      assert(vkDestroyDebugUtilsMessengerEXT != nullptr);
      vkDestroyDebugUtilsMessengerEXT(m_instance, m_dbgMessenger, m_settings.alloc);
    }
    vkDestroyInstance(m_instance, m_settings.alloc);
  }
};