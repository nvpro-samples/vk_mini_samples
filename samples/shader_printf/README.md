# Debug Printf

![img](docs/printf.png)

[Debug Printf](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/docs/debug_printf.md) allows to debug shaders using a debug printf function. 

Even with the help of a great debugging tool like [Nsight Graphics](https://developer.nvidia.com/nsight-graphics), debugging Vulkan shaders can be very challenging. [Debug Printf](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/docs/debug_printf.md) feature enables  to put Debug Print statements into shaders to debug them. This sample shows how it is added to [nvpro-sample](https://github.com/nvpro-samples) simple applications.




## Enabling Debug_printf

In `main()`, we will need a new extension [`VK_KHR_shader_non_semantic_info`](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shader_non_semantic_info.html) (prior to Vulkan 1.3) and enabling validation features through [`VkValidationFeaturesEXT`](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkValidationFeaturesEXT.html). In this sample, we are only enabling **debug printf** and not disabling any features.

````cpp
  // #debug_printf
  contextInfo.addDeviceExtension(VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME);
  std::vector<VkValidationFeatureEnableEXT> enables{VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
  std::vector<VkValidationFeatureDisableEXT> disables{};
  VkValidationFeaturesEXT features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  features.enabledValidationFeatureCount  = static_cast<uint32_t>(enables.size());
  features.pEnabledValidationFeatures     = enables.data();
  features.disabledValidationFeatureCount = static_cast<uint32_t>(disables.size());
  features.pDisabledValidationFeatures    = disables.data();
  contextInfo.instanceCreateInfoExt       = &features;
````

## Printf in Shaders

To print messages in a shader you need to add an extension.

````cpp
#extension GL_EXT_debug_printf : enable 
````

Then you can use `debugPrintfEXT()` to print messages, but please note the [limitations](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/docs/debug_printf.md).

````cpp
debugPrintfEXT("HERE");
````

## Catching Messages

To get messages, a Messenger need to be created with `INFO` flag to get the severty level that the printf send its messages. 

```
  // Creating the callback
  VkDebugUtilsMessengerEXT           dbg_messenger{};
  VkDebugUtilsMessengerCreateInfoEXT dbg_messenger_create_info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  dbg_messenger_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT;
  dbg_messenger_create_info.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
  dbg_messenger_create_info.pfnUserCallback = dbg_messenger_callback;
  NVVK_CHECK(vkCreateDebugUtilsMessengerEXT(app->getContext()->m_instance, &dbg_messenger_create_info, nullptr, &dbg_messenger));
  ```

The callback for the messages, have been written like this. Note that we are stripping the incomming string, to make the message more clear.

```
  // #debug_printf
  // Vulkan message callback - for receiving the printf in the shader
  // Note: there is already a callback in nvvk::Context, but by defaut it is not printing INFO severity
  //       this callback will catch the message and will make it clean for display.
  auto dbg_messenger_callback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                   const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData) -> VkBool32 {
    // Get rid of all the extra message we don't need
    std::string clean_msg = callbackData->pMessage;
    clean_msg             = clean_msg.substr(clean_msg.find_last_of('|') + 1);
    nvprintf(clean_msg.c_str());  // <- This will end up in the Logger
    return VK_FALSE;              // to continue
  };
  ``` 

  **NOTE** Do not forget to detroy the messenger at the end of the application.

  ```
    // #debug_printf : Removing the callback
  vkDestroyDebugUtilsMessengerEXT(app->getContext()->m_instance, dbg_messenger, nullptr);
  ```



## Interactive Debug

We should avoid many printf, more specifically avoiding to be executed once per fragment. By default, the message buffer has a limitation of 1024 bytes. This is quite small and we should be careful not to exceed the buffer size.

Example of the error when exceeding the buffer size.

````
WARNING - Debug Printf message was truncated, likely due to a buffer size that was too small for the message
````

### User Interaction

To limit the amount of debug data to the information under the mouse cursor, in `onUIRender()`, we check if the mouse button is down and set the mouse cordinate, otherwise the value is set to `{-1, -1}`, which cannot be hit. Before printing information in the shader, we check if the current fragment coordinate is equal to the mouse cursor value.

```
  if(fragCoord == ivec2(pushC.mouseCoord))
    debugPrintfEXT("\n[%d, %d] Color: %f, %f, %f\n", fragCoord.x, fragCoord.y, inFragColor.x, inFragColor.y, inFragColor.z);
```

## Logging output

The output goes normaly to the console, but there is a way to forward all the data to another function. 

In `main()` there is a line that is doing this

```
// For logging
nvprintSetCallback([](int /*level*/, const char* fmt) { g_logger.addLog("%s", fmt); });
```  

The variable `g_logger` is a `SampleAppLog` that collects all logs, this was taken from Imgui example. In combination with `LoggerEngine`, it allows to render all logs in a separate Imgui window. To to this, another engine need to be added.

```
  app->addEngine(std::make_unique<LoggerEngine>(true));  // Add logger window
```


## Vulkan Config

Using the [Vulkan Configurator](https://vulkan.lunarg.com/doc/view/1.2.135.0/windows/vkconfig.html), it is possible to overide the size of the printf buffer. Open the configurator, set the values and while it is still open, run sample. The messages can be longer.

![](docs/vkconfig.png)

## References

* <https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/docs/debug_printf.md>
