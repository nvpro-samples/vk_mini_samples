# Debug Printf

![img](docs/printf.png)

[Debug Printf](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/master/docs/debug_printf.md) allows to debug shaders using a debug printf function. This function works conjunctly with [VK_EXT_debug_utils](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_EXT_debug_utils.html), and the print messages will be sent with the `VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT` flag.

In debug, the [nvvk](https://github.com/nvpro-samples/nvpro_core/tree/master/nvvk) framework vulkan context creation ([context_vk](https://github.com/nvpro-samples/nvpro_core/blob/master/nvvk/context_vk.cpp)) uses validation layers by default. In addition, the while creating the Vulkan instance, the helper class creates the callback mechanism to catch various messages.

Look for [`Context::debugMessengerCallback()`](https://github.com/nvpro-samples/nvpro_core/blob/master/nvvk/context_vk.cpp#L122) for the callback and `Context::initDebugUtils()` for the creation.

## Enabling Debug_printf

In `main()`, we will need a new extension [`VK_KHR_shader_non_semantic_info`](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VK_KHR_shader_non_semantic_info.html) and enabling validation features through [`VkValidationFeaturesEXT`](https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkValidationFeaturesEXT.html). In this sample, we are only enabling **debug printf** and not disabling any features.

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
