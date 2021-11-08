# Nsight Aftermath

![img](docs/vk_aftermath.png)

This example shows how to add Nsight Aftermath SDK to the project, to generate a dump helping to dump GPU crashes.

We will make it crash when switching to raster rendering. :smiley: 


## SDK sources

This project requires to download the [Aftermath SDK](https://developer.nvidia.com/nsight-aftermath).

Extract its content and copy the location of the SDK.

### Setting the NSIGHT_AFTERMATH_SDK variable

When running CMake the first time, you probally have seen this error:

```
NSIGHT_AFTERMATH_SDK environment variable not set to a valid location (value: )
```

To fix this, in the CMake-gui, open the  `Ungrouped Entries` and fill the `NSIGHT_AFTERMATH_SDK` variable with the path of where the SDK was extracted.

Example:
![](docs/cmake-gui.png)

Now, rerun CMake and the the error should go away. 

## SDK Callbacks

Nsight Aftermath requires callback when there is a crash, to simplify the operation, a simple self-contain class can do the job. You can find this class locally under:

* [NsightAftermathGpuCrashTracker.cpp](NsightAftermathGpuCrashTracker.cpp)
* [NsightAftermathGpuCrashTracker.h](NsightAftermathGpuCrashTracker.h)

## Enabling Nsight Aftermath

To enable Nsight Aftermath, we need to enable some device extensions and set the callbacks **before** creating the Vulkan device.

In `main.cpp`, include 

```` C
#include "NsightAftermathGpuCrashTracker.h"
````

Then add the following extensions at the end of the other ones.

```` C
  // #Aftermath
  // Set up device creation info for Aftermath feature flag configuration.
  VkDeviceDiagnosticsConfigCreateInfoNV aftermathInfo{VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV};
  aftermathInfo.flags = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV
                        | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV
                        | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV;
  // Enable NV_device_diagnostic_checkpoints extension to be able to use Aftermath event markers.
  contextInfo.addDeviceExtension(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
  // Enable NV_device_diagnostics_config extension to configure Aftermath features.
  contextInfo.addDeviceExtension(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME, false, &aftermathInfo);
```` 

Immediately following and before the creation of the Vulkan context, we initialize the GPU crash tracker.

 ````C
  // #Aftermath - Initialization
  GpuCrashTracker m_gpuTracker;
  m_gpuTracker.Initialize();
```` 


## Testing

The best way to test if it works, is to have a `VK_ERROR_DEVICE_LOST`.

In the fragment shader `frag_shader.frag`, we can access the material far beyond the safe memory point.
```` C
  ShadingMaterial mat     = gltfMat.m[pc.materialId + 10000000];
```` 

Running the sample will generate a crash as soon as we switch to Raster mode. 

## Dump File

![img](docs/crash.png)

The Aftermath dump file by default is written in the current working directory. On Windows, this will be the project directory `$(VulanSamples)\build\aftermath`. 


Open [Nsight Graphics](https://developer.nvidia.com/nsight-graphics), and `File>Open Files` or drag and drop the `vk_aftermath-*.nv-gpudmp` file. 

Click on **Crash Info** to have information about the crash. You will see that it is a **General_PageFault**.

If you click on the **Shader Location** file, you will have an approximated line where the crash happend.

## Other Crashes

Fix the previous crash before adding new ones.

### Infinit Loop

In the vertex shader, at the end of the file, add the following lines

```` glsl
  float alpha = 1.0;
  while(alpha > 0.0)
  {
    alpha += frameInfo.clearColor.x;
  }
  worldPos.x += 0.1 * alpha;
````

If you inspect the crash in Nsight Graphics, you will see that it has crashed over the while loop.

### Wrong Binding

In the same vertex shader, we can replace the binding to something that doesn't exist, such as

```` glsl
layout(set = 0, binding = 5) readonly buffer SceneDesc_ { SceneDescription sceneDesc; } ;
````

Now this will be catch by validation layer, but if we let it through, it will produce a crash when we switch to the raster mode.

The error will now point where we are accessing `sceneDesc.instInfoAddress`.








