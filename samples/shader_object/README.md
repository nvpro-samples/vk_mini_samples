# Shader Object

![](docs/shader_object.png)


This sample demonstrates how to create and use Shader Objects, as outlined in the Vulkan specification (https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_object.html). The sample also highlights the dynamic modification of values without a predefined pipeline. 

In addition to Shader Objects, the sample generates a Menger Sponge (https://en.wikipedia.org/wiki/Menger_sponge) by creating individual cube instances and merging them into a single mesh. This sample recreates a variation of the Mengel Sponge through user interaction. It involves recreating the buffers that hold the geometry before displaying it.

Furthermore, this example uses a different memory allocator than the one used in other examples. There is no particular reason for this, other than to demonstrate the usage of the [DeviceMemoryAllocator](https://github.com/nvpro-samples/nvpro_core/tree/master/nvvk#class-nvvkdevicememoryallocator) from the nvvk library, instead of [VMA](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) from AMD.

**Note**: To see what has changed in the code regarding shader object, search for `#SHADER_OBJECT`


## onAttach

The `onAttach` function creates the descriptor set, but no longer creates a Vulkan pipeline. Instead, it creates a shader object for each of the shaders. This sample creates two shader objects: one for the vertex and one for the fragment. Refer to `createShaderObjects()`.

 
## onRender

Instead of creating a pipeline, this sample codes each value separately within the `setupShaderObjectPipeline()` function. This has the advantage of allowing modification of any of the pipeline values without creating a new one. For example, the line width or polygon mode can be altered to suit the needs.

After the pipeline values are set, bind the vertex and fragment shaders. Ensure that unused shaders are bound, but a null pointer is passed for their shader object.

Another way to optimize performance is by drawing wireframes on top and setting only the necessary changes, such as bias offset, polygon mode, and binding a different fragment shader. This approach may have a greater impact on some GPU architectures than pipeline switches, but it can significantly reduce the number of pipelines needed and potentially simplify the application architecture. 



