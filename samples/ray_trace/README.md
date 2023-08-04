# Raytrace

![](docs/raytrace.png)

Rendering simple primitives using ray tracing, GGX shading, shadows and reflections. 

:warning: If you are looking for global illumination, look at the gltf_viewer sample.


## `onAttach()`

The overrided method `onAttach()` will be creating the utility classes for building top and bottom acceleration structures (TLAS/BLAS) as well as the utility for creating the shading binding table (SBT) used by the RTX pipeline. 

The scene (`createScene`) is created, the vertices and indices are uploaded in Vulkan buffer (`createVkBuffers`). The BLAS and TLAS are created respectively in `createBottomLevelAS` and `createTopLevelAS`. The ray tracing pipeline and shading binding table are created in `createRtPipeline`, this is where the raygen, miss and closest-hit shaders are attached. 

## `onUIRender`

This is where the UI is rendered, and where the rendered image is set to cover the entire viewport.

## `onRender(cmd)`

The rendering of the scene is setting the information of the frame in buffers used by the shaders and call `vkCmdTraceRaysKHR`.

