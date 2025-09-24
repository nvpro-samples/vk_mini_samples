# Vulkan Samples

This repository contains numerous examples demonstrating various aspects of Vulkan, debugging techniques, and integration with other NVIDIA tools. For a comprehensive list, refer to the [Samples](#Samples) section below.

Each sample is accompanied by its own documentation detailing functionality and providing references for further information.


## Dependencies

* [nvpro_core2](https://github.com/nvpro-samples/nvpro_core2): A collection of Vulkan helper classes and utilities.

## Build Instructions

### Cloning Repositories

```bash
git clone --recursive --shallow-submodules https://github.com/nvpro-samples/nvpro_core2.git
git clone https://github.com/nvpro-samples/vk_mini_samples.git
```

### Generating Solution

```bash
cd vk_mini_samples
mkdir build
cd build
cmake ..
```


### Additional SDKs

The Aftermath sample requires the separate download of the [Nsight Aftermath SDK](https://developer.nvidia.com/nsight-aftermath).

### Shader Language Options: GLSL or SLANG

Samples can use either GLSL or Slang (default), and can be changed by changing the define `USE_SLANG` from `1` to `0` to use GLSL.

## Samples

For those new to this repository, the [solid color](samples/solid_color) and [rectangle](samples/rectangle) samples are recommended starting points to better understand the framework.


| Name | Description | Image | GLSL | Slang |
| ------ | ------ | ---- | ---- | ---- |
| [barycentric_wireframe](samples/barycentric_wireframe) | Single-pass solid-wireframe rendering using `gl_BaryCoordNV` | ![](samples/barycentric_wireframe/docs/bary_wireframe_th.jpg) | [x] |  [x] |
| [compute_multi_threaded](samples/compute_multi_threaded) | Executing a compute shader in a separate thread faster than the main thread.| ![](samples/compute_multi_threaded/docs/multi_threaded_th.jpg) | [x] | [x] |
| [compute_only](samples/compute_only) | Basic compute and display example | ![](samples/compute_only/docs/compute_only_th.jpg) | [x] | [x] |
| [crash_aftermath](samples/crash_aftermath) | Integration of Nsight Aftermath SDK into an existing application | ![](samples/crash_aftermath/docs/aftermath_th.jpg) | [x] | [x] |
| [gltf_raytrace](samples/gltf_raytrace) | glTF scene loading with path-tracing renderer |  ![](samples/gltf_raytrace/docs/gltf_th.jpg) | [ ] | [x] | 
| [gpu_monitor](samples/gpu_monitor) | GPU usage visualization | ![](samples/gpu_monitor/gpu_monitor_th.png) | [x] | [x] | 
| [image_ktx](samples/image_ktx) | KTX image display with tonemapping post-processing | ![](samples/image_ktx/docs/image_ktx_th.jpg) | [x] | [x] | 
| [image_viewer](samples/image_viewer) | Image loading with zoom and pan functionality | ![](samples/image_viewer/docs/image_viewer_th.jpg) | [x] | [x] | 
| [line_stipple](samples/line_stipple) | Dashed line rendering with stipple pattern | ![](samples/line_stipple/docs/line_stipple_th.jpg) | [x] | [x] | 
| [memory_budget](samples/memory_budget) | Dynamic memory allocation within budget constraints | ![](samples/memory_budget/docs/mem_budget_th.jpg) | [x] | [x] | 
| [mm_opacity](samples/mm_opacity) | Micromap opacity implementation  | ![](samples/mm_opacity/docs/opacity_th.jpg) | [x] | [x] | 
| [msaa](samples/msaa) | Hardware Multi-Sampling Anti-Aliasing demonstration  | ![](samples/msaa/docs/msaa_th.jpg) | [x] | [x] | 
| [offscreen](samples/offscreen) | Windowless rendering with image save functionality.  | ![](samples/offscreen/docs/offline_th.jpg) | [x] | [x] | 
| [ray_query](samples/ray_query) | Inline raytracing in compute shaders | ![](samples/ray_query/docs/ray_query_th.jpg) | [x] | [x] | 
| [ray_query_position_fetch](samples/ray_query_position_fetch) | Using VK_KHR_ray_tracing_position_fetch usage in ray query | ![](samples/ray_query_position_fetch/docs/ray_query_pos_fetch_th.jpg) | [x] | [x] |
| [ray_trace](samples/ray_trace) | Basic ray tracer with metallic-roughness shading, reflections, shadows, and sky shader.  | ![](samples/ray_trace/docs/raytrace_th.jpg) | [ ] | [x] | 
| [ray_trace_motion_blur](samples/ray_trace_motion_blur) | Motion blur for dynamic objects using NVIDIA raytracing extension | ![](samples/ray_trace_motion_blur/docs/motion_blur_th.jpg) | [x] | [x] |
| [ray_tracing_position_fetch](samples/ray_tracing_position_fetch) | VK_KHR_ray_tracing_position_fetch implementation. | ![](samples/ray_tracing_position_fetch/docs/fetch_th.jpg) | [x] | [x] |
| [realtime_analysis](samples/realtime_analysis) | Real-time GPU information display | ![](samples/realtime_analysis/docs/realtime_analysis_th.jpg) | [x] | [ ] |
| [rectangle](samples/rectangle) | 2D rectangle rendering to GBuffer.  | ![](samples/rectangle/docs/rectangle_th.jpg) | [x] | [x] | 
| [ser_pathtrace](samples/ser_pathtrace) | Shading Execution Reordering (SER) for optimized GPU usage.  | ![](samples/ser_pathtrace/docs/ser_2_th.jpg) | [x] | [x] | 
| [shader_object](samples/shader_object) | Shader object and dynamic pipeline usage | ![](samples/shader_object/docs/shader_object_th.jpg) | [x] | [x] | 
| [shader_printf](samples/shader_printf) | Shader debugging with printf functionality  | ![](samples/shader_printf/docs/printf_th.jpg) | [x] | [x] | 
| [simple_polygons](samples/simple_polygons) | Multi-polygon object rasterization.  | ![](samples/simple_polygons/docs/simple_polygons_th.jpg) | [x] | [x] | 
| [solid_color](samples/solid_color) | Single-pixel texture creation and display.  | ![](samples/solid_color/docs/solid_color_th.jpg) | [x] | [x] | 
| [texture 3d](samples/texture_3d) | 3D texture creation and ray marching. | ![](samples/texture_3d/docs/texture_3d_th.jpg) | [x] | [x] | 
| [tiny_shader_toy](samples/tiny_shader_toy) | Real-time shader compilation with error display and multi-stage pipelines.  | ![](samples/tiny_shader_toy/docs/tiny_shader_toy_th.jpg) | [x] | [x] |

## Rendering Architecture

Those samples demonstrates an indirect rendering approach, diverging from direct swapchain image rendering. The rendering pipeline is structured as follows:

1. **Off-screen Rendering**: The sample renders its content to an off-screen image buffer rather than directly to the swapchain image.
2. **GUI Integration**: The rendered off-screen image is incorporated as an element within the GUI layout.
3. **Composite Rendering**: The `nvapp::Application` framework manages the final composition step. It combines the GUI elements (including the embedded rendered image) into a unified layout.
4. **Swapchain Presentation**: The composite result from step 3 is then rendered to the swapchain image for final presentation.

This architecture provides several advantages:
- Decouples the sample's rendering from the final presentation
- Allows for flexible GUI integration of rendered content
- Facilitates additional post-processing or compositing operations

Developers should note that the actual swapchain image rendering is abstracted away within the `nvapp::Application` class, providing a clean separation of concerns between sample-specific rendering and final frame composition.

## Application Architecture

The examples in this repository leverage various utilities from the [nvpro_core2](https://github.com/nvpro-samples/nvpro_core2) framework. Central to each sample's implementation is the [`Application`](https://github.com/nvpro-samples/nvpro_core2/blob/master/nvapp/application.hpp) class, which provides core functionality for:

- Window creation and management
- User interface (UI) initialization
- Swapchain setup integrated with the ImGui framework

The `Application` class is an enhanced derivative of the Dear ImGui Vulkan example, optimized for our use cases.

### Modular Design

Samples are implemented as `Elements` and attached to the `Application` instance. This modular approach allows for:

1. Separation of concerns between core application logic and sample-specific code
2. Consistent handling of UI rendering and frame operations across different samples

### Application Lifecycle

The following diagram illustrates the complete application lifecycle, from initialization through the main rendering loop:

```mermaid
---
config:
  layout: dagre
---
flowchart LR
    subgraph s1["Application Element"]
        O["onAttach: Initialize"]
        P["onUIMenu: Menu Items"]
        Q["onUIRender: UI Widgets"]
        R["onPreRender: Pre-frame Setup"]
        S["onRender: GPU Commands"]
        T["onPostRender: Post-frame Setup"]
        U["onDetach: Cleanup"]
    end
    
    A["Constructor"] --> B["init: Setup Window, Vulkan, ImGui"]
    B --> C["run: Main Loop"]
    C --> D["Frame Setup: Events, ImGui, Viewport"]
    D --> E["prepareFrameResources"]
    E -- Success --> F["beginCommandRecording"]
    E -- Fail --> C
    F --> I["drawFrame: Element Processing"]
    I --> J["renderToSwapchain: ImGui"]
    J --> L["endFrame: Submit Commands"]
    L --> M["presentFrame"]
    M --> N["advanceFrame"]
    N --> C
    
    B -. addElement .-> O
    D -. Menu Bar .-> P
    I -. UI Phase .-> Q
    I -. "Pre-Render" .-> R
    I -. Render Phase .-> S
    I -. "Post-Render" .-> T
    
    C -->|Exit Event| V["shutdown()"]
    V -. onDetach .-> U
    
    E@{ shape: decision}
    style O fill:#f1f8e9
    style P fill:#f1f8e9
    style Q fill:#f1f8e9
    style R fill:#f1f8e9
    style S fill:#f1f8e9
    style T fill:#f1f8e9
    style U fill:#f1f8e9
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style I fill:#FFE0B2
    style M fill:#FFE0B2
    style V fill:#ffebee
```

### Initialization Process

The `init()` method orchestrates the following setup procedures:

1. **GLFW Initialization**: `glfwInit()` sets up the windowing system
2. **Vulkan Context**: `nvvk::Context::init()` creates the Vulkan instance, device, and queues
3. **Window Creation**: `ImGui_ImplVulkanH_CreateOrResizeWindow()` creates the window and swapchain
4. **ImGui Setup**: `ImGui_ImplVulkan_Init()` initializes the ImGui Vulkan backend
5. **Swapchain**: Manages presentation images

During initialization, core Vulkan resources are provided by the framework:
- **VkInstance**: Connection between application and Vulkan library
- **VkPysicalDevice**: Representation of the physical GPU
- **VkDevice**: Logical representation of the physical GPU
- **VkQueue**: Command submission queue for GPU operations

### Execution Cycle

The `run()` method implements the main application loop, continuing until a termination event is triggered. Each iteration follows this sequence:

#### Frame Preparation
1. **Frame Setup**: Process events, update ImGui, handle viewport changes
2. **Resource Management**: `prepareFrameResources()` acquires swapchain image
3. **Cleanup**: `freeResourcesQueue()` releases previous frame resources
4. **Synchronization**: `prepareFrameToSignal()` sets up frame synchronization

#### Command Recording
5. **Command Buffer**: `beginCommandRecording()` starts recording GPU commands
6. **Element Processing**: `drawFrame()` invokes element callbacks in sequence:
   - `onUIRender`: UI widget rendering
   - `onPreRender`: Pre-frame setup operations
   - `onRender`: Sample-specific GPU commands
   - `onPostRender`: Post-frame cleanup operations

#### Frame Completion
7. **ImGui Rendering**: `renderToSwapchain()` renders UI to swapchain image
8. **Synchronization**: `addSwapchainSemaphores()` sets up presentation synchronization
9. **Submission**: `endFrame()` submits command buffers to GPU
10. **Presentation**: `presentFrame()` presents the completed frame
11. **Advancement**: `advanceFrame()` moves to next frame resources

### Element Lifecycle

Elements attached to the application follow a well-defined lifecycle:

- **`onAttach`**: Called during `addElement()`, used for initialization
- **`onUIMenu`**: Called during frame setup, adds menu items to the menu bar
- **`onUIRender`**: Called during UI phase, renders ImGui widgets
- **`onPreRender`**: Called before main rendering, handles pre-frame setup
- **`onRender`**: Called during render phase, records GPU commands
- **`onPostRender`**: Called after main rendering, handles post-frame cleanup
- **`onDetach`**: Called during shutdown, used for cleanup and resource deallocation


## Shader Language Support

### SPIR-V Intermediate Representation

Vulkan utilizes SPIR-V as its intermediate shader representation, diverging from the direct consumption of human-readable shader text. This architectural decision enables support for multiple high-level shader languages, provided they can target the Vulkan SPIR-V environment.


### Supported Languages

#### Slang (Default)

[Slang](https://github.com/shader-slang/slang) is a high-level shader language with syntax resembling C++. It is extensively used in NVIDIA research due to its versatility in targeting multiple backends:

- SPIR-V (Vulkan)
- DirectX 12
- CUDA
- C++

To specify a custom Slang compiler version, modify the `Slang_VERSION` CMakeLists.txt.


#### GLSL

GLSL (OpenGL Shading Language) serves as the default shader language when neither Slang nor HLSL is explicitly enabled. It is natively supported by the Vulkan ecosystem.

This multi-language support strategy offers developers flexibility in shader authoring while maintaining compatibility with Vulkan's SPIR-V requirements.


### Resources

#### SLANG
- [GitHub Repository](https://github.com/shader-slang/slang)
- [Releases](https://github.com/shader-slang/slang/releases)
- [Getting Started Guide](https://shader-slang.com/getting-started.html)
- [User Guide](http://shader-slang.com/slang/user-guide/index.html)
- [Documentation](https://github.com/shader-slang/slang/tree/master/docs)
- [GLSL and SPIR-V Interoperability](https://shader-slang.com/slang/user-guide/a1-04-interop.html)
- [Standard Library Reference](https://shader-slang.com/stdlib-reference/index.html)
- [Language Specification](https://htmlpreview.github.io/?https://github.com/shader-slang/spec/blob/main/index.html#intro)

#### SPIR-V Intrinsics
- [GL_EXT_spirv_intrinsics](https://github.com/microsoft/DirectXShaderCompiler/wiki/GL_EXT_spirv_intrinsics-for-SPIR-V-code-gen)
- [KHR Extensions](https://github.com/KhronosGroup/SPIRV-Registry/tree/main/extensions/KHR)
- [JSON Specification](https://github.com/KhronosGroup/SPIRV-Headers/blob/main/include/spirv/unified1/spirv.json)
- [SPIR-V Specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)


## LICENSE

Copyright 2024-2025 NVIDIA CORPORATION. Released under Apache License,
Version 2.0. See "LICENSE" file for details.
