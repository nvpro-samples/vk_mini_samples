# Rectangle Rendering Sample

![Rectangle Rendering Sample](docs/rectangle.png)

## Overview

This sample demonstrates basic quad rendering using vertex and fragment shaders, building upon the concepts introduced in the `solid_color` sample. It showcases the implementation of a simple geometry pipeline and the use of G-Buffers for off-screen rendering.

## Implementation Details

### Initialization (`onAttach`)

When `RectangleSample` is added to `nvvk::Application`, the `onAttach(app)` method is invoked, performing the following tasks:

1. Instantiates a memory allocator for efficient resource management.
2. Initializes a debug utility for resource naming and visualization in Nsight Graphics.
3. Creates the rendering pipeline via `createPipeline()`, defining the rendering configuration.
4. Generates vertex and index buffers defining the rectangle geometry in `createGeometryBuffers()`.

### Resizing (`onResize`)

The `onResize(w,h)` method manages G-Buffer creation:

- Color format: `VK_FORMAT_R8G8B8A8_UNORM`
- Depth format: Dynamically queried to optimize for the current physical device

### Rendering

#### G-Buffer Rendering

- The rectangle is rendered into the G-Buffer, not directly to the swapchain.
- G-Buffer contents are displayed using ImGui in `onUIRender()`.

#### Main Rendering Pass (`onRender`)

Utilizes [Vulkan Dynamic Rendering](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_KHR_dynamic_rendering.adoc):

1. Attaches G-Buffer color and depth targets.
2. Executes primitive rendering.

## Technical Specifications

- **Shader Types**: Vertex and Fragment
- **Geometry**: Single quad (rectangle)
- **Rendering Technique**: Off-screen G-Buffer rendering with ImGui display
- **Vulkan Features**: Dynamic Rendering

