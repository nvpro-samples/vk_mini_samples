# Solid Color

![](docs/solid_color.png)

# Solid Color Sample

This sample demonstrates rendering a single-color texture to the viewport, bypassing direct swapchain image rendering.

## Overview

The sample utilizes a 1x1 texture to display a user-selected color, updating dynamically based on GUI input.

## Implementation Details

### Initialization

In `main()`:
1. `nvvkhl::Application` is instantiated using `nvvkhl::ApplicationCreateInfo`.
   - This initializes the Vulkan instance, device, physical device, GLFW window, and Dear ImGui.
   - For detailed initialization steps, refer to `nvvkhl::Application::init()`.
2. The `SolidColor` sample is created, derived from `nvvkhl::IAppElement`.

### Resource Allocation

In `onAttach()`:
1. `AllocVma` is instantiated, deriving from `nvvk::ResourceAllocator`.
   - This core helper class from `nvpro_core/nvvk` facilitates efficient resource creation.
   - The sample utilizes [Vulkan Memory Allocator (VMA)](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) for memory management.
2. A 1x1 texture is created for color display.

### Main Loop

The application enters an infinite loop via `->run()` until closure is requested. Each iteration invokes:
- `onUIRender()`: Renders the GUI in the "Settings" window.
- `onUIMenu()`: Handles menu interactions.
- `onResize(w,h)`: Manages viewport resizing.
- `onRender(cmd)`: Executes per-frame rendering logic.

### Rendering Process

1. The texture is displayed as a UI element in the "Viewport" window using `ImGui::Image()`.
2. `onRender()` is called each frame.
3. If color changes are detected, `setData(cmd)` updates the texture with the new color selected in the GUI.

This implementation showcases efficient texture updating and rendering techniques within a Vulkan-based application framework.