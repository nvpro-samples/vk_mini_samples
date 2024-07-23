# Vulkan Image Viewer

![Image Viewer Interface](docs/image_viewer.png)

## Overview

This sample demonstrates a Vulkan-based image viewer with zoom and pan functionality, and dynamic texture sampler adjustment.

## Key Features

- Image loading and display
- Zoom functionality (mouse wheel)
- Pan functionality (mouse drag)
- Texture sampler switching (nearest/linear)

## Implementation Details

### Zoom and Pan Mechanics
- Implemented in `onUIRender()`
- Mouse-over viewport detection
- Zoom level adjustment via mouse wheel
- Pan offset calculation accounting for zoom level
- Values pushed to shader in `onRender()` via orthographic matrix

### Rendering
- Image rendered on a square primitive
- Aspect ratio preservation:
  - Scaling applied in `onRender()`
  - Maintains image proportions relative to viewport dimensions

### Shader Integration
- Orthographic matrix in vertex shader:
  - Incorporates zoom (scale) and pan (translation)

### Texture Sampling
- Dynamic switching between nearest and linear sampling

## Performance Considerations

- Efficient viewport mouse position tracking
- Optimized zoom and pan calculations for smooth user experience