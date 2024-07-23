# Simple Polygons

![Simple Polygons Screenshot](docs/simple_polygons.png)

This project demonstrates a basic raster rendering example in Vulkan, featuring:
- Camera manipulation
- Various geometric primitives
- Instance rendering
- Basic lighting model

## Overview

The Simple Polygons sample showcases fundamental 3D graphics concepts using a modern Vulkan implementation. It serves as an educational tool for understanding basic rendering pipelines and scene management.

## Key Components

### Scene Initialization (`onAttach`)

The `onAttach` function sets up the rendering environment:

1. Initializes the VMA (Vulkan Memory Allocator) for efficient memory management
2. Constructs the scene graph, including:
   - Primitive geometries
   - Material definitions
   - Instance references to meshes
3. Creates Vulkan representations of the scene:
   - Generates vertex and index buffers via `createVkBuffers`
   - Establishes the rendering pipeline

### User Interface (`onUIRender`)

The UI component provides:
- A camera widget in the settings window for view manipulation
- A viewport window displaying the rendered G-Buffer

### Rendering Loop (`onRender`)

The main rendering function iterates through all instance nodes in the scene, drawing each element.

> **Note:** For performance optimization, especially with larger scenes, consider recording a command buffer instead of looping over rendering nodes in real-time.
