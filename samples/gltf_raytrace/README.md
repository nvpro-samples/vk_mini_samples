# Vulkan-based glTF Renderer

![glTF Render Example](docs/gltf.png)

## Overview

This sample demonstrates loading and rendering glTF scenes using Vulkan. It utilizes tinyGLTF for scene loading, `nvh::gltf::Scene` for internal representation, and `nvvkhl::SceneVk` for GPU resource creation.

## Key Components

- **Scene Loading**: tinyGLTF
- **Internal Representation**: `nvh::gltf::Scene`
- **GPU Resource Management**: `nvvkhl::SceneVk`
- **Ray Tracing Acceleration Structures**: `nvvkhl::SceneRtx`

## Pipeline Architecture

- Single compute shader with ray query
- Shader Object and Push Descriptors implementation
- Optimized for scenes with moderate element count

## Render Process

1. Update frame-specific buffers (e.g., camera data)
2. Push descriptors: TLAS, output image, scene data
3. Set push constants
4. Dispatch compute shader
5. Apply memory barrier for post-processing

## Usage

Pass glTF scene path as an executable argument for custom scene loading.

## Performance Considerations

Push Descriptors may generate warnings for scenes with numerous textures. Refer to `pushDescriptorSet()` implementation for details.