# Vulkan Raytracing Implementation

![Raytracing Render Example](docs/raytrace.png)

## Overview

This sample demonstrates basic raytracing in Vulkan, featuring GGX shading, shadows, and reflections.

## Key Components

### `onAttach()` Method
- Initializes utility classes for:
  - Top-Level Acceleration Structure (TLAS)
  - Bottom-Level Acceleration Structure (BLAS)
  - Shading Binding Table (SBT)
- Scene creation and buffer uploading
- Acceleration structure generation
- Ray tracing pipeline setup

### `onUIRender()` Method
- UI rendering
- Full viewport image display

### `onRender(cmd)` Method
- Frame information buffer updates
- `vkCmdTraceRaysKHR` invocation

## Implementation Details

1. **Scene Creation**: `createScene()`
2. **Vulkan Buffer Generation**: `createVkBuffers()`
3. **Acceleration Structures**:
   - BLAS: `createBottomLevelAS()`
   - TLAS: `createTopLevelAS()`
4. **Pipeline Creation**: `createRtPipeline()`
   - Shader attachment (raygen, miss, closest-hit)

## Technical Considerations
- Optimization of acceleration structure updates
- Efficient SBT management
- Performance tuning for real-time raytracing

## Note
For global illumination implementations, refer to the `gltf_viewer` sample.