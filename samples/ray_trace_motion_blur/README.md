# Raytraced Motion Blur in Vulkan

![Motion Blur Demonstration](docs/motion_blur.png)

## Overview

This sample demonstrates three motion blur methods in Vulkan raytracing: Matrix, SRT, and Object.

## Implementation Details

### Extension Requirements
- Enable `VK_NV_ray_tracing_motion_blur`
- Set `rayTracingMotionBlur` to true in `VkPhysicalDeviceRayTracingMotionBlurFeaturesNV`

### Top-Level Acceleration Structure (TLAS)
- Use `VkAccelerationStructureMotionInstanceNV` with padding for 160-byte stride
- Define MATRIX and SRT motion
- Specify transformations at time 0 and 1
- Mark static objects with `VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_STATIC_NV`

### Bottom-Level Acceleration Structure (BLAS)
- For object deformation, use `VkAccelerationStructureGeometryMotionTrianglesDataNV`
- Reference objects at time 0 and 1

### Pipeline Configuration
- Set `VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV` flag in `VkRayTracingPipelineCreateInfoKHR`

### Shader Implementation
- Use `traceRayMotionNV` for time-specific ray tracing
- Require `GL_NV_ray_tracing_motion_blur` extension

## Code References
Search for `#NV_Motion_blur` in the source code for specific implementations.

## Key Functions
- `createTopLevelAS()`: TLAS setup with motion data
- `createBottomLevelAS()`: BLAS configuration for deformation

## Technical Considerations
- Linear interpolation of transformations in shaders
- Performance impact of motion blur on raytracing
- Balancing motion blur quality and computational cost