# Position Fetch in Vulkan Ray Tracing

![Fetch Illustration](docs/fetch.png)

## Overview

This implementation demonstrates the use of `VK_KHR_ray_tracing_position_fetch` extension in Vulkan. It enables raytracing using only the acceleration structure data, without additional vertex buffers.

Key aspects:
- Positions are fetched directly from the acceleration structure
- Geometric normals are calculated from retrieved positions
- All buffers used for acceleration structure creation are deleted pre-rendering
- Results in a lightweight rendering process

For detailed information: [Vulkan Ray Tracing Position Fetch Extension](https://www.khronos.org/blog/introducing-vulkan-ray-tracing-position-fetch-extension)

## Implementation Details

### Acceleration Structure Creation
```cpp
// Add to bottom acceleration structure
VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR
```

### Extension Enablement
```cpp
VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR fetchFeatures{
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR
};
spec.vkSetup.addDeviceExtension(
    VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, 
    false, 
    &fetchFeatures
);
```

### Shader Implementation
In the closest hit shader (`raytrace.chit`):
```glsl
vec3 vertex = gl_HitTriangleVertexPositionsEXT[n];
```

