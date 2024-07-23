# Ray Query Position Fetch in Vulkan

![Ray Query Position Fetch Demonstration](docs/ray_query_pos_fetch.png)

## Overview

This sample demonstrates the use of ray queries in a compute shader to fetch triangle positions, as an alternative to the raytracing pipeline approach.

## Key Features

- Utilizes [VK_KHR_ray_query](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_ray_query.html) extension
- Compute shader implementation for position retrieval
- Dynamic scene animation via Top-Level Acceleration Structure (TLAS) updates

## Implementation Details

1. **Ray Query Execution**
   - Performed within compute shader
   - Retrieves triangle position data

2. **Geometric Processing**
   - Normal generation from retrieved position data
   - Shading application based on computed normals

3. **Animation Technique**
   - TLAS matrix modification
   - Dynamic rebuilding of TLAS for each frame

## Technical Considerations

- Performance comparison with raytracing pipeline approach
- Efficiency of compute shader-based position fetching
- TLAS update optimization for smooth animation

## Applicability

- Scenarios requiring efficient geometry data access without full raytracing pipeline
- Dynamic scenes with frequent TLAS updates