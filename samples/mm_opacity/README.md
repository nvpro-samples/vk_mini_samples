# Opacity Micro-Map in Vulkan

![Opacity Micro-Map Visualization](docs/opacity.png)

## Overview

This sample demonstrates the implementation of Opacity Micro-Maps in Vulkan for efficient ray tracing.

## Key Concepts

- Encodes visibility for triangle sub-regions
- Triangle classification: opaque (blue), transparent unknown (red), transparent (invisible)
- Selective AnyHit shader invocation based on opacity state

## Implementation Details

### Micro-Triangle Generation
- `4^subdivision` triangles per subdivision level
- Barycentric coordinate mapping using `BirdCurveHelper::micro2bary`
- World position projection for radius-based classification

### Data Structures
1. Value Buffer: 2-state (1-bit) or 4-state (2-bit) opacity encoding
2. `VkMicromapTriangleEXT` Buffer: Maps triangles to value buffer data
3. Index Buffer: Continuous array for `VkMicromapTriangleEXT` indexing
4. `VkMicromapEXT`: Constructed from above data

### BLAS Integration
- `VkAccelerationStructureTrianglesOpacityMicromapEXT` attachment
- Linked via `pNext` of `VkAccelerationStructureGeometryTrianglesDataKHR`

## Opacity-MicroMap-SDK

<p align="center">
    <img width=45% height=auto src="docs/omm_off.png">
    <img width=45% height=auto src="docs/omm_on.png">
</p>

### Features
- Automated asset conversion for ray tracing optimization
- Texture and geometry analysis
- Runtime or offline processing capabilities
- CPU and GPU conversion options

### Integration
- Compatible with various ray tracing applications
- Flexible API for pipeline integration
- Supports diverse hardware and software configurations

[SDK Repository](https://github.com/NVIDIAGameWorks/Opacity-MicroMap-SDK/tree/main)


