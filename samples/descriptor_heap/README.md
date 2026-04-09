# Descriptor Heap Sample

![Descriptor Heap](docs/descriptor_heap.png)

## Overview

This sample demonstrates [VK_EXT_descriptor_heap](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_descriptor_heap.html), which replaces traditional descriptor sets with GPU-visible heap buffers and push data. Descriptors are written directly to host-visible memory, and shaders access them either through set/binding mapping or direct heap indexing.

The scene is an NxNxN RGB color cube where each small cube has 6 unique face textures showing its hex color code. Two rendering modes are provided:

- **Per-Draw**: Shaders use traditional `set/binding` declarations. `VkShaderDescriptorSetAndBindingMappingInfoEXT` maps them to the descriptor heap at pipeline creation time. One draw call per cube.
- **Bindless**: Shaders use `layout(descriptor_heap)` for direct heap access with no set/binding indirection. All cubes are rendered in a single instanced draw call.

## Why Descriptor Heaps?

Descriptor heaps replace descriptor pools and sets with simple GPU-visible buffers. Benefits include:
- **Scalability**: No pool fragmentation or set allocation limits; just write descriptors to memory.
- **Bindless patterns**: Direct heap indexing enables flexible, data-driven rendering without rebinding.
- **Modern API alignment**: Matches the resource binding model of D3D12 and Metal.

## Key Concepts

- **Push Data**: `vkCmdPushDataEXT` replaces `vkCmdPushConstants`. Layout: `FrameInfo` at offset 0 (160 bytes), then either `DrawData` or `BindlessPushData` at offset 160.
- **Per-Draw Mapping**: `VkShaderDescriptorSetAndBindingMappingInfoEXT` maps `set/binding` declarations to heap locations:
  - `HEAP_WITH_PUSH_INDEX`: Heap index read from push data at a specified offset (e.g., `baseFaceTexIdx`).
  - `HEAP_WITH_CONSTANT_OFFSET`: Fixed heap offset (e.g., sampler at slot 0).
- **Bindless**: `layout(descriptor_heap)` declares unsized arrays that index the heap directly. The shader computes `texIdx = baseFaceTexIdx + faceIdx` to sample the correct texture.

## Data Flow

**Common (both modes)**:
1. Bind sampler and resource heaps once per frame (`vkCmdBindSamplerHeapEXT`, `vkCmdBindResourceHeapEXT`).
2. Push `FrameInfo` at offset 0 (camera matrices, animation time).

**Per-Draw mode**:
1. For each cube, push `DrawData` at offset 160 (transform, `baseFaceTexIdx`, `cubeIndex`).
2. Issue `vkCmdDrawIndexed`. The driver reads `baseFaceTexIdx` from push data (via the mapping's `pushOffset`) to resolve `faceTextures[0..5]` from the heap.
3. The shader never reads `baseFaceTexIdx` directly—the mapping consumes it.

**Bindless mode**:
1. Push `BindlessPushData` once at offset 160 (`gridSize`, `borderColor`).
2. Issue one instanced `vkCmdDrawIndexed` with `instanceCount = numCubes`.
3. The vertex shader uses `gl_InstanceIndex` to derive each cube's position and `baseFaceTexIdx`.
4. The fragment shader indexes `heapTextures[baseFaceTexIdx + faceIdx]` directly.

## Code Map

Search for `#DESC_HEAP` in the C++ source to find all descriptor-heap-related code.

| Responsibility | Location |
|----------------|----------|
| Heap property query | `initHeaps()` |
| Sampler/resource heap allocation | `initHeaps()`, `resizeResourceHeap()` |
| Descriptor writes to staging | `writeImageDescriptor()` |
| Staging upload to device | `rebuildScene()` |
| Heap binding | `cmdBindHeaps()` |
| Push data | `cmdPushData()` |
| Per-draw mapping setup | `createShaders()` (per-draw block) |
| Bindless shader setup | `createShaders()` (bindless block) |
| Draw loop | `onRender()` |

## Converting from Legacy Descriptor Sets

If you're familiar with traditional descriptor sets (e.g., the [gltf_raytrace](../gltf_raytrace/) sample), this table maps legacy concepts to descriptor heaps:

| Legacy Concept | Example | Descriptor Heap Equivalent |
|----------------|---------|----------------------------|
| Descriptor pool + set | `vkCreateDescriptorPool`, `vkAllocateDescriptorSets` | Resource heap buffer with `VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT`; `vkWriteResourceDescriptorsEXT` to staging; upload to device |
| Set layout in pipeline | `VkDescriptorSetLayout` | No set layout needed; use mapping (`VkShaderDescriptorSetAndBindingMappingInfoEXT`) or bindless (`layout(descriptor_heap)`) |
| Push constants | `vkCmdPushConstants` | `vkCmdPushDataEXT`; push data also feeds heap indices via mapping |
| Push descriptors | `vkCmdPushDescriptorSet` | Write descriptors to heap at known indices; pass base index in push data |
| Binding sets each frame | `vkCmdBindDescriptorSets` | `vkCmdBindSamplerHeapEXT` + `vkCmdBindResourceHeapEXT` once per frame |

**When to use mapping vs bindless**: Use mapping for incremental migration (keeps `set/binding` in shaders). Use bindless for maximum flexibility, single-draw instanced patterns, and data-driven rendering.

## Implementation Details

### Descriptor Heap Setup

- Physical device properties are queried for descriptor sizes, alignment, and reserved range requirements.
- Sampler and resource heaps are allocated as device-local `VkBuffer` objects with `VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT` for fast GPU reads.
- `vkWriteSamplerDescriptorsEXT` and `vkWriteResourceDescriptorsEXT` write descriptors into host staging memory. The staging data is then uploaded to the device-local heaps via `StagingUploader`.

### Push Data

`vkCmdPushDataEXT` replaces `vkCmdPushConstants` when using descriptor heaps. Frame-level data (camera matrices, animation time) is pushed once, and per-draw data (transform, texture index) is pushed before each draw call.

### Per-Draw Shaders

Shader objects are created with `VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT` and no descriptor set layout. `VkShaderDescriptorSetAndBindingMappingInfoEXT` is chained into each shader's `pNext` to define how `set/binding` declarations map to heap offsets. (For pipeline-based workflows, the equivalent flag is `VK_PIPELINE_CREATE_2_DESCRIPTOR_HEAP_BIT_EXT`.)

- Textures use `VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT` (heap offset derived from push data).
- Samplers use `VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT` (fixed heap offset).

### Bindless Shaders

Shaders declare `layout(descriptor_heap) uniform texture2D heapTextures[]` for direct heap access. No set/binding mapping is needed. The SPIR-V requires `VK_KHR_shader_untyped_pointers`.

### Extensions

- [VK_EXT_descriptor_heap](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_descriptor_heap.html) — Core descriptor heap functionality.
- [VK_KHR_shader_untyped_pointers](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_shader_untyped_pointers.html) — Required by SPIR-V emitted for `layout(descriptor_heap)`.
