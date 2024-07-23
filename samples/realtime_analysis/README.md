# Real-time GPU Analysis Tools in Vulkan

![Real-time Analysis](docs/realtime_analysis.png)

This implementation demonstrates non-intrusive GPU inspection tools for runtime analysis in Vulkan applications.

## Inspector

Enables real-time visualization of GPU data without knowledge of internal structures.

### Image Inspection
![Image Inspection](docs/inspect_image.png)

- Captures images at any rendering stage
- Example: GBuffer inspection with per-pixel value analysis

### Buffer Inspection
![Buffer Inspection](docs/inspect_buffer.png)

- Displays buffer contents
- Example: Particle buffer and sorting buffer inspection
- Requires buffer structure information during initialization

### Fragment Shader Inspection
![Fragment Inspection](docs/inspect_fragment.png)

- Live output of fragment shader variables
- Example: 2x2 pixel area around cursor with per-pixel value display

Implementation details marked with `#INSPECTOR` in source code.

## Profiler

![Profiler](docs/profiler.png)

- Measures GPU operation execution time
- Supports nested function profiling
- Example: `onRender` function timing breakdown

Implementation details marked with `#PROFILER` in source code.

## NVML Monitor

![NVML Monitor](docs/nvml.png)

Application-independent GPU monitoring tool providing:
- Real-time memory usage
- GPU utilization
- Additional performance metrics

Note: All tools operate directly on Vulkan objects for seamless integration.


