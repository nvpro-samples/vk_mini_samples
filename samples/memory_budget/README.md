# Dynamic Memory
![screenshot](docs/mem_budget.png)]

This is a simple example of how to dynamically allocate memory during rendering. It also shows how to use the `VK_EXT_memory_budget` extension to query the memory budget of the GPU.

## Features

- Dynamic memory allocation
- Memory budget query
- Dealocation of memory if the budget is exceeded

## Description

The example will create a Menger sponge fractal with the placement for even more complex Menger sponge fractals. Then, in a separate thread, it will allocate memory for these meshes and render them. The memory allocation is done in a loop, and the memory budget is queried at each frame iteration. If the budget is exceeded, the memory is deallocated and the loop is stopped.

Since the memory budget is queried at each frame, the example will deallocate some of the mesh if its budget gets lower than what it is using. This can be seen when running two instances of the sample at the same time, one with a higher number of meshes than the other. The one with the focus will have more budget than the other, and the other will reduce the number of meshes.

Being in budget means that the memory used by the application is less than the budget. This is not the same as having free memory, because the budget is the amount of memory that the application can use, and it is not the same as free memory. This allows the application to run at full speed because it doesn't have to transfer memory from system memory to the device and back. This avoids stuttering and improves the performance of the application.

## Details

In the image above, 32 sponges are rendered. The memory usage is 4.29 GB and the memory budget is 5.4 GB. This allows 32 meshes to be rendered with a total of 49.14 million triangles. The application is in budget and no memory is released. If the memory budget is reduced to 4.3 GB, the application will release some of the meshes and reduce memory usage. Note that the memory budget will fluctuate depending on what other applications are running. Therefore, it is important to query the memory budget at every frame.

Memory allocation occurs in a separate thread to prevent blocking the rendering thread. A different queue family is used for memory allocation than for rendering. The memory is then transferred using a transfer queue to prevent blocking the rendering queue and allow it to continue rendering while the memory is being transferred.

Note: Each Menger sponge uses about 142MB of memory. This is a rough estimate and actual memory usage may vary as each Menger Sponge is randomly generated.