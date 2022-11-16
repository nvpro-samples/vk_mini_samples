# Multi-Sampling Anti-Aliasing (MSAA)

![img](docs/msaa.png)

This sample is showing the usage of Multi-Sampling Anti-Aliasing (MSAA). 

## MSAA Image G-Buffers

The creation of multi-sampling buffers is done in `createMsaaBuffers()`. Those buffers will be re-created if the size of the image changes or the `VkSampleCountFlagBits`. 

### Images

The MSAA images are created using only the following usage for color and depth respectively:

```` C
color.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
depth.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
````

## Normal G-Buffer

There is alway a normal G-Buffer, an image without multi-sampling because multi-sampling textures can't be attached and displayed. The multi-sampled image will need to be resolved in the normal G-Buffer, and this is the image that will be displayed.

## Rendering and resolving MSAA

In `onRender(cmd)`, rendering is done using dynamic rendering. The G-Buffer is attached as the target and in the case of multi-sampling, the multi-sampled image is attached and the G-Buffer is attached as resoleved image. 


## References

Other Vulkan projects using MSAA

* <https://github.com/KhronosGroup/Vulkan-Samples/blob/master/samples/performance/msaa/msaa_tutorial.md>
* <https://vulkan-tutorial.com/Multisampling>
* <https://github.com/SaschaWillems/Vulkan/blob/master/examples/multisampling/multisampling.cpp>
