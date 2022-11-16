# KTX Textures

![](docs/image_ktx.png)

This sample shows how to use KTX textures with nvpro_core helpers, and applying tonemapper on the final frame as post-process.

## Resources

![](docs/KTX_100px_Sep20.png)

* Khronos KTX: https://www.khronos.org/ktx/
* Specification:  https://github.khronos.org/KTX-Specification
* NVIDIA Texture Tools Exporter:  https://developer.nvidia.com/nvidia-texture-tools-exporter
    * [Standalone Application](https://developer.nvidia.com/designworks/texture-tools-for-photoshop/secure/2021.2.0/NVIDIA_Texture_Tools_2021.2.0.exe)
    * [Adobe Photoshpp Plugin](https://developer.nvidia.com/designworks/texture-tools-for-photoshop/secure/2021.2.0/NVIDIA_Texture_Tools_for_Adobe_Photoshop_2021.2.0.exe)
* KTX-Software: https://github.com/KhronosGroup/KTX-Software

## TextureKtx 

Most of the texture creation and habdling is done in the `TextureKtx` class, more specifically in the `create(ktximage)` function. The texture and all its mipmaps are uploaded and the format of the image is kept. This means that the texture can be sRGB, and a tonemapper is required to see properly the image.

## The application

In `main()`, the `VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME` extension has been added because the tonemapper uses it. 

In `onAttach()`, the default image is loaded, searching in the default search paths. The scene is created which is a simple sphere, and Vulkan buffers are created containing the vertices and indices of the "scene". The pipepline is created, using a simple vertex and fragment shader with basic shading.

:warning: In the fragment shader we could have corrected the color (Gamma correction), but now there are many more shaders and the final image can be composed by many operations, all done in linear. Therefore, adding a tonemapper at the end of the process is a better way to do things.

In `onRender()` the camera matrices will be updated in the `m_frameInfo.buffer`, buffer that is used in the vertex shader. The scene is then rendered, follwing by the post-process, which is applying the tonemapper. 

## Tonemapper

The tonemmaper is using the `TonemapperPostProcess` class, which is implemeting the post-process in graphic and compute. For graphic, this means it will draw a quad covering the viewport and uses the incoming and final image to draw with a tonemapper applied. In compute, the incoming image is processed and wrote in an outgoing image, which can be the same. Choosing one methode over the other depends on the application.

See in the application `use_compute` to switch between the two methods.

## Extra dependencies

The following libraries have beed added to [nvpro-core third party libraries](https://github.com/nvpro-samples/nvpro_core/tree/master/third_party).


* **[Zstandard Supercompression](https://github.com/facebook/zstd)**: Zstandard, or zstd as short version, is a fast lossless compression algorithm.
* **[Basis Universal](https://github.com/BinomialLLC/basis_universal)**: Basis Universal Supercompressed GPU Texture Codec
 
See in Makefile 

```
# KTX needs extra libraries
_add_package_KTX()
target_link_libraries (${PROJECT_NAME} libzstd_static zlibstatic basisu)
```





