# KTX Textures
![](docs/KTX_100px_Sep20.png)

This sample shows how to use KTX textures included in the glTF scene.

## Resources

* Khronos KTX: https://www.khronos.org/ktx/
* Specification:  https://github.khronos.org/KTX-Specification
* NVIDIA Texture Tools Exporter:  https://developer.nvidia.com/nvidia-texture-tools-exporter
    * [Standalone Application](https://developer.nvidia.com/designworks/texture-tools-for-photoshop/secure/2021.2.0/NVIDIA_Texture_Tools_2021.2.0.exe)
    * [Adobe Photoshpp Plugin](https://developer.nvidia.com/designworks/texture-tools-for-photoshop/secure/2021.2.0/NVIDIA_Texture_Tools_for_Adobe_Photoshop_2021.2.0.exe)
* KTX-Software: https://github.com/KhronosGroup/KTX-Software




## Preparation 

By default, tiny_gltf is loading textures, but it only supports the type that [stb_image](https://github.com/nothings/stb) is supporting. To enable other image type, we need to disable the loading of those files.

Add to the very top of the `vulkan_sample.cpp` 

```` CPP
#define TINYGLTF_NO_EXTERNAL_IMAGE
#include "fileformats/nv_ktx.h"
```` 

Then in `VulkanSample::loadScene()` we remove the image loader from tiny_gltf.

```` CPP
  // #KTX
  tcontext.RemoveImageLoader();
  if(!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename))
````


In `VulkanSample::createTextureImages()` we will change the way create images, by a function that will load and create the images.

```` CPP
  // First - create the images
  m_images.reserve(gltfModel.images.size());
  for(auto& image : gltfModel.images)
  {
    // #KTX
    if(loadCreateImage(cmdBuf, basedir, image) == false)
    {
      addDefaultImage({255, 0, 255, 255});  // Image not present or incorrectly loaded (image.empty)
      continue;
    }
  }
````

## Loading Texture

In the bottom of [vulkan_sample.cpp](src/vulkan_sample.cpp), look for the function `VulkanSample::loadCreateImage()`.

The first part loads the KTX images, creates them with their mipmaps. Note that not all formats in KTX2 are supported directly on NVIDIA hardware. For specific ones, we simply return an error which will be display as a uniform magenta color. 

KTX2 format can be in sRGB or Unorm. By setting sRGB to the image format, the hardware will convert the image to linear space and this conversion will not be needed in the shader. 

To make the support for this we are adding:
* `int isSrgb` to `FrameInfo (host_device.h)`
* And early return in `srgbToLinear() (pbr_gltf.glsl)` to skip to manual conversion of sRGB to linear space.


The second part of the load function loads all the other format, like `.jpg`, `.png`. This is using `stb_image`.




## Extra dependencies

The following libraries have beed added to [nvpro-core third party libraries](https://github.com/nvpro-samples/nvpro_core/tree/master/third_party).


* **[Zstandard Supercompression](https://github.com/facebook/zstd)**: Zstandard, or zstd as short version, is a fast lossless compression algorithm.
* **[Basis Universal](https://github.com/BinomialLLC/basis_universal)**: Basis Universal Supercompressed GPU Texture Codec
 
Headers for reading:





