/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#include "nvshaders/slang_types.h"

#include "nvshaders/sky_io.h.slang"
#include "nvshaders/gltf_scene_io.h.slang"


NAMESPACE_SHADERIO_BEGIN()

#define WORKGROUP_SIZE 32

// Descriptor-heap slot layout, shared by the host (descriptor writes) and the shader (handle
// construction). The resource heap is laid out images first, then buffers; the absolute heap index
// is `regionBase + slot`, where the region bases are passed in the push constant.
static const uint kHeapImgOutput        = 0;  // image region: path-trace output (storage image / RWTexture2D)
static const uint kHeapImgHdr           = 1;  // image region: HDR environment (sampled image)
static const uint kHeapImgTexturesStart = 2;  // image region: first glTF texture (sampled image)
static const uint kHeapBufEnvSampling   = 0;  // buffer region: HDR importance-sampling alias table (SSBO)

// Sampler-heap slot layout. Slot 0 is the default sampler (used for the HDR/env map, the output
// image, and any texture whose glTF sampler is unspecified). glTF sampler `s` maps to slot
// `kHeapSmpSceneStart + s`, giving the shader a trivial `samplerIndex -> slot` mapping.
static const uint kHeapSmpDefault    = 0;   // default sampler (linear, repeat)
static const uint kHeapSmpSceneStart = 1;   // first glTF scene sampler
static const uint kMaxSamplers       = 16;  // reserved scene-sampler slots

// Camera info
struct CameraInfo
{
  float4x4 projInv;
  float4x4 viewInv;
};

// All constants that are passed to the shader via push constants.
struct PushConstant
{
  int                    maxDepth              = 5;       // Maximum depth of the ray
  int                    frame                 = 0;       // Frame number
  float                  fireflyClampThreshold = 10.f;    // Firefly clamp threshold
  int                    maxSamples            = 1;       // Maximum samples
  float2                 mouseCoord            = {0, 0};  // Mouse coordinates (use for debug)
  int                    environmentType       = 0;       // Environment type; 0: sky, 1: environment map
  uint                   imageHeapBase         = 0;       // Resource heap: base index of the image region
  uint                   bufferHeapBase        = 0;       // Resource heap: base index of the buffer region
  uint                   samplerHeapBase       = 0;       // Sampler heap: base index of the sampler region
  uint64_t               tlasAddress           = 0;       // TLAS device address (converted to an AS in-shader)
  CameraInfo*            cameraInfo;                      // Camera info
  SkyPhysicalParameters* skyParams;                       // Sky physical parameters
  GltfScene*             gltfScene;                       // GLTF scene
};


NAMESPACE_SHADERIO_END()

#endif  // HOST_DEVICE_H
