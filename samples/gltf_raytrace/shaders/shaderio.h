/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#include "nvshaders/slang_types.h"

#include "nvshaders/sky_io.h.slang"
#include "nvshaders/gltf_scene_io.h.slang"

#define WORKGROUP_SIZE 32

#define B_tlas 0
#define B_outImage 1
#define B_textures 2

// Camera info
struct CameraInfo
{
  float4x4 projInv;
  float4x4 viewInv;
};

// Push constant
struct PushConstant
{
  int                    maxDepth              = 5;       // Maximum depth of the ray
  int                    frame                 = 0;       // Frame number
  float                  fireflyClampThreshold = 10.f;    // Firefly clamp threshold
  int                    maxSamples            = 1;       // Maximum samples
  float2                 mouseCoord            = {0, 0};  // Mouse coordinates (use for debug)
  int                    environmentType       = 0;       // Environment type; 0: sky, 1: environment map
  CameraInfo*            cameraInfo;                      // Camera info
  SkyPhysicalParameters* skyParams;                       // Sky physical parameters
  GltfScene*             gltfScene;                       // GLTF sceneF
};


#endif  // HOST_DEVICE_H
