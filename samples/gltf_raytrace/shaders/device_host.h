/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

struct PushConstant
{
  int   maxDepth;
  int   frame;
  float fireflyClampThreshold;
  int   maxSamples;
};

struct CameraInfo
{
#if defined(__SLANG__) || (__HLSL_VERSION > 2020)
  float4x4 projInv;
  float4x4 viewInv;
#else
  mat4 projInv;
  mat4 viewInv;
#endif
};


#endif  // HOST_DEVICE_H
