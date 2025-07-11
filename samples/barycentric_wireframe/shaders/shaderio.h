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

#include "nvshaders/slang_types.h"


struct PushConstant
{
  float4x4 transfo;
  float4   color;
  float4   clearColor;
};

#define BIND_FRAME_INFO 0
#define BIND_SETTINGS 1

struct FrameInfo
{
  float4x4 proj;
  float4x4 view;
  float3   camPos;
};

struct WireframeSettings
{
  float  thickness;       // Thickness of wireframe
  float3 color;           // Color
  float2 thicknessVar;    // Variation of edge thickness
  float  smoothing;       // Can be different
  int    screenSpace;     // Thickness in screen space
  float3 backFaceColor;   // Backface wire in different color
  int    enableStipple;   // Using dash lines
  int    stippleRepeats;  // How many repeats
  float  stippleLength;   // Length of each dash [0,1]
  int    onlyWire;        // Discard everything except wire
};
