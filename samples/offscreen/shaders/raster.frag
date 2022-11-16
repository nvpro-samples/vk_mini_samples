/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460

#extension GL_GOOGLE_include_directive : enable
#include "device_host.h"

layout(location = 0) in vec2 outUV;
layout(location = 0) out vec4 fragColor;


layout(push_constant) uniform PushConstant_
{
  PushConstant pushC;
};

void main()
{
  vec2 uv = outUV;

  uv.x *= pushC.aspectRatio;

  // Time varying pixel color
  vec3 col = 0.5 + 0.5 * cos(pushC.iTime + uv.xyx + vec3(0, 2, 4));

  for(int i = 0; i < 2; ++i)
  {
    uv -= 0.5;
    uv *= 5.;

    float radius = .5 + (sin(pushC.iTime * .3) * .15) - cos(pushC.iTime) * float(i) * .1;
    vec2  fuv    = fract(uv) - .5;
    float x      = smoothstep(radius * .9, radius * 1.1, length(fuv));
    float y      = 1.0 - smoothstep(radius * .3, radius * .7, length(fuv));

    col -= vec3(x + y);
  }

  // Output to screen
  fragColor = vec4(col, 1.0);
}
