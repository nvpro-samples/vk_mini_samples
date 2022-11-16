/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 450

#extension GL_EXT_debug_printf : enable
#extension GL_GOOGLE_include_directive : enable

#include "device_host.h"


layout(location = 0) in vec3 inFragColor;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PushConstant_
{
  PushConstant pushC;
};


void main()
{
  ivec2 fragCoord = ivec2(floor(gl_FragCoord.xy));

  if(fragCoord == ivec2(pushC.mouseCoord))
    debugPrintfEXT("\n[%d, %d] Color: %f, %f, %f\n", fragCoord.x, fragCoord.y, inFragColor.x, inFragColor.y, inFragColor.z);

  outColor = vec4(inFragColor, 1.0);
}