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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#version 450

#extension GL_GOOGLE_include_directive : enable

#include "device_host.h"

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 outFragPos;

layout(set = 0, binding = 0) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

layout(push_constant) uniform PushConstant_
{
  PushConstant pushC;
};

void main()
{
  vec4 pos    = pushC.transfo * vec4(inPosition.xyz, 1.0);
  gl_Position = frameInfo.proj * frameInfo.view * vec4(pos);
  outFragPos = pos.xyz;
}
