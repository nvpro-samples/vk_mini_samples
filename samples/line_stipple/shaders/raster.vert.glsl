/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#version 450

#extension GL_EXT_scalar_block_layout : enable


layout(location = 0) in vec3 inPosition;
layout(location = 0) out vec4 outColor;

layout(push_constant, scalar) uniform PushConstant_
{
  vec3  color;
  float rotate;
  vec2  translate;
  int   _pad0;
  int   _pad1;
}
pushConstants;

void main()
{
  vec2 pos = inPosition.xy;
  pos.y *= -1.0;  // Flip the y coordinate
  // Translate and rotate the vertex position
  float angle = pushConstants.rotate;
  mat2  rot   = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
  pos         = rot * pos;
  pos += pushConstants.translate;

  gl_Position = vec4(pos, inPosition.z, 1.0);
  outColor    = vec4(pushConstants.color, 1.0);
}
