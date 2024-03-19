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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */
#version 450

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable


#include "device_host.h"

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;

out gl_PerVertex
{
  vec4 gl_Position;
};

layout(location = 0) out vec2 outUv;
layout(location = 1) out flat int outParticleID;

layout(binding = eFrameInfo, scalar) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

layout(binding = eParticles, scalar) buffer Particles_
{
  Particle particles[];
};


void main()
{
  outUv         = inUV;
  outParticleID = gl_InstanceIndex;

  vec2 particle = inPos.xy * frameInfo.radius + (particles[gl_InstanceIndex].position * frameInfo.scale * 2);
  gl_Position   = frameInfo.proj * vec4(particle, 0.0, 1.0);
}
