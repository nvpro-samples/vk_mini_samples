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


#define INSPECTOR_MODE_FRAGMENT
#define INSPECTOR_DESCRIPTOR_SET 0
#define INSPECTOR_INSPECTION_DATA_BINDING eFragInspectorData
#define INSPECTOR_METADATA_BINDING eFragInspectorMeta
#include "nvvkhl/shaders/dh_inspector.h"

layout(location = 0) in vec2 inUv;
layout(location = 1) in flat int inParticleID;
layout(location = 0) out vec4 outColor;

layout(binding = eFrameInfo, scalar) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

layout(binding = eParticles, scalar) buffer Particles_
{
  Particle particles[];
};


// utility for temperature
float fade(float low, float high, float value)
{
  float mid   = (low + high) * 0.5;
  float range = (high - low) * 0.5;
  float x     = 1.0 - clamp(abs(mid - value) / range, 0.0, 1.0);
  return smoothstep(0.0, 1.0, x);
}

// Return a cold-hot color based on intensity [0-1]
vec3 temperature(float intensity)
{
  const vec3 blue   = vec3(0.0, 0.0, 1.0);
  const vec3 cyan   = vec3(0.0, 1.0, 1.0);
  const vec3 green  = vec3(0.0, 1.0, 0.0);
  const vec3 yellow = vec3(1.0, 1.0, 0.0);
  const vec3 red    = vec3(1.0, 0.0, 0.0);

  vec3 color = (fade(-0.25, 0.25, intensity) * blue    //
                + fade(0.0, 0.5, intensity) * cyan     //
                + fade(0.25, 0.75, intensity) * green  //
                + fade(0.5, 1.0, intensity) * yellow   //
                + smoothstep(0.75, 1.0, intensity) * red);
  return color;
}

#define M_PI_OVER_2 1.570796
void main()
{
  vec2  centerOffset       = (inUv - 0.5) * 2.0;
  float sqrDstFromDistance = dot(centerOffset, centerOffset);
  float circleAlpha        = cos(sqrDstFromDistance * sqrDstFromDistance * M_PI_OVER_2);
  float delta              = fwidth(sqrt(sqrDstFromDistance));
  circleAlpha              = min(circleAlpha, 1 - smoothstep(1 - delta, 1 + delta, sqrDstFromDistance));

  float maxVelocity = 0.5 / frameInfo.scale;
  float velocity    = length(particles[inParticleID].velocity);
  vec3  col         = clamp(temperature(velocity / maxVelocity), 0, 1);

  // #INSPECTOR
  inspect32BitValue(0, floatBitsToUint(velocity));

  outColor = vec4(col.xyz, circleAlpha);
}