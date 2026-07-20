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

#include <nvshaders/simple_shading.h.slang>

#include "shaderio.h"

// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin
{
  float3 position : POSITION;
};

// Output of the vertex shader, and input to the fragment shader.
struct PSin
{
  float3 position : POSITION;
};

// Output of the vertex shader
struct VSout
{
  PSin   stage;
  float4 sv_position : SV_Position;
};

// Output of the fragment shader
struct PSout
{
  float4 color : SV_Target;
};


[[vk::push_constant]]
ConstantBuffer<PushConstant> pushConst;

// Ray structure
struct Ray
{
  float3 origin;
  float3 direction;
};

struct Bbox
{
  float3 bMin;
  float3 bMax;
};

bool intersectCube(in Ray ray, in Bbox bbox, out float3 p1, out float3 p2)
{
  p1 = float3(0.0, 0.0, 0.0);
  p2 = float3(0.0, 0.0, 0.0);

  float3 invDir = 1.0 / ray.direction;

  float3 tMin = (bbox.bMin - ray.origin) * invDir;
  float3 tMax = (bbox.bMax - ray.origin) * invDir;

  float3 t1 = min(tMin, tMax);
  float3 t2 = max(tMin, tMax);

  float tNear = max(max(t1.x, t1.y), t1.z);
  float tFar  = min(min(t2.x, t2.y), t2.z);

  if(tNear > tFar || tFar < 0.0)
  {
    return false;
  }

  p1 = ray.origin + ray.direction * max(tNear, 0.0);
  p2 = ray.origin + ray.direction * tFar;

  return true;
}

float3 computeVolumeGradient(Texture3D volume, SamplerState volumeSampler, float3 p, float voxelSize)
{
  float inc = voxelSize * 0.5F;
  float dx =
      (volume.Sample(volumeSampler, p - float3(inc, 0, 0)).r - volume.Sample(volumeSampler, p + float3(inc, 0, 0)).r) / voxelSize;
  float dy =
      (volume.Sample(volumeSampler, p - float3(0, inc, 0)).r - volume.Sample(volumeSampler, p + float3(0, inc, 0)).r) / voxelSize;
  float dz =
      (volume.Sample(volumeSampler, p - float3(0, 0, inc)).r - volume.Sample(volumeSampler, p + float3(0, 0, inc)).r) / voxelSize;

  return normalize(float3(dx, dy, dz));
}

bool rayMarching(Texture3D volume, SamplerState volumeSampler, const float3 p1, const float3 p2, const int numSteps, const float threshold, out float3 hitPoint)
{
  const float3 stepSize = (p2 - p1) / float(numSteps);
  hitPoint              = p1;

  float3 prevPoint = hitPoint;
  float  value     = volume.Sample(volumeSampler, hitPoint).r;
  float  prevValue = value;

  for(int i = 0; i < numSteps; ++i)
  {
    if(value > threshold)
    {
      float t  = clamp((threshold - prevValue) / (value - prevValue), 0.0, 1.0);
      hitPoint = lerp(prevPoint, hitPoint, t);
      return true;
    }

    prevValue = value;
    prevPoint = hitPoint;
    hitPoint += stepSize;
    value = volume.Sample(volumeSampler, hitPoint).r;
  }

  return false;
}

[require(spvDescriptorHeapEXT)]
[shader("vertex")]
VSout vertexMain(VSin input)
{
  FrameInfo frameInfo = *ConstantBuffer<FrameInfo>.Handle(uint2(pushConst.bufferHeapBase + kHeapBufFrameInfo, 0));

  float4 pos = mul(float4(input.position.xyz, 1.0), pushConst.transfo);

  VSout output;
  output.sv_position    = mul(mul(pos, frameInfo.view), frameInfo.proj);
  output.stage.position = pos.xyz;

  return output;
}


[require(spvDescriptorHeapEXT)]
[shader("pixel")]
PSout fragmentMain(PSin stage, bool isFrontFacing: SV_IsFrontFace)
{
  FrameInfo frameInfo = *ConstantBuffer<FrameInfo>.Handle(uint2(pushConst.bufferHeapBase + kHeapBufFrameInfo, 0));

  Texture3D    volume        = Texture3D.Handle(uint2(kHeapImgVolume, 0));
  SamplerState volumeSampler = SamplerState.Handle(uint2(kHeapSmpVolume, 0));

  PSout output;

  Ray ray;
  ray.origin    = frameInfo.camPos;
  ray.direction = normalize(stage.position - frameInfo.camPos);

  float3 p1, p2;
  Bbox   bbox = { float3(-0.5, -0.5, -0.5), float3(0.5, 0.5, 0.5) };
  bool   hit  = intersectCube(ray, bbox, p1, p2);
  if(!hit)
    discard;

  bool isInside = all(p1 == ray.origin);
  if((isInside && isFrontFacing) || (!isInside && !isFrontFacing))
    discard;

  p1 = p1 - bbox.bMin / (bbox.bMax - bbox.bMin);
  p2 = p2 - bbox.bMin / (bbox.bMax - bbox.bMin);

  float3 hitPoint;
  hit = rayMarching(volume, volumeSampler, p1, p2, pushConst.steps, pushConst.threshold, hitPoint);
  if(!hit)
    discard;

  uint3 dim;
  uint  levels;
  volume.GetDimensions(0, dim.x, dim.y, dim.z, levels);
  float3 normal = computeVolumeGradient(volume, volumeSampler, hitPoint, 1.0 / dim.x);
  if(dot(ray.direction, normal) > 0)
    normal *= -1;

  float3 lightDir  = (frameInfo.headlight == 1) ? -ray.direction : normalize(frameInfo.toLight);
  float3 V         = -ray.direction;
  output.color.xyz = simpleShading(V, lightDir, normal, pushConst.color.xyz, 16.0);
  output.color.w   = 1;

  return output;
}
