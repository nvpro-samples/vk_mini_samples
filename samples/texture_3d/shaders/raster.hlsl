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

#include "device_host.h"
#include "functions.hlsli"

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
  PSin stage;
  float4 sv_position : SV_Position;
};

// Output of the fragment shader
struct PSout
{
  float4 color : SV_Target;
};


[[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;
[[vk::binding(0, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(1)]] [[vk::combinedImageSampler]] Texture3D g_Volume;
[[vk::binding(1)]] [[vk::combinedImageSampler]] SamplerState g_Sampler;

struct Sampler3D
{
  Texture3D t;
  SamplerState s;
};

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

// Determines whether a ray intersects with a bounding box (bbox) representing a cube.
// If an intersection occurs, it calculates the two points of intersection and returns
// them as output.
// The function returns a boolean value indicating whether an intersection took place.
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
  float tFar = min(min(t2.x, t2.y), t2.z);

  
  if (tNear > tFar || tFar < 0.0)
  {
    return false;
  }

  p1 = ray.origin + ray.direction * max(tNear, 0.0);
  p2 = ray.origin + ray.direction * tFar;

  return true;
}

// Computes the gradient of a 3D volume at a given position by sampling neighboring voxels and estimating
// the rate of change in each direction using finite differences. The resulting gradient vector represents
// the direction (normal) and magnitude of the steepest ascent/descent in the volume at that position.
float3 computeVolumeGradient(Sampler3D volume, float3 p, float voxelSize)
{
  float inc = voxelSize * 0.5F;
  float dx = (volume.t.Sample(volume.s, p - float3(inc, 0, 0)).r - volume.t.Sample(volume.s, p + float3(inc, 0, 0)).r) / voxelSize;
  float dy = (volume.t.Sample(volume.s, p - float3(0, inc, 0)).r - volume.t.Sample(volume.s, p + float3(0, inc, 0)).r) / voxelSize;
  float dz = (volume.t.Sample(volume.s, p - float3(0, 0, inc)).r - volume.t.Sample(volume.s, p + float3(0, 0, inc)).r) / voxelSize;

  return normalize(float3(dx, dy, dz));
}
    
// Traces a ray through a volume by taking multiple steps and sampling the volume texture at each step.
// It stops when the sampled value exceeds a threshold, and then performs interpolation to refine the hit point.
// The function returns a boolean indicating if a hit point was found and outputs the final hit point position.
bool rayMarching(const Sampler3D volume, const float3 p1, const float3 p2, const int numSteps, const float threshold, out float3 hitPoint)
{
  const float3 stepSize = (p2 - p1) / float(numSteps);
  hitPoint = p1;

  float3 prevPoint = hitPoint;
  float value = volume.t.Sample(volume.s, hitPoint).r;
  float prevValue = value;

  for (int i = 0; i < numSteps; ++i)
  {
    if (value > threshold)
    {
      float t = clamp((threshold - prevValue) / (value - prevValue), 0.0, 1.0);
      hitPoint = lerp(prevPoint, hitPoint, t);
      return true;
    }

    prevValue = value;
    prevPoint = hitPoint;
    hitPoint += stepSize;
    value = volume.t.Sample(volume.s, hitPoint).r;
  }

  return false;
}

// Vertex  Shader
[shader("vertex")]
VSout vertexMain(VSin input)
{
  float4 pos = mul(pushConst.transfo, float4(input.position.xyz, 1.0));

  VSout output;
  output.sv_position = mul(frameInfo.proj, mul(frameInfo.view, float4(pos)));
  output.stage.position = pos.xyz;

  return output;
}


// Fragment Shader
[shader("pixel")]
PSout fragmentMain(PSin stage, bool isFrontFacing : SV_IsFrontFace)
{
  PSout output;
  
  Ray ray;
  ray.origin = frameInfo.camPos;
  ray.direction = normalize(stage.position - frameInfo.camPos);

  // Intersection against the cube
  float3 p1, p2;
  Bbox bbox = { float3(-0.5, -0.5, -0.5), float3(0.5, 0.5, 0.5) };
  bool hit = intersectCube(ray, bbox, p1, p2);
  if (!hit)
    discard;

  // Avoid drawing triangle that can't be seen
  bool isInside = all(p1 == ray.origin);
  if ((isInside && isFrontFacing) || (!isInside && !isFrontFacing))
    discard;
  
  // Uniform position [0,1] for sampling in the volume
  p1 = p1 - bbox.bMin / (bbox.bMax - bbox.bMin);
  p2 = p2 - bbox.bMin / (bbox.bMax - bbox.bMin);

  Sampler3D volume;
  volume.t = g_Volume;
  volume.s = g_Sampler;

  // Ray-marching
  float3 hitPoint;
  hit = rayMarching(volume, p1, p2, pushConst.steps, pushConst.threshold, hitPoint);
  if (!hit)
    discard;

  // Find normal at position
  uint3 dim;
  uint levels;
  volume.t.GetDimensions(0, dim.x, dim.y, dim.z, levels);
  float3 normal = computeVolumeGradient(volume, hitPoint, 1.0 / dim.x);
  if (dot(ray.direction, normal) > 0)  // Make nornal pointing toward origin
    normal *= -1;

  float3 lightDir = (frameInfo.headlight == 1) ? -ray.direction : normalize(frameInfo.toLight);
  float3 V = -ray.direction;
  output.color.xyz =  simpleShading(V, lightDir, normal, pushConst.color.xyz);
  output.color.w = 1;
  
  return output;
}
