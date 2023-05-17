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
#extension GL_EXT_scalar_block_layout : enable

#include "device_host.h"

layout(location = 0) in vec3 inFragPos;
layout(location = 0) out vec4 outColor;


layout(set = 0, binding = 0, scalar) uniform FrameInfo_
{
  FrameInfo frameInfo;
};
layout(set = 0, binding = 1) uniform sampler3D inVolume;
layout(push_constant) uniform PushConstant_
{
  PushConstant pushC;
};

// Ray structure
struct Ray
{
  vec3 origin;
  vec3 direction;
};

struct Bbox
{
  vec3 bMin;
  vec3 bMax;
};

// Performs shading calculations for a 3D surface point, taking into account light direction, view direction,
// surface color, and surface normal. It combines diffuse and specular components to determine the light
// intensity, multiplies it with the surface color, adds an ambient term (sky), and returns the final shaded color.
vec3 calculateShading(in vec3 surfaceColor, in vec3 viewDirection, in vec3 surfaceNormal, in vec3 lightDirection)
{
  vec3 shadedColor             = surfaceColor;
  vec3 worldUpDirection        = vec3(0, 1, 0);
  vec3 reflectedLightDirection = normalize(-reflect(lightDirection, surfaceNormal));

  // Diffuse + Specular
  float lightIntensity =
      max(dot(surfaceNormal, lightDirection) + pow(max(0, dot(reflectedLightDirection, viewDirection)), 16.0), 0);
  shadedColor *= lightIntensity;

  // Ambient term (sky effect)
  vec3 skyAmbientColor = mix(vec3(0.1, 0.1, 0.4), vec3(0.8, 0.6, 0.2), dot(surfaceNormal, worldUpDirection.xyz) * 0.5 + 0.5) * 0.2;
  shadedColor += skyAmbientColor;

  return shadedColor;
}

// Determines whether a ray intersects with a bounding box (bbox) representing a cube.
// If an intersection occurs, it calculates the two points of intersection and returns
// them as output.
// The function returns a boolean value indicating whether an intersection took place.
bool intersectCube(in Ray ray, in Bbox bbox, out vec3 p1, out vec3 p2)
{
  vec3 invDir = 1.0 / ray.direction;

  vec3 tMin = (bbox.bMin - ray.origin) * invDir;
  vec3 tMax = (bbox.bMax - ray.origin) * invDir;

  vec3 t1 = min(tMin, tMax);
  vec3 t2 = max(tMin, tMax);

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

// Computes the gradient of a 3D volume at a given position by sampling neighboring voxels and estimating
// the rate of change in each direction using finite differences. The resulting gradient vector represents
// the direction (normal) and magnitude of the steepest ascent/descent in the volume at that position.
vec3 computeVolumeGradient(sampler3D volume, vec3 p, float voxelSize)
{
  float inc = voxelSize * 0.5F;
  float dx  = (texture(volume, p - vec3(inc, 0, 0)).r - texture(volume, p + vec3(inc, 0, 0)).r) / voxelSize;
  float dy  = (texture(volume, p - vec3(0, inc, 0)).r - texture(volume, p + vec3(0, inc, 0)).r) / voxelSize;
  float dz  = (texture(volume, p - vec3(0, 0, inc)).r - texture(volume, p + vec3(0, 0, inc)).r) / voxelSize;

  return normalize(vec3(dx, dy, dz));
}

// Traces a ray through a volume by taking multiple steps and sampling the volume texture at each step.
// It stops when the sampled value exceeds a threshold, and then performs interpolation to refine the hit point.
// The function returns a boolean indicating if a hit point was found and outputs the final hit point position.
bool rayMarching(const sampler3D volume, const vec3 p1, const vec3 p2, const int numSteps, const float threshold, out vec3 hitPoint)
{
  const vec3 stepSize = (p2 - p1) / float(numSteps);
  hitPoint            = p1;

  vec3  prevPoint = hitPoint;
  float value     = texture(volume, hitPoint).r;
  float prevValue = value;

  for(int i = 0; i < numSteps; ++i)
  {
    if(value > threshold)
    {
      float t  = clamp((threshold - prevValue) / (value - prevValue), 0.0, 1.0);
      hitPoint = mix(prevPoint, hitPoint, t);
      return true;
    }

    prevValue = value;
    prevPoint = hitPoint;
    hitPoint += stepSize;
    value = texture(volume, hitPoint).r;
  }

  return false;
}


void main()
{
  Ray ray;
  ray.origin    = frameInfo.camPos;
  ray.direction = normalize(inFragPos - frameInfo.camPos);

  // Intersection against the cube
  vec3 p1, p2;
  Bbox bbox = Bbox(vec3(-0.5), vec3(0.5));
  bool hit  = intersectCube(ray, bbox, p1, p2);
  if(!hit)
    discard;

  // Avoid drawing triangle that can't be seen
  bool isInside = (p1 == ray.origin);
  if((isInside && gl_FrontFacing) || (!isInside && !gl_FrontFacing))
    discard;


  // Uniform position [0,1] for sampling in the volume
  p1 = p1 - bbox.bMin / (bbox.bMax - bbox.bMin);
  p2 = p2 - bbox.bMin / (bbox.bMax - bbox.bMin);

  // Ray-marching
  vec3 hitPoint;
  hit = rayMarching(inVolume, p1, p2, pushC.steps, pushC.threshold, hitPoint);
  if(!hit)
    discard;

  // Find normal at position
  vec3 normal = computeVolumeGradient(inVolume, hitPoint, 1.0 / textureSize(inVolume, 0).x);
  if(dot(ray.direction, normal) > 0)  // Make nornal pointing toward origin
    normal *= -1;

  vec3 toLight = (frameInfo.headlight == 1) ? -ray.direction : normalize(frameInfo.toLight);
  vec3 color   = calculateShading(pushC.color.xyz, -ray.direction, normal, toLight);

  outColor = vec4(color, pushC.color.w);
}