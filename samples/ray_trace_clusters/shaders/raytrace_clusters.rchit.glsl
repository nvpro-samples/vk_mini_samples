/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_spirv_intrinsics : require
#extension GL_GOOGLE_include_directive : enable

#include "shaderio.h"

// Cluster ID built-in from SPV_NV_cluster_acceleration_structure.
// Requires the pipeline to be created with allowClusterAccelerationStructure=VK_TRUE.
//   decoration 11 = BuiltIn, built-in id 5436 = ClusterIDNV,
//   capability 5437 = RayTracingClusterAccelerationStructureNV
spirv_decorate(extensions = ["SPV_NV_cluster_acceleration_structure"], capabilities = [5437], 11, 5436) in int gl_ClusterIDNV_;

layout(location = 0) rayPayloadInEXT vec3 payloadColor;

layout(push_constant) uniform PushConstant_
{
  PushConstant pc;
};

// Hash a cluster ID to a distinct, pleasant color (IQ cosine palette).
vec3 clusterColor(uint id)
{
  float t = fract(float(id) * 0.6180339887);  // golden-ratio spread in [0,1)
  vec3  a = vec3(0.5);
  vec3  b = vec3(0.5);
  vec3  c = vec3(1.0);
  vec3  d = vec3(0.00, 0.33, 0.67);
  return a + b * cos(6.2831853 * (c * t + d));
}

void main()
{
  int clusterID = gl_ClusterIDNV_;

  // Unit sphere centered at the origin: surface normal is the normalized hit position.
  vec3 hitPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
  vec3 normal = normalize(hitPos);

  vec3 base = (pc.colorByCluster != 0) ? clusterColor(uint(clusterID)) : vec3(0.8);

  // Slight per-triangle brightness variation (triangle index within the cluster) so the
  // individual triangles that make up a cluster are visible.
  uint  triID  = uint(gl_PrimitiveID);
  float triVar = 0.8 + 0.2 * fract(float(triID) * 0.6180339887);

  float diffuse = max(dot(normal, normalize(pc.lightDir)), 0.0);
  payloadColor  = base * triVar * (0.15 + 0.85 * diffuse);
}
