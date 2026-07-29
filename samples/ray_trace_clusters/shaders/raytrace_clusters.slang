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

// Ray tracing a mesh whose bottom-level AS is built from Cluster Acceleration
// Structures (CLAS). On a hit, we read the per-cluster ID exposed by
// VK_NV_cluster_acceleration_structure and use it to colorize the surface, so
// the cluster decomposition is directly visible.

#include "shaderio.h"

#define INFINITE 1e32

// --- Resources (classic descriptor set, shared layout with the GLSL variant) ---
[[vk::binding(B_tlas)]]
RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage)]]
RWTexture2D<float4> outImage;
[[vk::binding(B_frameInfo)]]
ConstantBuffer<FrameInfo> frameInfo;

[[vk::push_constant]]
ConstantBuffer<PushConstant> pushConst;

// Read the per-hit cluster ID built-in (ClusterIDNV) from
// SPV_NV_cluster_acceleration_structure. Emitted as inline SPIR-V because Slang has no
// dedicated intrinsic for it outside of HitObject. Requires the pipeline to be created
// with allowClusterAccelerationStructure=VK_TRUE.
int getClusterID()
{
  return spirv_asm
  {
    OpCapability RayTracingClusterAccelerationStructureNV;
    OpExtension "SPV_NV_cluster_acceleration_structure";
  result:
    $$int = OpLoad builtin(ClusterIDNV: int);
  };
}

struct Payload
{
  float3 color;
};

// Hash a cluster ID to a distinct, pleasant color (IQ cosine palette).
float3 clusterColor(uint id)
{
  float  t = frac(float(id) * 0.6180339887);  // golden-ratio spread in [0,1)
  float3 a = float3(0.5, 0.5, 0.5);
  float3 b = float3(0.5, 0.5, 0.5);
  float3 c = float3(1.0, 1.0, 1.0);
  float3 d = float3(0.00, 0.33, 0.67);
  return a + b * cos(6.2831853 * (c * t + d));
}

[shader("raygeneration")]
void rgenMain()
{
  uint2 launchID   = DispatchRaysIndex().xy;
  uint2 launchSize = DispatchRaysDimensions().xy;

  const float2 clipCoords = (float2(launchID) + 0.5) / float2(launchSize) * 2.0 - 1.0;
  const float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), frameInfo.projInv);

  RayDesc ray;
  ray.Origin    = mul(float4(0.0, 0.0, 0.0, 1.0), frameInfo.viewInv).xyz;
  ray.Direction = mul(float4(normalize(viewCoords.xyz), 0.0), frameInfo.viewInv).xyz;
  ray.TMin      = 0.001;
  ray.TMax      = INFINITE;

  Payload payload;
  payload.color = float3(0.0);
  TraceRay(topLevelAS, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

  outImage[int2(launchID)] = float4(payload.color, 1.0);
}

[shader("closesthit")]
void rchitMain(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  // Which cluster did we hit?
  int clusterID = getClusterID();

  // The mesh is a unit sphere centered at the origin: the surface normal is
  // simply the normalized hit position. This avoids any vertex/normal buffers.
  float3 hitPos = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
  float3 normal = normalize(hitPos);

  float3 base = (pushConst.colorByCluster != 0) ? clusterColor(uint(clusterID)) : float3(0.8, 0.8, 0.8);

  // Slight per-triangle brightness variation (triangle index within the cluster) so the
  // individual triangles that make up a cluster are visible.
  uint  triID  = PrimitiveIndex();
  float triVar = 0.8 + 0.2 * frac(float(triID) * 0.6180339887);

  float diffuse = max(dot(normal, normalize(pushConst.lightDir)), 0.0);
  payload.color = base * triVar * (0.15 + 0.85 * diffuse);
}

[shader("miss")]
void rmissMain(inout Payload payload)
{
  payload.color = float3(0.10, 0.11, 0.13);  // background
}
