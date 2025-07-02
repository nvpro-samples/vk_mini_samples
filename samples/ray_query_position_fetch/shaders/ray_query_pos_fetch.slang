/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dh_bindings.h"

#include "nvshaders/bsdf_functions.h.slang"


// clang-format off
// Bindings
[[vk::binding(B_tlas)]]     RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage)]] RWTexture2D<float4> outImage;
// clang-format on

float3 ggxEvaluate(float3 V, float3 L, PbrMaterial mat)
{
  BsdfEvaluateData data;
  data.k1 = V;
  data.k2 = L;

  bsdfEvaluateSimple(data, mat);

  return data.bsdf_glossy + data.bsdf_diffuse;
}

//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx: SV_DispatchThreadID)
{
  float2 launchID = (float2)threadIdx.xy;

  uint2 imgSize;
  outImage.GetDimensions(imgSize.x, imgSize.y);  // DispatchRaysDimensions();
  float2 launchSize = imgSize;

  if(launchID.x >= launchSize.x || launchID.y >= launchSize.y)
    return;

  // Creating the ray (Orthographic camera, positioned at 0,0,5, looking at
  // 0,0,-1, with a width of 2.0)
  float  FOV         = 2.0;
  float2 inUV        = float2(launchID) / float2(launchSize);
  float  aspectRatio = float(launchSize.y) / float(launchSize.x);
  float2 d           = (inUV * 2.0 - 1.0) * FOV;
  float3 origin      = float3(d.x, d.y * aspectRatio, 5);
  float3 direction   = float3(0, 0, -1);

  RayDesc ray;
  ray.Direction = direction;
  ray.Origin    = origin;
  ray.TMin      = 0;
  ray.TMax      = 1e32;

  RayQuery<RAY_FLAG_NONE> q;
  q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
  while(q.Proceed())
  {
    if(q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
      q.CommitNonOpaqueTriangleHit();  // forcing to be opaque
  }

  float3 pixel_color = float3(0, 0, 0);  // The return value

  if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
  {
    float2   barycentricCoords = q.CommittedTriangleBarycentrics();
    float4x3 worldToObject     = q.CommittedWorldToObject4x3();

    float3 barycentrics = float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

    // #FETCH
    float3 positions[3] = q.CommittedGetIntersectionTriangleVertexPositions();

    const float3 geoNormal      = normalize(cross(positions[1] - positions[0], positions[2] - positions[0]));
    float3       worldGeoNormal = normalize(mul(worldToObject, geoNormal).xyz);

    // Gold Material
    float3 gold_basecolor = float3(1.0, 0.84, 0.0);
    float  gold_metallic  = 1.00;
    float  gold_roughness = 0.6;

    // Lights
    float3 light_dir[2]      = { float3(1, -1, 1), float3(-1, 1, 1) };
    float  light_intensty[2] = { 2.0, 0.5 };

    // Contribution
    for(int l = 0; l < 2; l++)
    {
      float3 V = normalize(-direction);
      float3 L = normalize(light_dir[l]);

      PbrMaterial mat = defaultPbrMaterial(gold_basecolor, gold_metallic, gold_roughness, worldGeoNormal, worldGeoNormal);
      pixel_color += ggxEvaluate(V, L, mat) * light_intensty[l];
    }
  }
  else
  {
    pixel_color = float3(0.1, 0.1, 0.15);  // Hitting the environment
  }

  outImage[int2(launchID)] = float4(pixel_color, 1.0);
}

