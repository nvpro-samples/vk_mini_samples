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


#include "dh_bindings.h"

#include "common/shaders/sky.hlsli"
#include "common/shaders/ggx.hlsli"
#include "common/shaders/constants.hlsli"
#include "common/shaders/random.hlsli"

#define WORKGROUP_SIZE 16

// Bindings
[[vk::binding(B_tlas)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage)]] RWTexture2D<float4> outImage;


//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  float2 launchID = (float2)threadIdx.xy;
  
  uint2 imgSize;
  outImage.GetDimensions(imgSize.x, imgSize.y); //DispatchRaysDimensions();
  float2 launchSize = imgSize;

  if(launchID.x >= launchSize.x || launchID.y >= launchSize.y)
    return;
  
  // Creating the ray (Orthographic camera, positioned at 0,0,5, looking at 0,0,-1, with a width of 2.0)
  float FOV = 2.0;
  float2 inUV = float2(launchID) / float2(launchSize);
  float aspectRatio = float(launchSize.y) / float(launchSize.x);
  float2 d = (inUV * 2.0 - 1.0) * FOV;
  float3 origin = float3(d.x, d.y * aspectRatio, 5);
  float3 direction = float3(0, 0, -1);

  RayDesc ray;
  ray.Direction = direction;
  ray.Origin = origin;
  ray.TMin = 0;
  ray.TMax = INFINITE;
  
  RayQuery < RAY_FLAG_NONE > q;
  q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
  while(q.Proceed())
  {
    if(q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
      q.CommitNonOpaqueTriangleHit(); // forcing to be opaque
  }
  
  float3 pixel_color = float3(0, 0, 0); // The return value
  
  if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
  {
    float2 barycentricCoords = q.CommittedTriangleBarycentrics();
    float3x4 worldToObject = q.CommittedWorldToObject3x4();
    float3x4 objectToWorld = q.CommittedObjectToWorld3x4();

    float3 barycentrics = float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

    // #FETCH
    float3 positions[3] = q.CommittedGetIntersectionTriangleVertexPositions();
    
    const float3 geoNormal = normalize(cross(positions[1] - positions[0], positions[2] - positions[0]));
    float3 worldGeoNormal = normalize(mul(geoNormal, worldToObject).xyz);
    
    // Gold Material
    float3 gold_basecolor = float3(1.0, 0.84, 0.0);
    float gold_metallic = 1.0;
    float gold_roughness = 0.2;

    // Lights
    float3 light_dir[2] = {float3(1, -1, 1), float3(-1, 1, 1)};
    float light_intensty[2] = {1.0, 0.5};

    // Contribution
    for(int l = 0; l < 2; l++)
    {
      pixel_color += ggxEvaluate(-direction, worldGeoNormal, light_dir[l], gold_basecolor, gold_metallic, gold_roughness)
                     * light_intensty[l];
    }
  }
  else
  {
    pixel_color = float3(0.1, 0.1, 0.15); // Hitting the environment
  }
  
  outImage[int2(launchID)] = float4(pixel_color, 1.0);
}

