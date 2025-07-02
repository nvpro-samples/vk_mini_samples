/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_ray_tracing_position_fetch : require  // #FETCH

#include "dh_bindings.h"

#include "nvshaders/bsdf_functions.h.slang"

// GROUP_SIZE 16  // <-- defined in dh_bindings.h

layout(local_size_x = WORKGROUP_SIZE, local_size_y = WORKGROUP_SIZE) in;

layout(set = 0, binding = B_tlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = B_outImage, rgba32f) uniform image2D image;

vec3 ggxEvaluate(vec3 V, vec3 L, PbrMaterial mat)
{
  BsdfEvaluateData data;
  data.k1 = V;
  data.k2 = L;

  bsdfEvaluateSimple(data, mat);

  return data.bsdf_glossy + data.bsdf_diffuse;
}


void main()
{
  ivec2 launchID   = ivec2(gl_GlobalInvocationID.xy);
  ivec2 launchSize = imageSize(image);

  // Check if not outside boundaries
  if(launchID.x >= launchSize.x || launchID.y >= launchSize.y)
    return;

  // Creating the ray (Orthographic camera, positioned at 0,0,5, looking at 0,0,-1, with a width of 2.0)
  float FOV         = 2.0;
  vec2  inUV        = vec2(launchID) / vec2(launchSize);
  float aspectRatio = float(launchSize.y) / float(launchSize.x);
  vec2  d           = (inUV * 2.0 - 1.0) * FOV;
  vec3  origin      = vec3(d.x, d.y * aspectRatio, 5);
  vec3  direction   = vec3(0, 0, -1);

  rayQueryEXT rayQuery;
  rayQueryInitializeEXT(rayQuery, topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, origin, 0.0, direction, 1e32);
  rayQueryProceedEXT(rayQuery);

  vec3 pixel_color = vec3(0, 0, 0);  // The return value
  if(rayQueryGetIntersectionTypeEXT(rayQuery, true) == gl_RayQueryCommittedIntersectionNoneEXT)
  {
    pixel_color = vec3(0.1, 0.1, 0.15);  // Hitting the environment
  }
  else
  {
    vec2   barycentricCoords = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    mat4x3 worldToObject     = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
    mat4x3 objectToWorld     = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);

    vec3 barycentrics = vec3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

    // #FETCH
    vec3 positions[3];
    rayQueryGetIntersectionTriangleVertexPositionsEXT(rayQuery, true, positions);
    const vec3 geoNormal      = normalize(cross(positions[1] - positions[0], positions[2] - positions[0]));
    vec3       worldGeoNormal = normalize(vec3(geoNormal * worldToObject));

    // Gold Material
    vec3  gold_basecolor = vec3(1.0, 0.84, 0.0);
    float gold_metallic  = 0.90;
    float gold_roughness = 0.2;

    // Lights
    vec3  light_dir[2]      = {vec3(1, -1, 1), vec3(-1, 1, 1)};
    float light_intensty[2] = {1.0, 0.5};

    // Contribution
    for(int l = 0; l < 2; l++)
    {
      vec3 V = normalize(-direction);
      vec3 L = normalize(light_dir[l]);

      PbrMaterial mat = defaultPbrMaterial(gold_basecolor, gold_metallic, gold_roughness, worldGeoNormal, worldGeoNormal);
      pixel_color += ggxEvaluate(V, L, mat) * light_intensty[l];
    }
  }

  imageStore(image, launchID, vec4(pixel_color, 1.0F));
}
