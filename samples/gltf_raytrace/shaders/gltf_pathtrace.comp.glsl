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

#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_buffer_reference2 : require

#extension GL_NV_shader_invocation_reorder : enable


const int GROUP_SIZE = 16;
layout(local_size_x = GROUP_SIZE, local_size_y = GROUP_SIZE) in;

#include "device_host.h"
#include "dh_bindings.h"
#include "nvvkhl/shaders/bsdf_functions.h"
#include "nvvkhl/shaders/bsdf_structs.h"
#include "nvvkhl/shaders/constants.glsl"
#include "nvvkhl/shaders/dh_lighting.h"
#include "nvvkhl/shaders/dh_scn_desc.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/ggx.glsl"
#include "nvvkhl/shaders/light_contrib.glsl"
#include "nvvkhl/shaders/pbr_mat_struct.h"
#include "nvvkhl/shaders/random.glsl"
#include "nvvkhl/shaders/ray_util.glsl"


// clang-format off
layout(buffer_reference, scalar) readonly buffer PrimMeshInfoBuf { PrimMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer GltfMaterialBuf { GltfShadeMaterial m[]; };
layout(buffer_reference, scalar) readonly buffer Vertices { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uvec3 i[]; };

layout(set = 0, binding = B_tlas)               uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = B_outImage, rgba32f)  uniform image2D           image;
layout(set = 0, binding = B_cameraInfo, scalar) uniform cameraInfo_       { CameraInfo cameraInfo; };
layout(set = 0, binding = B_sceneDesc, scalar) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 0, binding = B_textures)           uniform sampler2D         texturesMap[]; // all textures
layout(set = 0, binding = B_skyParam,  scalar)  uniform SkyInfo_          { ProceduralSkyShaderParameters skyInfo; };
// clang-format on

#include "nvvkhl/shaders/pbr_mat_eval.glsl"  // Need texturesMap[]

layout(push_constant, scalar) uniform RtxPushConstant_
{
  PushConstant pushConst;
};

const float DIRAC = -1.0;  // Used for direct light sampling

struct Ray
{
  vec3 origin;
  vec3 direction;
};

// Hit state information
struct HitState
{
  vec3 pos;
  vec3 nrm;
  vec3 geonrm;
  vec2 uv;
  vec3 tangent;
  vec3 bitangent;
};

// Payload for the path tracer
struct HitPayload
{
  float    hitT;
  int      meshID;
  HitState hitState;
};

//-----------------------------------------------------------------------
// Return hit information: position, normal, geonormal, uv, tangent, bitangent
HitState getHitState(PrimMeshInfo pinfo, vec2 barycentricCoords, mat4x3 worldToObject, mat4x3 objectToWorld, int primitiveID, vec3 worldRayDirection)
{
  HitState hit;

  // Vextex and indices of the primitive
  Vertices vertices = Vertices(pinfo.vertexAddress);
  Indices  indices  = Indices(pinfo.indexAddress);

  // Barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices.i[primitiveID];

  // All vertex attributes of the triangle.
  Vertex v0 = vertices.v[triangleIndex.x];
  Vertex v1 = vertices.v[triangleIndex.y];
  Vertex v2 = vertices.v[triangleIndex.z];

  // Position
  const vec3 pos0     = v0.position.xyz;
  const vec3 pos1     = v1.position.xyz;
  const vec3 pos2     = v2.position.xyz;
  const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos             = vec3(objectToWorld * vec4(position, 1.0));

  // Normal
  const vec3 nrm0           = v0.normal.xyz;
  const vec3 nrm1           = v1.normal.xyz;
  const vec3 nrm2           = v2.normal.xyz;
  const vec3 normal         = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  vec3       worldNormal    = normalize(vec3(normal * worldToObject));
  const vec3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  vec3       worldGeoNormal = normalize(vec3(geoNormal * worldToObject));
  hit.geonrm                = worldGeoNormal;
  hit.nrm                   = worldNormal;

  // TexCoord
  const vec2 uv0 = vec2(v0.position.w, v0.normal.w);
  const vec2 uv1 = vec2(v1.position.w, v1.normal.w);
  const vec2 uv2 = vec2(v2.position.w, v2.normal.w);
  hit.uv         = mixBary(uv0, uv1, uv2, barycentrics);

  // Tangent - Bitangent
  const vec4 tng0 = vec4(v0.tangent);
  const vec4 tng1 = vec4(v1.tangent);
  const vec4 tng2 = vec4(v2.tangent);
  hit.tangent     = normalize(mixBary(tng0.xyz, tng1.xyz, tng2.xyz, barycentrics));
  hit.tangent     = vec3(objectToWorld * vec4(hit.tangent, 0.0));
  hit.tangent     = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent));
  hit.bitangent   = cross(hit.nrm, hit.tangent) * tng0.w;

  // Adjusting normal
  const vec3 V = (-worldRayDirection);
  if(dot(hit.geonrm, V) < 0)  // Flip if back facing
    hit.geonrm = -hit.geonrm;

  // If backface
  if(dot(hit.geonrm, hit.nrm) < 0)  // Make Normal and GeoNormal on the same side
  {
    hit.nrm       = -hit.nrm;
    hit.tangent   = -hit.tangent;
    hit.bitangent = -hit.bitangent;
  }

  // handle low tessellated meshes with smooth normals
  vec3 k2 = reflect(-V, hit.nrm);
  if(dot(hit.geonrm, k2) < 0.0F)
    hit.nrm = hit.geonrm;

  // For low tessalated, avoid internal reflection
  vec3 r = reflect(normalize(worldRayDirection), hit.nrm);
  if(dot(r, hit.geonrm) < 0)
    hit.nrm = hit.geonrm;

  return hit;
}

struct DirectLight
{
  vec3  direction;        // Direction to the light
  vec3  radianceOverPdf;  // Radiance over pdf
  float distance;         // Distance to the light
  float pdf;              // Probability of sampling this light
};


//-----------------------------------------------------------------------
// This should sample any lights in the scene, but we only have the sun
void sampleLights(in vec3 pos, vec3 normal, in vec3 worldRayDirection, inout uint seed, out DirectLight directLight)
{
  vec3  radiance = vec3(0);
  float lightPdf = 1.0;

  // Light contribution
  Light sun;
  sun.type                  = eLightTypeDirectional;
  sun.angularSizeOrInvRange = skyInfo.angularSizeOfLight;
  sun.direction             = -skyInfo.directionToLight;
  sun.color                 = skyInfo.lightColor;
  sun.intensity             = 1;
  vec2         rand_val     = vec2(rand(seed), rand(seed));
  LightContrib lightContrib = singleLightContribution(sun, pos, normal, -worldRayDirection, rand_val);

  // Return the light contribution
  directLight.direction       = normalize(-lightContrib.incidentVector);
  directLight.radianceOverPdf = lightContrib.intensity / lightPdf;
  directLight.distance        = INFINITE;
  directLight.pdf             = DIRAC;
}

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return true if it is opaque
bool hitTest(in rayQueryEXT rayQuery, inout uint seed)
{
  int meshID      = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false);
  int primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);

  // Retrieve the Primitive mesh buffer information
  PrimMeshInfo pInfo = PrimMeshInfoBuf(sceneDesc.primInfoAddress).i[meshID];

  // Find the material of the primitive
  const uint        matIndex  = max(0, pInfo.materialIndex);                 // material of primitive mesh
  GltfMaterialBuf   materials = GltfMaterialBuf(sceneDesc.materialAddress);  // Buffer of materials
  GltfShadeMaterial material  = materials.m[matIndex];
  if(material.alphaMode == ALPHA_OPAQUE)
    return true;

  float baseColorAlpha = material.pbrBaseColorFactor.a;
  if(material.pbrBaseColorTexture > -1)
  {
    // Primitive buffer addresses
    Vertices vertices = Vertices(pInfo.vertexAddress);
    Indices  indices  = Indices(pInfo.indexAddress);

    // Getting the 3 indices of the triangle (local)
    uvec3 triangleIndex = indices.i[primitiveID];

    // All vertex attributes of the triangle.
    Vertex v0 = vertices.v[triangleIndex.x];
    Vertex v1 = vertices.v[triangleIndex.y];
    Vertex v2 = vertices.v[triangleIndex.z];

    // Get the texture coordinate
    vec2       bary         = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);
    const vec3 barycentrics = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    const vec2 uv0          = vec2(v0.position.w, v0.normal.w);
    const vec2 uv1          = vec2(v1.position.w, v1.normal.w);
    const vec2 uv2          = vec2(v2.position.w, v2.normal.w);
    vec2       texcoord0    = mixBary(uv0, uv1, uv2, barycentrics);

    // Uv Transform
    //texcoord0 = (vec4(texcoord0.xy, 1, 1) * material.uvTransform).xy;

    baseColorAlpha *= texture(texturesMap[nonuniformEXT(material.pbrBaseColorTexture)], texcoord0).a;
  }

  float opacity;
  if(material.alphaMode == ALPHA_MASK)
  {
    opacity = baseColorAlpha > material.alphaCutoff ? 1.0 : 0.0;
  }
  else
  {
    opacity = baseColorAlpha;
  }

  // do alpha blending the stochastically way
  if(rand(seed) > opacity)
    return false;

  return true;
}

//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void traceRay(in rayQueryEXT rayQuery, Ray ray, inout HitPayload payload, inout uint seed)
{
  payload.hitT = INFINITE;  // Default when not hitting anything

  uint rayFlags = gl_RayFlagsNoneEXT;
  rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, ray.origin, 0.0, ray.direction, INFINITE);
  while(rayQueryProceedEXT(rayQuery))
  {
    if(hitTest(rayQuery, seed))
    {
      rayQueryConfirmIntersectionEXT(rayQuery);
    }
  }

  if(rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT)
  {
    vec2   barycentricCoords = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    int    meshID            = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    int    primitiveID       = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    mat4x3 worldToObject     = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
    mat4x3 objectToWorld     = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    float  hitT              = rayQueryGetIntersectionTEXT(rayQuery, true);
    vec3   worldRayDirection = rayQueryGetWorldRayDirectionEXT(rayQuery);

    // Retrieve the Primitive mesh buffer information
    PrimMeshInfoBuf meshes = PrimMeshInfoBuf(sceneDesc.primInfoAddress);
    PrimMeshInfo    pinfo  = meshes.i[meshID];

    HitState hit = getHitState(pinfo, barycentricCoords, worldToObject, objectToWorld, primitiveID, worldRayDirection);

    payload.hitT     = hitT;
    payload.meshID   = meshID;
    payload.hitState = hit;
  }
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool traceShadow(in rayQueryEXT rayQuery, Ray ray, float maxDist, inout uint seed)
{
  uint rayFlags = gl_RayFlagsNoneEXT;  //  gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT;
  rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, ray.origin, 0.0, ray.direction, maxDist);

  while(rayQueryProceedEXT(rayQuery))
  {
    if(hitTest(rayQuery, seed))
    {
      rayQueryConfirmIntersectionEXT(rayQuery);
    }
  }

  return (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);  // Is Hit ?
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 pathTrace(Ray ray, inout uint seed)
{
  vec3 radiance   = vec3(0.0F);
  vec3 throughput = vec3(1.0F);
  bool isInside   = false;

  // Setting up the material
  GltfMaterialBuf materials = GltfMaterialBuf(sceneDesc.materialAddress);  // Buffer of materials
  PrimMeshInfoBuf meshes    = PrimMeshInfoBuf(sceneDesc.primInfoAddress);  // Buffer of meshes
  rayQueryEXT     rayQuery;
  HitPayload      payload;

  for(int depth = 0; depth < pushConst.maxDepth; depth++)
  {
    traceRay(rayQuery, ray, payload, seed);

    // Hitting the environment, then exit
    if(payload.hitT == INFINITE)
    {
      vec3 sky_color = proceduralSky(skyInfo, ray.direction, 0);
      return radiance + (sky_color * throughput);
    }

    // Getting the hit information (primitive/mesh that was hit)
    HitState hit = payload.hitState;

    // Setting up the material
    PrimMeshInfo      pInfo         = meshes.i[payload.meshID];     // Primitive information
    int               materialIndex = max(0, pInfo.materialIndex);  // Material ID of hit mesh
    GltfShadeMaterial material      = materials.m[materialIndex];   // Material of the hit object
    PbrMaterial       pbrMat        = evaluateMaterial(material, hit.nrm, hit.tangent, hit.bitangent, hit.uv, isInside);

    // Adding emissive
    radiance += pbrMat.emissive * throughput;

    // Apply volume attenuation
    bool thin_walled = pbrMat.thicknessFactor == 0;
    if(isInside && !thin_walled)
    {
      const vec3 abs_coeff = absorptionCoefficient(pbrMat);
      throughput.x *= abs_coeff.x > 0.0 ? exp(-abs_coeff.x * payload.hitT) : 1.0;
      throughput.y *= abs_coeff.y > 0.0 ? exp(-abs_coeff.y * payload.hitT) : 1.0;
      throughput.z *= abs_coeff.z > 0.0 ? exp(-abs_coeff.z * payload.hitT) : 1.0;
    }

    vec3 contribution = vec3(0);  // Direct lighting contribution

    // Light contribution; can be environment or punctual lights
    DirectLight directLight;
    sampleLights(hit.pos, pbrMat.normal, ray.direction, seed, directLight);

    // Evaluation of direct light (sun)
    const bool nextEventValid = ((dot(directLight.direction, hit.geonrm) > 0.0F) != isInside) && directLight.pdf != 0.0F;
    if(nextEventValid)
    {
      BsdfEvaluateData evalData;
      evalData.k1 = -ray.direction;
      evalData.k2 = directLight.direction;
      bsdfEvaluate(evalData, pbrMat);

      if(evalData.pdf > 0.0)
      {
        const float mis_weight = (directLight.pdf == DIRAC) ? 1.0F : directLight.pdf / (directLight.pdf + evalData.pdf);

        // sample weight
        const vec3 w = throughput * directLight.radianceOverPdf * mis_weight;
        contribution += w * evalData.bsdf_diffuse;
        contribution += w * evalData.bsdf_glossy;
      }
    }

    // Sample BSDF
    {
      BsdfSampleData sampleData;
      sampleData.k1 = -ray.direction;  // outgoing direction
      sampleData.xi = vec4(rand(seed), rand(seed), rand(seed), rand(seed));
      bsdfSample(sampleData, pbrMat);

      throughput *= sampleData.bsdf_over_pdf;
      ray.direction = sampleData.k2;

      if(sampleData.event_type == BSDF_EVENT_ABSORB)
      {
        break;  // Need to add the contribution ?
      }
      else
      {
        // Continue path
        bool isSpecular     = (sampleData.event_type & BSDF_EVENT_SPECULAR) != 0;
        bool isTransmission = (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0;

        vec3 offsetDir = dot(ray.direction, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
        ray.origin     = offsetRay(hit.pos, offsetDir);

        if(isTransmission)
        {
          isInside = !isInside;
        }
      }
    }

    // We are adding the contribution to the radiance only if the ray is not occluded by an object.
    if(nextEventValid)
    {
      Ray  shadowRay = Ray(ray.origin, directLight.direction);
      bool inShadow  = traceShadow(rayQuery, shadowRay, directLight.distance, seed);
      if(!inShadow)
      {
        radiance += contribution;
      }
    }

    // Russian-Roulette (minimizing live state)
    float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
    if(rand(seed) >= rrPcont)
      break;                // paths with low throughput that won't contribute
    throughput /= rrPcont;  // boost the energy of the non-terminated paths
  }

  return radiance;
}


//-----------------------------------------------------------------------
// Sampling the pixel
//-----------------------------------------------------------------------
vec3 samplePixel(inout uint seed, uvec2 launchID, ivec2 launchSize)
{
  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  vec2 subPixelJitter = vec2(rand(seed), rand(seed));
  vec2 clipCoords     = (vec2(launchID) + subPixelJitter) / vec2(launchSize) * 2.0 - 1.0;
  vec4 viewCoords     = cameraInfo.projInv * vec4(clipCoords, -1.0, 1.0);
  viewCoords /= viewCoords.w;

  const vec3 origin    = vec3(cameraInfo.viewInv[3].xyz);
  const vec3 direction = normalize((cameraInfo.viewInv * viewCoords).xyz - origin);

  Ray ray = Ray(origin.xyz, direction.xyz);

  vec3 radiance = pathTrace(ray, seed);

  // Removing fireflies
  float lum = dot(radiance, vec3(0.212671F, 0.715160F, 0.072169F));
  if(lum > pushConst.fireflyClampThreshold)
  {
    radiance *= pushConst.fireflyClampThreshold / lum;
  }

  return radiance;
}

//-----------------------------------------------------------------------
// Main function
//-----------------------------------------------------------------------
void main()
{
  ivec2 launchSize = imageSize(image);
  uvec2 launchID   = gl_GlobalInvocationID.xy;

  // Check if not outside boundaries
  if(launchID.x >= launchSize.x || launchID.y >= launchSize.y)
    return;

  // Initialize the random number
  uint seed = xxhash32(uvec3(launchID.xy, pushConst.frame));

  // Sampling n times the pixel
  vec3 pixel_color = vec3(0.0F, 0.0F, 0.0F);
  for(int s = 0; s < pushConst.maxSamples; s++)
  {
    pixel_color += samplePixel(seed, launchID, launchSize);
  }
  pixel_color /= pushConst.maxSamples;

  bool firstFrame = (pushConst.frame <= 1);
  // Saving result
  if(firstFrame)
  {  // First frame, replace the value in the buffer
    imageStore(image, ivec2(launchID.xy), vec4(pixel_color, 1.0F));
  }
  else
  {  // Do accumulation over time
    float a         = 1.0F / float(pushConst.frame + 1);
    vec3  old_color = imageLoad(image, ivec2(launchID.xy)).xyz;
    imageStore(image, ivec2(launchID.xy), vec4(mix(old_color, pixel_color, a), 1.0F));
  }
}
