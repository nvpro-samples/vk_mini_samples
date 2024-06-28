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


// This file is included in the shaders to provide access to the vertex data


#ifndef VERTEX_ACCESSOR_SLANG
#define VERTEX_ACCESSOR_SLANG

#include "common/shaders/dh_scn_desc.hlsli"
#include "common/shaders/functions.hlsli"


uint3 getTriangleIndices(RenderPrimitive renderPrim, uint idx)
{
  return renderPrim.indexAddress[idx];
}

float3 getVertexPosition(RenderPrimitive renderPrim, uint idx)
{
  return renderPrim.vertexBuffer.positionAddress[idx];
}

float3 getInterpolatedVertexPosition(RenderPrimitive renderPrim, uint3 idx, float3 barycentrics)
{
  float3 *positions = renderPrim.vertexBuffer.positionAddress;
  float3 pos[3];
  pos[0] = positions[idx.x];
  pos[1] = positions[idx.y];
  pos[2] = positions[idx.z];
  return pos[0] * barycentrics.x + pos[1] * barycentrics.y + pos[2] * barycentrics.z;
}

bool hasVertexNormal(RenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.normalAddress != nullptr;
}

float3 getVertexNormal(RenderPrimitive renderPrim, uint idx)
{
  if(!hasVertexNormal(renderPrim))
    return float3(0.0f, 0.0f, 1.0f);

  return renderPrim.vertexBuffer.normalAddress[idx];
}

float3 getInterpolatedVertexNormal(RenderPrimitive renderPrim, uint3 idx, float3 barycentrics)
{
  if(!hasVertexNormal(renderPrim))
    return float3(0.0f, 0.0f, 1.0f);

  float3 *normals = renderPrim.vertexBuffer.normalAddress;
  float3 nrm[3];
  nrm[0] = normals[idx.x];
  nrm[1] = normals[idx.y];
  nrm[2] = normals[idx.z];

  return nrm[0] * barycentrics.x + nrm[1] * barycentrics.y + nrm[2] * barycentrics.z;
}

bool hasVertexTexCoord0(RenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.texCoord0Address != nullptr;
}

float2 getVertexTexCoord0(RenderPrimitive renderPrim, uint idx)
{
  if(!hasVertexTexCoord0(renderPrim))
    return float2(0.0f, 0.0f);

  return renderPrim.vertexBuffer.texCoord0Address[idx];
}

float2 getInterpolatedVertexTexCoord0(RenderPrimitive renderPrim, uint3 idx, float3 barycentrics)
{
  if (!hasVertexTexCoord0(renderPrim))
    return float2(0.0f, 0.0f);

  float2 *texcoords = renderPrim.vertexBuffer.texCoord0Address;
  float2 uv[3];
  uv[0] = texcoords[idx.x];
  uv[1] = texcoords[idx.y];
  uv[2] = texcoords[idx.z];  return uv[0] * barycentrics.x + uv[1] * barycentrics.y + uv[2] * barycentrics.z;
}

bool hasVertexTangent(RenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.tangentAddress != nullptr;
}

float4 getVertexTangent(RenderPrimitive renderPrim, uint idx)
{
  if (!hasVertexTangent(renderPrim))
    return float4(1, 0, 0, 1);

  return renderPrim.vertexBuffer.tangentAddress[idx];
}

float4 getInterpolatedVertexTangent(RenderPrimitive renderPrim, uint3 idx, float3 barycentrics)
{
  if (!hasVertexTangent(renderPrim))
    return float4(1, 0, 0, 1);

  float4 *tangents = renderPrim.vertexBuffer.tangentAddress;
  float4 tng[3];
  tng[0] = tangents[idx.x];
  tng[1] = tangents[idx.y];
  tng[2] = tangents[idx.z];
  return tng[0] * barycentrics.x + tng[1] * barycentrics.y + tng[2] * barycentrics.z;
}

bool hasVertexColor(RenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.colorAddress != nullptr;
}

float4 getVertexColor(RenderPrimitive renderPrim, uint idx)
{
  if (!hasVertexColor(renderPrim))
    return float4(1.0f, 1.0f, 1.0f, 1.0f);

  return unpackUnorm4x8(renderPrim.vertexBuffer.colorAddress[idx]);
}

float4 getInterpolatedVertexColor(RenderPrimitive renderPrim, uint3 idx, float3 barycentrics)
{
  if(!hasVertexColor(renderPrim))
    return float4(1.0f, 1.0f, 1.0f, 1.0f);

  uint *colors = renderPrim.vertexBuffer.colorAddress;
  float4 col[3];
  col[0] = unpackUnorm4x8(colors[idx.x]);
  col[1] = unpackUnorm4x8(colors[idx.y]);
  col[2] = unpackUnorm4x8(colors[idx.z]);
  return col[0] * barycentrics.x + col[1] * barycentrics.y + col[2] * barycentrics.z;
}

#endif  // VERTEX_ACCESSOR_SLANG
