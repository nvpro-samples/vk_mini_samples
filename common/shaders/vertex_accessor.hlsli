/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


// This file is included in the shaders to provide access to the vertex data


#ifndef VERTEX_ACCESSOR_HLSLI
#define VERTEX_ACCESSOR_HLSLI

#include "common/shaders/dh_scn_desc.hlsli"
#include "common/shaders/functions.hlsli"


uint3 getTriangleIndices(RenderPrimitive renderPrim, uint idx)
{
  uint64_t indexOffset = sizeof(uint3) * idx;
  return vk::RawBufferLoad<uint3> (renderPrim.indexAddress + indexOffset);
}

float3 getVertexPosition(RenderPrimitive renderPrim, uint idx)
{
  uint64_t offset = sizeof(float3) * idx;
  return vk::RawBufferLoad<float3> (renderPrim.vertexBuffer.positionAddress + offset);
}

float3 getInterpolatedVertexPosition(RenderPrimitive renderPrim,uint3 idx,float3 barycentrics)
{
  float3 pos[3];
  pos[0] = vk::RawBufferLoad<float3> (renderPrim.vertexBuffer.positionAddress + sizeof(float3) * idx.x);
  pos[1] = vk::RawBufferLoad<float3> (renderPrim.vertexBuffer.positionAddress + sizeof(float3) * idx.y);
  pos[2] = vk::RawBufferLoad<float3> (renderPrim.vertexBuffer.positionAddress + sizeof(float3) * idx.z);
  return pos[0] * barycentrics.x + pos[1] * barycentrics.y + pos[2] * barycentrics.z;
}

bool hasVertexNormal(RenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.normalAddress != 0;
}

float3 getVertexNormal(RenderPrimitive renderPrim, uint idx)
{
  if(!hasVertexNormal(renderPrim))
    return float3(0, 0, 1);
  
  uint64_t offset = sizeof(float3) * idx;
  return vk::RawBufferLoad<float3> (renderPrim.vertexBuffer.normalAddress + offset);
  
}

float3 getInterpolatedVertexNormal(RenderPrimitive renderPrim,uint3 idx,float3 barycentrics)
{
  if(!hasVertexNormal(renderPrim))
    return float3(0, 0, 1);
  
  float3 val[3];
  val[0] = vk::RawBufferLoad<float3> (renderPrim.vertexBuffer.normalAddress + sizeof(float3) * idx.x);
  val[1] = vk::RawBufferLoad<float3> (renderPrim.vertexBuffer.normalAddress + sizeof(float3) * idx.y);
  val[2] = vk::RawBufferLoad<float3> (renderPrim.vertexBuffer.normalAddress + sizeof(float3) * idx.z);
  return val[0] * barycentrics.x + val[1] * barycentrics.y + val[2] * barycentrics.z;
  
}

bool hasVertexTexCoord0(RenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.texCoord0Address != 0;
}

float2 getVertexTexCoord0(RenderPrimitive renderPrim, uint idx)
{
  if(!hasVertexTexCoord0(renderPrim))
    return float2(0, 0);

  return vk::RawBufferLoad<float2> (renderPrim.vertexBuffer.texCoord0Address + sizeof(float2) * idx);
}


float2 getInterpolatedVertexTexCoord0(RenderPrimitive renderPrim,uint3 idx,float3 barycentrics)
{
  if(!hasVertexTexCoord0(renderPrim))
    return float2(0, 0);
  
  float2 val[3];
  val[0] = vk::RawBufferLoad<float2> (renderPrim.vertexBuffer.texCoord0Address + sizeof(float2) * idx.x);
  val[1] = vk::RawBufferLoad<float2> (renderPrim.vertexBuffer.texCoord0Address + sizeof(float2) * idx.y);
  val[2] = vk::RawBufferLoad<float2> (renderPrim.vertexBuffer.texCoord0Address + sizeof(float2) * idx.z);
  return val[0] * barycentrics.x + val[1] * barycentrics.y + val[2] * barycentrics.z;
}


bool hasVertexTexCoord1(RenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.texCoord1Address != 0;
}

float2 getVertexTexCoord1(RenderPrimitive renderPrim, uint idx)
{
  if (!hasVertexTexCoord1(renderPrim))
    return float2(0, 0);

  return vk::RawBufferLoad <
  float2 > (renderPrim.vertexBuffer.texCoord1Address + sizeof(float2) * idx);
}


float2 getInterpolatedVertexTexCoord1(RenderPrimitive renderPrim, uint3 idx, float3 barycentrics)
{
  if (!hasVertexTexCoord1(renderPrim))
    return float2(0, 0);
  
  float2 val[3];
  val[0] = vk::RawBufferLoad <
  float2 > (renderPrim.vertexBuffer.texCoord1Address + sizeof(float2) * idx.x);
  val[1] = vk::RawBufferLoad <
  float2 > (renderPrim.vertexBuffer.texCoord1Address + sizeof(float2) * idx.y);
  val[2] = vk::RawBufferLoad <
  float2 > (renderPrim.vertexBuffer.texCoord1Address + sizeof(float2) * idx.z);
  return val[0] * barycentrics.x + val[1] * barycentrics.y + val[2] * barycentrics.z;
}



bool hasVertexTangent(RenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.tangentAddress != 0;
}

float4 getVertexTangent(RenderPrimitive renderPrim, uint idx)
{
  if(!hasVertexTangent(renderPrim))
    return float4(1, 0, 0, 1);

  return vk::RawBufferLoad<float4> (renderPrim.vertexBuffer.tangentAddress + sizeof(float4) * idx);
}

float4 getInterpolatedVertexTangent(RenderPrimitive renderPrim,uint3 idx,float3 barycentrics)
{
  if(!hasVertexTangent(renderPrim))
    return float4(1, 0, 0, 1);

  float4 val[3];
  val[0] = vk::RawBufferLoad<float4> (renderPrim.vertexBuffer.tangentAddress + sizeof(float4) * idx.x);
  val[1] = vk::RawBufferLoad<float4> (renderPrim.vertexBuffer.tangentAddress + sizeof(float4) * idx.y);
  val[2] = vk::RawBufferLoad<float4> (renderPrim.vertexBuffer.tangentAddress + sizeof(float4) * idx.z);
  return val[0] * barycentrics.x + val[1] * barycentrics.y + val[2] * barycentrics.z;
  
}

bool hasVertexColor(RenderPrimitive renderPrim)
{
  return renderPrim.vertexBuffer.colorAddress != 0;
}

float4 getVertexColor(RenderPrimitive renderPrim, uint idx)
{
  if(!hasVertexColor(renderPrim))
    return float4(1, 1, 1, 1);

  uint col =vk::RawBufferLoad<uint> (renderPrim.vertexBuffer.colorAddress + sizeof(uint) * idx);
  return unpackUnorm4x8(col);
}

float4 getInterpolatedVertexColor(RenderPrimitive renderPrim,uint3 idx,float3 barycentrics)
{
  if(!hasVertexColor(renderPrim))
    return float4(1, 1, 1, 1);

  uint icol0 = vk::RawBufferLoad<uint> (renderPrim.vertexBuffer.colorAddress + sizeof(uint) * idx.x);
  uint icol1 = vk::RawBufferLoad<uint> (renderPrim.vertexBuffer.colorAddress + sizeof(uint) * idx.y);
  uint icol2 = vk::RawBufferLoad<uint> (renderPrim.vertexBuffer.colorAddress + sizeof(uint) * idx.z);
  float4 col[3];
  col[0] = unpackUnorm4x8(icol0);
  col[1] = unpackUnorm4x8(icol1);
  col[2] = unpackUnorm4x8(icol2);
  return col[0] * barycentrics.x + col[1] * barycentrics.y + col[2] * barycentrics.z;
}

#endif  // VERTEX_ACCESSOR_HLSLI
