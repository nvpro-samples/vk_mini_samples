
#ifndef SHADERIO_RT_GLTF_H
#define SHADERIO_RT_GLTF_H

#include <nvshaders/slang_types.h>

#include "nvshaders/sky_io.h.slang"

enum BindingIndex
{
  eTlas      = 0,
  eOutImage  = 1,
  eSceneDesc = 2,
};

// GLTF
struct BufferView
{
  uint32_t offset     = 0;
  uint32_t count      = 0;
  uint32_t byteStride = 0;
};

struct TriangleMesh
{
  BufferView indices   = {};
  BufferView positions = {};
  BufferView normals   = {};
};

struct RtMeshInfo
{
  TriangleMesh* mesh              = nullptr;
  uint64_t      baseBufferAddress = 0;
};

struct RtInstanceInfo
{
  uint32_t meshIndex = 0;
  float4x4 transform;
  float3   color;
};

struct RtGltfSceneInfo
{
  float4x4              projInvMatrix;
  float4x4              viewInvMatrix;
  RtInstanceInfo*       instances = nullptr;
  RtMeshInfo*           meshes    = nullptr;
  SkyPhysicalParameters skyParams = {};
};


struct RtGltfMetallicRoughness
{
  float4 baseColorFactor;
  float  metallicFactor;
  float  roughnessFactor;
};

struct RtGltfPushConstant
{
  float    metallic  = 0.2f;
  float    roughness = 0.5f;
  uint32_t maxDepth  = 3;
};


#endif  // SHADERIO_RT_GLTF_H
