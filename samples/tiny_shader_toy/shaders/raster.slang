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

public typealias vec2 = vector<float, 2>;
public typealias vec3 = vector<float, 3>;
public typealias vec4 = vector<float, 4>;

public typealias bvec2 = vector<bool, 2>;
public typealias bvec3 = vector<bool, 3>;
public typealias bvec4 = vector<bool, 4>;

public typealias ivec2 = vector<int, 2>;
public typealias ivec3 = vector<int, 3>;
public typealias ivec4 = vector<int, 4>;

public typealias uvec2 = vector<uint, 2>;
public typealias uvec3 = vector<uint, 3>;
public typealias uvec4 = vector<uint, 4>;

public typealias mat2 = matrix<float, 2, 2>;
public typealias mat3 = matrix<float, 3, 3>;
public typealias mat4 = matrix<float, 4, 4>;

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
public T mix(T x, T y, T a)
{
    return lerp(x, y, a);
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
public vector<T, N> mix(vector<T, N> x, vector<T, N> y, T a)
{
    return lerp(x, y, vector<T, N>(a));
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
public vector<T, N> mix(vector<T, N> x, vector<T, N> y, vector<T, N> a)
{
    return lerp(x, y, a);
}

__generic<T : __BuiltinFloatingPointType>
[__readNone]
[ForceInline]
public T mod(T x, T y)
{
    return fmod(x, y);
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
public vector<T, N> mod(vector<T, N> x, T y)
{
    return fmod(x, vector<T, N>(y));
}

__generic<T : __BuiltinFloatingPointType, let N : int>
[__readNone]
[ForceInline]
public vector<T, N> mod(vector<T, N> x, vector<T, N> y)
{
    return fmod(x, y);
}

[[vk::push_constant]] cbuffer MyPush // Layout std140 :-(
{
    float3 iResolution;
    float iTime;
    float4 iMouse; // 16
    float iTimeDelta; //32
    int iFrame; // 36
    int iFrameRate; // 40
    int _pad0;
    float iChannelTime[1];
    //int _pad1;
    //int _pad2;
    //int _pad3;
    float3 iChannelResolution[1]; // 48
};

[[vk::binding(0)]] Sampler2D iChannel0;

// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin
{
  [[vk::location(0)]] float2 position : POSITION;
};

// Output of the vertex shader
struct VSout
{
  float4 sv_position : SV_Position;
};

// Vertex  Shader
[shader("vertex")]
VSout vertexMain(VSin input)
{
  
  VSout output;
  output.sv_position = float4(input.position, 0.0, 1.0);

  return output;
}

#ifdef INCLUDE_IMAGE
#include "image.glsl"
#else
#include "buffer_a.glsl"
#endif

// Fragment Shader
[shader("pixel")]
float4 fragmentMain(in float4 fragCoord: SV_Position) : SV_Target
{
    // Inverting fragCoord for OpenGL
    vec2 invFragCoord = fragCoord.xy;
    invFragCoord.y = iResolution.y - fragCoord.y;

    float4 fragColor;
    mainImage(fragColor, invFragCoord.xy);

    return fragColor;
}
