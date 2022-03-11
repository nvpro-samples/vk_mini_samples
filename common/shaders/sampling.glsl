/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

const float M_PI        = 3.14159265358979323846;   // PI
const float M_TWO_PI    = 6.28318530717958648;      // 2*PI
const float M_PI_2      = 1.57079632679489661923;   // PI/2
const float M_PI_4      = 0.785398163397448309616;  // PI/4
const float M_1_OVER_PI = 0.318309886183790671538;  // 1/PI
const float M_2_OVER_PI = 0.636619772367581343076;  // 2/PI

// Generate a seed for the random generator.
// Input - pixel.x, pixel.y, frame_nb
// From https://github.com/Cyan4973/xxHash, https://www.shadertoy.com/view/XlGcRh
uint xxhash32(uvec3 p)
{
  const uvec4 primes = uvec4(2246822519U, 3266489917U, 668265263U, 374761393U);
  uint        h32;
  h32 = p.z + primes.w + p.x * primes.y;
  h32 = primes.z * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 += p.y * primes.y;
  h32 = primes.z * ((h32 << 17) | (h32 >> (32 - 17)));
  h32 = primes.x * (h32 ^ (h32 >> 15));
  h32 = primes.y * (h32 ^ (h32 >> 13));
  return h32 ^ (h32 >> 16);
}

//-----------------------------------------------------------------------
// https://www.pcg-random.org/
//-----------------------------------------------------------------------
uint pcg(inout uint state)
{
  uint prev = state * 747796405u + 2891336453u;
  uint word = ((prev >> ((prev >> 28u) + 4u)) ^ prev) * 277803737u;
  state     = prev;
  return (word >> 22u) ^ word;
}

//-----------------------------------------------------------------------
// Generate a random float in [0, 1) given the previous RNG state
//-----------------------------------------------------------------------
float rand(inout uint seed)
{
  uint r = pcg(seed);
  return r * (1.0 / float(0xffffffffu));
}


//-------------------------------------------------------------------------------------------------
// Sampling
//-------------------------------------------------------------------------------------------------

// Randomly sampling around +Z
vec3 cosineSamplingHemisphere(inout uint seed, in vec3 x, in vec3 y, in vec3 z, in vec3 randVal)
{
#define M_PI 3.141592

  float r1 = randVal.x;
  float r2 = randVal.y;
  float sq = sqrt(1.0 - r2);

  vec3 direction = vec3(cos(2 * M_PI * r1) * sq, sin(2 * M_PI * r1) * sq, sqrt(r2));
  direction      = direction.x * x + direction.y * y + direction.z * z;

  return direction;
}

// Return the tangent and binormal from the incoming normal
void createCoordinateSystem(in vec3 N, out vec3 Nt, out vec3 Nb)
{
  if(abs(N.x) > abs(N.y))
    Nt = vec3(N.z, 0, -N.x) / sqrt(N.x * N.x + N.z * N.z);
  else
    Nt = vec3(0, -N.z, N.y) / sqrt(N.y * N.y + N.z * N.z);
  Nb = cross(N, Nt);
}


//-----------------------------------------------------------------------
// Return the UV in a lat-long HDR map
//-----------------------------------------------------------------------
vec2 getSphericalUv(vec3 v)
{
  float gamma = asin(-v.y);
  float theta = atan(v.z, v.x);

  vec2 uv = vec2(theta * M_1_OVER_PI * 0.5, gamma * M_1_OVER_PI) + 0.5;
  return uv;
}

vec3 rotate(vec3 v, vec3 k, float theta)
{
  float cos_theta = cos(theta);
  float sin_theta = sin(theta);

  return (v * cos_theta) + (cross(k, v) * sin_theta) + (k * dot(k, v)) * (1 - cos_theta);
}
