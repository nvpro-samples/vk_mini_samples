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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#ifdef __glsl
#define inline
#endif

struct PushConstant
{
  mat4  transfo;
  vec4  color;
  float threshold;
  int   steps;
  int   size;
};

struct FrameInfo
{
  mat4 proj;
  mat4 view;
  vec3 camPos;
  vec3 toLight;
  int  headlight;
};

struct PerlinSettings
{
  int   octave;
  float power;
  float frequency;
};

inline PerlinSettings PerlinDefaultValues()
{
  PerlinSettings perlin;
  perlin.power     = 1.0F;
  perlin.octave    = 3;
  perlin.frequency = 1.0F;
  return perlin;
}
