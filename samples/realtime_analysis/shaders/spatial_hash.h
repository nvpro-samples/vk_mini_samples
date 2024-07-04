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

// Code converted from: https://github.com/SebLague/Fluid-Sim

const ivec2 offsets2D[9] = {
    ivec2(-1, 1), ivec2(0, 1),   ivec2(1, 1),  ivec2(-1, 0), ivec2(0, 0),
    ivec2(1, 0),  ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1),
};

// Constants used for hashing
const uint hashK1 = 15823;
const uint hashK2 = 9737333;

// Convert floating point position into an integer cell coordinate
ivec2 getCell2D(vec2 position, float radius)
{
  return ivec2(floor(position / radius));
}

// Hash cell coordinate to a single unsigned integer
uint hashCell2D(ivec2 cell)
{
  cell   = ivec2(uvec2(cell.x, cell.y));
  uint a = cell.x * hashK1;
  uint b = cell.y * hashK2;
  return (a + b);
}

uint keyFromHash(uint hash, uint tableSize)
{
  return hash % tableSize;
}
