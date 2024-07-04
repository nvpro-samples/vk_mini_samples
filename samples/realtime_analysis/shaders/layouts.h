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

layout(local_size_x = WORKGROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = eCompParticles, scalar) buffer _Particles
{
  Particle particles[];
};
layout(set = 0, binding = eCompSort, scalar) buffer _SpatialInfo
{
  SpatialInfo spatialInfo[];
};

layout(set = 0, binding = eCompSetting, scalar) uniform _ParticleSetting
{
  ParticleSetting setting;
};

layout(push_constant) uniform _PushConst
{
  PushConstant pushC;
};
