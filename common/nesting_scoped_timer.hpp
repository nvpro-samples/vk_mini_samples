
/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <string>
#include "nvh/nvprint.hpp"
#include "nvh/timesampler.hpp"


//--------------------------------------------------------------------------------------------------
// Print the time a function takes and indent nested functions
//
struct NestingScopedTimer
{
  explicit NestingScopedTimer(std::string str)
      : m_name(std::move(str))
  {
    LOGI("%s%s:\n", indent().c_str(), m_name.c_str());
    ++s_depth;
  }

  ~NestingScopedTimer()
  {
    --s_depth;
    LOGI("%s|-> (%.3f ms)\n", indent().c_str(), m_sw.elapsed());
  }

  static std::string indent();

  std::string                m_name;
  nvh::Stopwatch             m_sw;
  static thread_local size_t s_depth;
};
