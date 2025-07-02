/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#version 450

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_fragment_shader_barycentric : enable  // #BARY_WIRE
#extension GL_EXT_scalar_block_layout : enable

#include "shaderio.h"
#include "nvshaders/functions.h.slang"
#include "nvshaders/simple_shading.h.slang"
#include "nvshaders/constants.h.slang"


layout(location = 0) in vec3 inFragPos;
layout(location = 1) in vec3 inFragNrm;
layout(location = 0) out vec4 outColor;

// clang-format off
layout(set = 0, binding = BIND_FRAME_INFO) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = BIND_SETTINGS, scalar) uniform WireframeSettings_ { WireframeSettings settings;};
layout(push_constant) uniform PushConstant_ { PushConstant pushC; };
// clang-format on


// Return the width [0..1] for which the line should be displayed or not
float getLineWidth(in vec3 deltas, in float thickness, in float smoothing, in vec3 barys)
{
  barys         = smoothstep(deltas * (thickness), deltas * (thickness + smoothing), barys);
  float minBary = min(barys.x, min(barys.y, barys.z));
  return 1.0 - minBary;
}

// Position along the edge [0..1]
float edgePosition()
{
  return max(gl_BaryCoordEXT.z, max(gl_BaryCoordEXT.y, gl_BaryCoordEXT.x));
}

// Return 0 or 1 if edgePos should be diplayed or not
float stipple(in float stippleRepeats, in float stippleLength, in float edgePos)
{
  float offset = 1.0 / stippleRepeats;
  offset *= 0.5 * stippleLength;
  float pattern = fract((edgePos + offset) * stippleRepeats);
  return 1.0 - step(stippleLength, pattern);
}

// Vary the thickness along the edge
float edgeThickness(in vec2 thicknessVar, in float edgePos)
{
  return mix(thicknessVar.x, thicknessVar.y, (1.0 - sin(edgePos * M_PI)));
}

void main()
{
  vec3 toEye = frameInfo.camPos - inFragPos;
  vec3 color = simpleShading(toEye, toEye, inFragNrm, vec3(0.8), 16.0) * pushC.color.xyz;

  // For a one liner simple wireframe, this can be done for grey wireframe on top of the geometry
  // color = mix(color, vec3(0.8), getLineWidth(fwidthFine(gl_BaryCoordEXT), 0.5, 0.5, gl_BaryCoordEXT));

  // Wireframe Settings
  float thickness     = settings.thickness * 0.5;  // Thickness for both side of the edge, must be divided by 2
  vec3  wireColor     = settings.color;            // Color of the wireframe
  float smoothing     = settings.thickness * settings.smoothing;  // Could be thickness
  bool  enableStipple = (settings.enableStipple == 1);

  // Uniform position on the edge [0, 1]
  float edgePos = edgePosition();

  if(!gl_FrontFacing)
  {
    enableStipple = true;  // Forcing backface to always stipple the line
    wireColor     = settings.backFaceColor;
  }

  // [optional] Vary the thickness along the edge
  thickness *= edgeThickness(settings.thicknessVar, edgePos);

  // fwidth � return the sum of the absolute value of derivatives in x and y
  //          which makes the width in screen space
  vec3 deltas = (settings.screenSpace == 1) ? fwidthFine(gl_BaryCoordEXT) : vec3(1);

  // Get the wireframe line width
  float lineWidth = getLineWidth(deltas, thickness, smoothing, gl_BaryCoordEXT);

  // [optional]
  if(enableStipple)
  {
    float stippleFact = stipple(settings.stippleRepeats, settings.stippleLength, edgePos);
    lineWidth *= stippleFact;  // 0 or 1
  }

  // To see through, we discard faces and blend with the background
  if(settings.onlyWire == 1)
  {
    color = pushC.clearColor.xyz;
    if(lineWidth < 0.1)
      discard;
  }

  // Final color
  color = mix(color, wireColor, lineWidth);

  // Alpha is 1.0, because ImGui will blend it with its background
  outColor = vec4(color, 1.0);
}