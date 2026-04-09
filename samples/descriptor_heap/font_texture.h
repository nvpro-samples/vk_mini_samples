/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Bitmap font texture generation for the descriptor_heap sample.
// Generates small RGBA textures with hex color codes and face labels.

#ifndef FONT_TEXTURE_H
#define FONT_TEXTURE_H

#include <cstdint>
#include <cstdio>
#include <vector>

static constexpr int s_TEX_SIZE = 48;

namespace font_texture {

// 5x7 pixel glyphs for hex digits and labels. Each row is 5 bits (MSB =
// leftmost).
// clang-format off
static const char     s_chars[] = "0123456789ABCDEF#+-XYZ :";
static const uint8_t  s_glyphs[][7] = {
  {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, // 0
  {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 1
  {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, // 2
  {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}, // 3
  {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 4
  {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, // 5
  {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, // 6
  {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
  {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 8
  {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, // 9
  {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, // A
  {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, // B
  {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, // C
  {0x1C,0x12,0x11,0x11,0x11,0x12,0x1C}, // D
  {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, // E
  {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, // F
  {0x04,0x0A,0x11,0x1F,0x11,0x0A,0x04}, // #
  {0x00,0x04,0x04,0x1F,0x04,0x04,0x00}, // +
  {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}, // -
  {0x11,0x0A,0x04,0x04,0x0A,0x11,0x00}, // X
  {0x11,0x0A,0x04,0x04,0x04,0x04,0x00}, // Y
  {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, // Z
  {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // space
  {0x00,0x00,0x0C,0x0C,0x00,0x0C,0x0C}, // :
};
// clang-format on

inline void setPixel(std::vector<uint8_t>& px, int x, int y, uint8_t r, uint8_t g, uint8_t b)
{
  if(x < 0 || x >= s_TEX_SIZE || y < 0 || y >= s_TEX_SIZE)
    return;
  int idx     = (y * s_TEX_SIZE + x) * 4;
  px[idx]     = r;
  px[idx + 1] = g;
  px[idx + 2] = b;
  px[idx + 3] = 255;
}

inline void drawString(std::vector<uint8_t>& px, int ox, int oy, const char* str, uint8_t r, uint8_t g, uint8_t b)
{
  for(int i = 0; str[i]; i++)
  {
    int idx = 22;  // space
    for(int j = 0; s_chars[j]; j++)
      if(s_chars[j] == str[i])
      {
        idx = j;
        break;
      }
    for(int row = 0; row < 7; row++)
      for(int col = 0; col < 5; col++)
        if(s_glyphs[idx][row] & (1 << (4 - col)))
          setPixel(px, ox + i * 6 + col, oy + row, r, g, b);
  }
}

// Generate a s_TEX_SIZE x s_TEX_SIZE RGBA texture for one cube face.
inline void generateFaceTexture(std::vector<uint8_t>& px, uint8_t ir, uint8_t ig, uint8_t ib, const char* faceName)
{
  px.resize(s_TEX_SIZE * s_TEX_SIZE * 4);

  // Fill solid color
  for(int i = 0; i < s_TEX_SIZE * s_TEX_SIZE; i++)
  {
    px[i * 4]     = ir;
    px[i * 4 + 1] = ig;
    px[i * 4 + 2] = ib;
    px[i * 4 + 3] = 255;
  }

  // Dark border
  uint8_t br = ir / 3, bg = ig / 3, bb = ib / 3;
  for(int i = 0; i < s_TEX_SIZE; i++)
  {
    setPixel(px, i, 0, br, bg, bb);
    setPixel(px, i, s_TEX_SIZE - 1, br, bg, bb);
    setPixel(px, 0, i, br, bg, bb);
    setPixel(px, s_TEX_SIZE - 1, i, br, bg, bb);
  }

  // Text: white or black depending on luminance
  uint8_t tc = (ir * 0.299f + ig * 0.587f + ib * 0.114f) > 128.0f ? 0 : 255;
  char    hex[8];
  snprintf(hex, sizeof(hex), "#%02X%02X%02X", ir, ig, ib);
  drawString(px, 3, 3, hex, tc, tc, tc);
  drawString(px, 3, 13, faceName, tc, tc, tc);
}

}  // namespace font_texture

#endif
