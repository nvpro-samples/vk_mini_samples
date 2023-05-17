// Main ShaderToy tab, the result fragColor will be displayed
// Image shaders implement the mainImage() function in order to generate the procedural images by
// computing a color for each pixel. This function is expected to be called once per pixel, and
// it is responsability of the host application to provide the right inputs to it and get the
// output color from it and assign it to the screen pixel.

// Input Uniforms
/**
Shader can be fed with different types of per-frame static information by using the following uniform variables:
uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform float iFrame;
uniform float iChannelTime[1];
uniform vec4 iMouse;
uniform vec3 iChannelResolution[1];
uniform samplerXX iChanneli;

Ex: to read from result of buffer_a
    fragColor = texelFetch(iChannel0, ivec2(fragCoord), 0);

**/


void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
  // From https://www.shadertoy.com/view/XsVSzW
  float time = iTime * 0.5;
  vec2  uv   = (fragCoord.xy / iResolution.xx - 0.5) * 5.0;
  float i0   = 1.0;
  float i1   = 1.0;
  float i2   = 1.0;
  float i4   = 0.0;
  for(int s = 0; s < 10; s++)
  {
    vec2 r;
    r = vec2(cos(uv.y * i0 - i4 + time / i1), sin(uv.x * i0 - i4 + time / i1)) / i2;
    r += vec2(-r.y, r.x) * 0.3;
    uv.xy += r;

    i0 *= 1.93;
    i1 *= 1.15;
    i2 *= 1.7;
    i4 += 0.05 + 0.1 * time * i1;
  }
  float r = sin(uv.x - time) * 0.5 + 0.5;
  float b = sin(uv.y + time) * 0.5 + 0.5;
  float g = sin((uv.x + uv.y + sin(time * 0.5)) * 0.5) * 0.5 + 0.5;

  fragColor = vec4(r, g, b, 1.0);
}
