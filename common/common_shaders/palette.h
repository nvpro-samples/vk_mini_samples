//https://iquilezles.org/articles/palettes/
float3 palette(float t)
{
  float3 a = float3(0.5, 0.5, 0.5);
  float3 b = float3(0.5, 0.5, 0.5);
  float3 c = float3(1.0, 1.0, 0.5);
  float3 d = float3(0.0, 0.15, 0.2);

  return a + b * cos(6.28318f * (c * t + d));
}
