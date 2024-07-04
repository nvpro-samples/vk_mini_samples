
float smoothingKernelPoly6(float dst, float radius)
{
  if(dst < radius)
  {
    float v = radius * radius - dst * dst;
    return v * v * v * setting.poly6ScalingFactor;
  }
  return 0;
}

float spikyKernelPow3(float dst, float radius)
{
  if(dst < radius)
  {
    float v = radius - dst;
    return v * v * v * setting.spikyPow3ScalingFactor;
  }
  return 0;
}

float spikyKernelPow2(float dst, float radius)
{
  if(dst < radius)
  {
    float v = radius - dst;
    return v * v * setting.spikyPow2ScalingFactor;
  }
  return 0;
}

float derivativeSpikyPow3(float dst, float radius)
{
  if(dst <= radius)
  {
    float v = radius - dst;
    return -v * v * setting.spikyPow3DerivativeScalingFactor;
  }
  return 0;
}

float derivativeSpikyPow2(float dst, float radius)
{
  if(dst <= radius)
  {
    float v = radius - dst;
    return -v * setting.spikyPow2DerivativeScalingFactor;
  }
  return 0;
}
