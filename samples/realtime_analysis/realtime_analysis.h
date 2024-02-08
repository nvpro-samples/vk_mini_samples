#pragma once

enum
{
  eCalculateDensitiesShd,
  eCalculatePressureForceShd,
  eCalculateViscosityShd,
  eExternalForcesShd,
  eUpdatePositionsShd,
  eUpdateSpatialHashShd,
  eBitonicSort,
  eBitonicSortOffsets,
  numCompShaders
};

static DH::ParticleSetting TestA = {
    .gravity                = -9.8f,
    .collisionDamping       = 0.25f,
    .smoothingRadius        = 0.25f,
    .targetDensity          = 200,
    .pressureMultiplier     = 15,
    .nearPressureMultiplier = 15,
    .viscosityStrength      = 0.01f,
    .boundsMultiplier       = 10,
};

static DH::ParticleSetting TestB = {
    .gravity                = 0,
    .collisionDamping       = 0.95f,
    .smoothingRadius        = 0.35f,
    .targetDensity          = 200,
    .pressureMultiplier     = 40,
    .nearPressureMultiplier = 30,
    .viscosityStrength      = 0.06f,
    .boundsMultiplier       = 12,
};

static DH::ParticleSetting TestC = {
    .gravity                = -3,
    .collisionDamping       = 0.90f,
    .smoothingRadius        = 0.65f,
    .targetDensity          = 200,
    .pressureMultiplier     = 50,
    .nearPressureMultiplier = 20,
    .viscosityStrength      = 0.04f,
    .boundsMultiplier       = 10,
};

static DH::ParticleSetting TestD = {
    .gravity                = 0,
    .collisionDamping       = 0.95f,
    .smoothingRadius        = 0.3f,
    .targetDensity          = 13,
    .pressureMultiplier     = 1,
    .nearPressureMultiplier = 1,
    .viscosityStrength      = 0.02f,
    .boundsMultiplier       = 9,
};


template <typename T>  // Return memory usage size
inline size_t getShaderSize(const std::vector<T>& vec)
{
  using baseType = typename std::remove_reference<T>::type;
  return sizeof(baseType) * vec.size();
}
