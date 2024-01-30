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

static ParticleSetting TestA = {
    .gravity                = -9.8,
    .collisionDamping       = 0.25,
    .smoothingRadius        = 0.25,
    .targetDensity          = 200,
    .pressureMultiplier     = 15,
    .nearPressureMultiplier = 15,
    .viscosityStrength      = 0.01,
    .boundsMultiplier       = 10,
};

static ParticleSetting TestB = {
    .gravity                = 0,
    .collisionDamping       = 0.95,
    .smoothingRadius        = 0.35,
    .targetDensity          = 200,
    .pressureMultiplier     = 40,
    .nearPressureMultiplier = 30,
    .viscosityStrength      = 0.06,
    .boundsMultiplier       = 12,
};

static ParticleSetting TestC = {
    .gravity                = -3,
    .collisionDamping       = 0.90,
    .smoothingRadius        = 0.65,
    .targetDensity          = 200,
    .pressureMultiplier     = 50,
    .nearPressureMultiplier = 20,
    .viscosityStrength      = 0.04,
    .boundsMultiplier       = 10,
};

static ParticleSetting TestD = {
    .gravity                = 0,
    .collisionDamping       = 0.95,
    .smoothingRadius        = 0.3,
    .targetDensity          = 13,
    .pressureMultiplier     = 1,
    .nearPressureMultiplier = 1,
    .viscosityStrength      = 0.02,
    .boundsMultiplier       = 9,
};


template <typename T>  // Return memory usage size
inline size_t getShaderSize(const std::vector<T>& vec)
{
  using baseType = typename std::remove_reference<T>::type;
  return sizeof(baseType) * vec.size();
}
