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


#ifndef GGX_HLSLI
#define GGX_HLSLI 1

#include "constants.hlsli"
#include "functions.hlsli"
#include "pbr_mat_struct.hlsli"

#define OUT_TYPE(T) out T
#define INOUT_TYPE(T) inout T
#define ARRAY_TYPE(T, N, name) T name[N]


#define BSDF_EVENT_ABSORB 0
#define BSDF_EVENT_DIFFUSE 1
#define BSDF_EVENT_GLOSSY (1 << 1)
#define BSDF_EVENT_SPECULAR (1 << 2)
#define BSDF_EVENT_REFLECTION (1 << 3)
#define BSDF_EVENT_TRANSMISSION (1 << 4)

#define BSDF_EVENT_DIFFUSE_REFLECTION (BSDF_EVENT_DIFFUSE | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_DIFFUSE_TRANSMISSION (BSDF_EVENT_DIFFUSE | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_GLOSSY_REFLECTION (BSDF_EVENT_GLOSSY | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_GLOSSY_TRANSMISSION (BSDF_EVENT_GLOSSY | BSDF_EVENT_TRANSMISSION)
#define BSDF_EVENT_SPECULAR_REFLECTION (BSDF_EVENT_SPECULAR | BSDF_EVENT_REFLECTION)
#define BSDF_EVENT_SPECULAR_TRANSMISSION (BSDF_EVENT_SPECULAR | BSDF_EVENT_TRANSMISSION)

#define BSDF_USE_MATERIAL_IOR (-1.0F)

#ifndef GGX_MIN_ALPHA_ROUGHNESS
#define GGX_MIN_ALPHA_ROUGHNESS 1e-03F
#endif


float3 make_float3(float a) 
{
 return float3(a, a, a);
}

/** @DOC_START
# struct BsdfEvaluateData
>  Data structure for evaluating a BSDF
@DOC_END */
struct BsdfEvaluateData
{
  float3  k1;            // [in] Toward the incoming ray
  float3  k2;            // [in] Toward the sampled light
  float3  xi;            // [in] 3 random [0..1]
  float3  bsdf_diffuse;  // [out] Diffuse contribution
  float3  bsdf_glossy;   // [out] Specular contribution
  float pdf;           // [out] PDF
};

/** @DOC_START
# struct BsdfSampleData
>  Data structure for sampling a BSDF
@DOC_END  */
struct BsdfSampleData
{
  float3  k1;             // [in] Toward the incoming ray
  float3  k2;             // [out] Toward the sampled light
  float3  xi;             // [in] 3 random [0..1]
  float pdf;            // [out] PDF
  float3  bsdf_over_pdf;  // [out] contribution / PDF
  int   event_type;     // [out] one of the event above
};

float3 absorptionCoefficient(PbrMaterial mat)
{
  float tmp1 = mat.attenuationDistance;
  return tmp1 <= 0.0F ? float3(0.0F, 0.0F, 0.0F) :
                        -float3(log(mat.attenuationColor.x), log(mat.attenuationColor.y), log(mat.attenuationColor.z)) / tmp1;
}


float3 schlickFresnel(float3 F0, float3 F90, float VdotH)
{
  return F0 + (F90 - F0) * pow(1.0F - VdotH, 5.0F);
}

float schlickFresnel(float ior, float VdotH)
{
  // Calculate reflectance at normal incidence (R0)
  float R0 = pow((1.0F - ior) / (1.0F + ior), 2.0F);

  // Fresnel reflectance using Schlick's approximation
  return R0 + (1.0F - R0) * pow(1.0F - VdotH, 5.0F);
}

//-----------------------------------------------------------------------
// Visibility term for the GGX BRDF, using the height-correlated
// Smith shadowing-masking function.
// Note: Vis = G / (4 * NdotL * NdotV), where G is the usual Smith term.
// see Eric Heitz. 2014. Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs. Journal of Computer Graphics Techniques, 3
// see Real-Time Rendering, 4th edition, page 341, equation (9.43).
// see V_SmithGGXCorrelated in https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf/geometricshadowing(specularg)
//-----------------------------------------------------------------------
float ggxSmithVisibility(float NdotL, float NdotV, float alphaRoughness)
{
  alphaRoughness         = max(alphaRoughness, GGX_MIN_ALPHA_ROUGHNESS);
  float alphaRoughnessSq = alphaRoughness * alphaRoughness;

  float ggxV = NdotL * sqrt(NdotV * NdotV * (1.0F - alphaRoughnessSq) + alphaRoughnessSq);
  float ggxL = NdotV * sqrt(NdotL * NdotL * (1.0F - alphaRoughnessSq) + alphaRoughnessSq);

  return 0.5F / (ggxV + ggxL);
}

//-----------------------------------------------------------------------
// The D() term in the GGX BRDF. This models the distribution of microfacet
// normals across the material.
// Implementation from "Average Irregularity Representation of a Roughened Surface for Ray Reflection" by T. S. Trowbridge, and K. P. Reitz
// This is also equivalent to Equation 4 in https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf,
// using alphaRoughness == roughness^2 (following the Burley parameterization
// used in glTF.)
// It's the distribution of normals of an ellipsoid scaled by
// (alphaRoughness, alphaRoughness, 1.f).
//-----------------------------------------------------------------------
float ggxDistribution(float NdotH, float alphaRoughness)  // alphaRoughness    = roughness * roughness;
{
  if(NdotH < 0.0f)
  {
    return 0.0f;
  }
  alphaRoughness = max(alphaRoughness, GGX_MIN_ALPHA_ROUGHNESS);
  float alphaSqr = alphaRoughness * alphaRoughness;

  float NdotHSqr = NdotH * NdotH;
  float denom    = NdotHSqr * (alphaSqr - 1.0F) + 1.0F;

  return alphaSqr / (M_PI * denom * denom);
}

//-----------------------------------------------------------------------
// The single-scattering GGX BRDF (reflection only).
// See Section 4.4 of https://google.github.io/filament/Filament.md.html#materialsystem/specularbrdf
// Also see https://www.pbr-book.org/4ed/Reflection_Models/Conductor_BRDF.
// This assumes you've already flipped the normal so that NdotV >= 0.f.
// alphaRoughness is roughness * roughness.
//-----------------------------------------------------------------------
float3 ggxBRDF(float3 f0, float3 f90, float alphaRoughness, float VdotH, float NdotL, float NdotV, float NdotH)
{
  if(NdotV <= 0.0f || NdotL <= 0.0f)
    return float3(0.f,0.f,0.f);  // Opposite sides, or at the horizon

  float3  f   = schlickFresnel(f0, f90, VdotH);
  float vis = ggxSmithVisibility(NdotL, NdotV, alphaRoughness);  // Vis = G / (4 * NdotL * NdotV)
  float d   = ggxDistribution(NdotH, alphaRoughness);

  return f * vis * d;
}

//-----------------------------------------------------------------------
// Samples the GGX distribution, returning the half vector (i.e. the sampled
// microfacet normal) of a surface with normal (0,0,1).
// This samples from the set of visible normals (aka the VNDF), so it requires
// the input direction.
//-----------------------------------------------------------------------
float3 ggxSamplingHalfVector(float3 wi, float alphaRoughness, float2 rand)
{
  alphaRoughness = max(alphaRoughness, GGX_MIN_ALPHA_ROUGHNESS);
  // We use the implementation from Dupuy and Benyoub,
  // "Sampling Visible GGX Normals with Spherical Caps", 2023,
  // https://arxiv.org/pdf/2306.05044, Listings 1 and 3.

  // Warp the ellipsoid of normals to the unit hemisphere, and warp `wi` along
  // with it:
  float3 wi_std = normalize(float3(wi.xy * alphaRoughness, wi.z));
  // If we were to reflect `wi_std` by the visible normals, we'd get points
  // distributed in a spherical cap. Find a lower bound for this spherical cap:
  float lower_bound = -wi_std.z;
  // Sample the spherical cap:
  float z         = lerp(lower_bound, 1.0f, rand.y);
  float sin_theta = sqrt(max(0.f, 1.0f - z * z));
  float phi       = M_TWO_PI * rand.x;
  float3  wo_std    = float3(sin_theta * cos(phi), sin_theta * sin(phi), z);
  // Compute the microfacet normal:
  float3 m_std = wi_std + wo_std;
  // Warp the unit hemisphere to the ellipsoid of normals. Since we now have
  // a microfacet normal instead of a direction, the math is the same, since
  // we use the inverse transpose here:
  return normalize(float3(m_std.xy * alphaRoughness, m_std.z));
}


//-----------------------------------------------------------------------
// Sample the GGX distribution
// - Return the half vector
//-----------------------------------------------------------------------
float3 ggxSampling(float alphaRoughness, float2 rand)
{
  float alphaSqr = max(alphaRoughness * alphaRoughness, 1e-07F);

  float phi      = M_TWO_PI * rand.x;
  float cosTheta = sqrt((1.0F - rand.y) / (1.0F + (alphaSqr - 1.0F) * rand.y));
  float sinTheta = sqrt(1.0F - cosTheta * cosTheta);

  return float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

//-----------------------------------------------------------------------
// Returns the PDF for ggxSampling, when used for reflection.
// Counterintuitively, this mostly depends on dot(N, V). The reason is that
// the size of the spherical cap that microsurface normals are sampled from
// depends on the view vector
//-----------------------------------------------------------------------
float ggxSamplingReflectionPDF(float NdotV, float NdotH, float alphaRoughness)
{
  if(NdotH <= 0.0f)
    return 0.0f;
  alphaRoughness = max(alphaRoughness, GGX_MIN_ALPHA_ROUGHNESS);
  // @nbickford simplified this to work with only NdotV -- but this only
  // works if roughness is isotropic!
  // t is sqrt((l.x * alphaRoughness)^2 + (l.y * alphaRoughness)^2 + l.z^2)
  //   == sqrt(alphaRoughness^2 + l.z^2 - (alphaRoughness*l.z)^2)
  float a2     = alphaRoughness * alphaRoughness;
  float NdotV2 = NdotV * NdotV;
  float t      = sqrt(a2 + (1.0f - a2) * NdotV2);
  return 0.5f * ggxDistribution(NdotH, alphaRoughness) / (abs(NdotV) + t);
}


//-----------------------------------------------------------------------
// MDL Bases functions
//-----------------------------------------------------------------------

float3 mix_rgb(const float3 base, const float3 layer, const float3 factor)
{
  return (1.0f - max(factor.x, max(factor.y, factor.z))) * base + factor * layer;
}

// Return the sin and cos of the input angle
void sincos(float angle, OUT_TYPE(float) si, OUT_TYPE(float) co)
{
  si = sin(angle);
  co = cos(angle);
}

// Squere the input
float sqr(float x)
{
  return x * x;
}

// Check for total internal reflection.
bool isTIR(const float2 ior, const float kh)
{
  const float b = ior.x / ior.y;
  return (1.0f < (b * b * (1.0f - kh * kh)));
}


// Evaluate anisotropic GGX distribution on the non-projected hemisphere.
float hvd_ggx_eval(const float2 invAlpha,
                   const float3 h)  // == float3(dot(tangent, h), dot(bitangent, h), dot(normal, h))
{
  const float x     = h.x * invAlpha.x;
  const float y     = h.y * invAlpha.y;
  const float aniso = x * x + y * y;
  const float f     = aniso + h.z * h.z;

  return M_1_PI * invAlpha.x * invAlpha.y * h.z / (f * f);
}

// sample visible (Smith-masked) half vector according to the anisotropic GGX distribution
// (see Eric Heitz - A Simpler and Exact Sampling Routine for the GGX Distribution of Visible
// normals)
float3 hvd_ggx_sample_vndf(float3 k, float2 roughness, float2 xi)
{
  const float3 v = normalize(float3(k.x * roughness.x, k.y * roughness.y, k.z));

  const float3 t1 = (v.z < 0.99999f) ? normalize(cross(v, float3(0.0f, 0.0f, 1.0f))) : float3(1.0f, 0.0f, 0.0f);
  const float3 t2 = cross(t1, v);

  const float a = 1.0f / (1.0f + v.z);
  const float r = sqrt(xi.x);

  const float phi = (xi.y < a) ? xi.y / a * M_PI : M_PI + (xi.y - a) / (1.0f - a) * M_PI;
  float       sp, cp;
  sincos(phi, sp, cp);
  const float p1 = r * cp;
  const float p2 = r * sp * ((xi.y < a) ? 1.0f : v.z);

  float3 h = p1 * t1 + p2 * t2 + sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

  h.x *= roughness.x;
  h.y *= roughness.y;
  h.z = max(0.0f, h.z);
  return normalize(h);
}

// Smith-masking for anisotropic GGX
float smith_shadow_mask(const float3 k, const float2 roughness)
{
  const float ax     = k.x * roughness.x;
  const float ay     = k.y * roughness.y;
  const float inv_a2 = (ax * ax + ay * ay) / (k.z * k.z);

  return 2.0f / (1.0f + sqrt(1.0f + inv_a2));
}


float ggx_smith_shadow_mask(OUT_TYPE(float) G1, OUT_TYPE(float) G2, const float3 k1, const float3 k2, const float2 roughness)
{
  G1 = smith_shadow_mask(k1, roughness);
  G2 = smith_shadow_mask(k2, roughness);

  return G1 * G2;
}


// Compute squared norm of s/p polarized Fresnel reflection coefficients and phase shifts in complex unit circle.
// Born/Wolf - "Principles of Optics", section 13.4
float2 fresnel_conductor(OUT_TYPE(float2) phase_shift_sin,
                       OUT_TYPE(float2) phase_shift_cos,
                       const float n_a,
                       const float n_b,
                       const float k_b,
                       const float cos_a,
                       const float sin_a_sqd)
{
  const float k_b2   = k_b * k_b;
  const float n_b2   = n_b * n_b;
  const float n_a2   = n_a * n_a;
  const float tmp0   = n_b2 - k_b2;
  const float half_U = 0.5f * (tmp0 - n_a2 * sin_a_sqd);
  const float half_V = sqrt(max(0.0f, half_U * half_U + k_b2 * n_b2));

  const float u_b2 = half_U + half_V;
  const float v_b2 = half_V - half_U;
  const float u_b  = sqrt(max(0.0f, u_b2));
  const float v_b  = sqrt(max(0.0f, v_b2));

  const float tmp1 = tmp0 * cos_a;
  const float tmp2 = n_a * u_b;
  const float tmp3 = (2.0f * n_b * k_b) * cos_a;
  const float tmp4 = n_a * v_b;
  const float tmp5 = n_a * cos_a;

  const float tmp6 = (2.0f * tmp5) * v_b;
  const float tmp7 = (u_b2 + v_b2) - tmp5 * tmp5;

  const float tmp8 = (2.0f * tmp5) * ((2.0f * n_b * k_b) * u_b - tmp0 * v_b);
  const float tmp9 = sqr((n_b2 + k_b2) * cos_a) - n_a2 * (u_b2 + v_b2);

  const float tmp67      = tmp6 * tmp6 + tmp7 * tmp7;
  const float inv_sqrt_x = (0.0f < tmp67) ? (1.0f / sqrt(tmp67)) : 0.0f;
  const float tmp89      = tmp8 * tmp8 + tmp9 * tmp9;
  const float inv_sqrt_y = (0.0f < tmp89) ? (1.0f / sqrt(tmp89)) : 0.0f;

  phase_shift_cos = float2(tmp7 * inv_sqrt_x, tmp9 * inv_sqrt_y);
  phase_shift_sin = float2(tmp6 * inv_sqrt_x, tmp8 * inv_sqrt_y);

  return float2((sqr(tmp5 - u_b) + v_b2) / (sqr(tmp5 + u_b) + v_b2),
              (sqr(tmp1 - tmp2) + sqr(tmp3 - tmp4)) / (sqr(tmp1 + tmp2) + sqr(tmp3 + tmp4)));
}


// Simplified for dielectric, no phase shift computation.
float2 fresnel_dielectric(const float n_a, const float n_b, const float cos_a, const float cos_b)
{
  const float naca = n_a * cos_a;
  const float nbcb = n_b * cos_b;
  const float r_s  = (naca - nbcb) / (naca + nbcb);

  const float nacb = n_a * cos_b;
  const float nbca = n_b * cos_a;
  const float r_p  = (nbca - nacb) / (nbca + nacb);

  return float2(r_s * r_s, r_p * r_p);
}


float3 thin_film_factor(float coating_thickness, const float coating_ior, const float base_ior, const float incoming_ior, const float kh)
{
  coating_thickness = max(0.0f, coating_thickness);

  const float sin0_sqr  = max(0.0f, 1.0f - kh * kh);
  const float eta01     = incoming_ior / coating_ior;
  const float eta01_sqr = eta01 * eta01;
  const float sin1_sqr  = eta01_sqr * sin0_sqr;

  if(1.0f < sin1_sqr)  // TIR at first interface
  {
    return float3(1.0f,1.0f,1.0f);
  }

  const float cos1 = sqrt(max(0.0f, 1.0f - sin1_sqr));
  const float2  R01  = fresnel_dielectric(incoming_ior, coating_ior, kh, cos1);

  float2       phi12_sin, phi12_cos;
  const float2 R12 = fresnel_conductor(phi12_sin, phi12_cos, coating_ior, base_ior, /* base_k = */ 0.0f, cos1, sin1_sqr);

  const float tmp = (4.0f * M_PI) * coating_ior * coating_thickness * cos1;

  const float R01R12_s = max(0.0f, R01.x * R12.x);
  const float r01r12_s = sqrt(R01R12_s);

  const float R01R12_p = max(0.0f, R01.y * R12.y);
  const float r01r12_p = sqrt(R01R12_p);

  float3 xyz = float3(0.0f,0.0f,0.0f);

  //!! using low res color matching functions here
  float lambda_min  = 400.0f;
  float lambda_step = ((700.0f - 400.0f) / 16.0f);

  const float3 cie_xyz[16] = {{0.02986f, 0.00310f, 0.13609f}, {0.20715f, 0.02304f, 0.99584f},
                            {0.36717f, 0.06469f, 1.89550f}, {0.28549f, 0.13661f, 1.67236f},
                            {0.08233f, 0.26856f, 0.76653f}, {0.01723f, 0.48621f, 0.21889f},
                            {0.14400f, 0.77341f, 0.05886f}, {0.40957f, 0.95850f, 0.01280f},
                            {0.74201f, 0.97967f, 0.00060f}, {1.03325f, 0.84591f, 0.00000f},
                            {1.08385f, 0.62242f, 0.00000f}, {0.79203f, 0.36749f, 0.00000f},
                            {0.38751f, 0.16135f, 0.00000f}, {0.13401f, 0.05298f, 0.00000f},
                            {0.03531f, 0.01375f, 0.00000f}, {0.00817f, 0.00317f, 0.00000f}};

  float lambda = lambda_min + 0.5f * lambda_step;

  for(int i = 0; i < 16; ++i)
  {
    const float phi = tmp / lambda;

    float phi_s = sin(phi);
    float phi_c = cos(phi);

    const float cos_phi_s = phi_c * phi12_cos.x - phi_s * phi12_sin.x;  // cos(a+b) = cos(a) * cos(b) - sin(a) * sin(b)
    const float tmp_s     = 2.0f * r01r12_s * cos_phi_s;
    const float R_s       = (R01.x + R12.x + tmp_s) / (1.0f + R01R12_s + tmp_s);

    const float cos_phi_p = phi_c * phi12_cos.y - phi_s * phi12_sin.y;  // cos(a+b) = cos(a) * cos(b) - sin(a) * sin(b)
    const float tmp_p     = 2.0f * r01r12_p * cos_phi_p;
    const float R_p       = (R01.y + R12.y + tmp_p) / (1.0f + R01R12_p + tmp_p);

    xyz += cie_xyz[i] * (0.5f * (R_s + R_p));

    lambda += lambda_step;
  }

  xyz *= (1.0f / 16.0f);

  // ("normalized" such that the loop for no shifted wave gives reflectivity (1,1,1))
  return clamp(float3(xyz.x * (3.2406f / 0.433509f) + xyz.y * (-1.5372f / 0.433509f) + xyz.z * (-0.4986f / 0.433509f),
                    xyz.x * (-0.9689f / 0.341582f) + xyz.y * (1.8758f / 0.341582f) + xyz.z * (0.0415f / 0.341582f),
                    xyz.x * (0.0557f / 0.32695f) + xyz.y * (-0.204f / 0.32695f) + xyz.z * (1.057f / 0.32695f)),
               0.0f, 1.0f);
}


// Compute half vector (convention: pointing to outgoing direction, like shading normal)
float3 compute_half_vector(const float3 k1, const float3 k2, const float3 normal, const float2 ior, const float nk2, const bool transmission, const bool thinwalled)
{
  float3 h;

  if(transmission)
  {
    if(thinwalled)  // No refraction!
    {
      h = k1 + (normal * (nk2 + nk2) + k2);  // Use corresponding reflection direction.
    }
    else
    {
      h = k2 * ior.y + k1 * ior.x;  // Points into thicker medium.

      if(ior.y > ior.x)
      {
        h *= -1.0f;  // Make pointing to outgoing direction's medium.
      }
    }
  }
  else
  {
    h = k1 + k2;  // unnormalized half-vector
  }

  return normalize(h);
}

float3 refract(const float3  k,       // direction (pointing from surface)
             const float3  n,       // normal
             const float b,       // (reflected side IOR) / (transmitted side IOR)
             const float nk,      // dot(n, k)
             OUT_TYPE(bool) tir)  // total internal reflection
{
  const float refraction = b * b * (1.0f - nk * nk);

  tir = (1.0f <= refraction);

  return (tir) ? (n * (nk + nk) - k) : normalize((-k * b + n * (b * nk - sqrt(1.0f - refraction))));
}


// Fresnel equation for an equal mix of polarization.
float ior_fresnel(const float eta,  // refracted / reflected ior
                  const float kh)   // cosine of angle between normal/half-vector and direction
{
  float costheta = 1.0f - (1.0f - kh * kh) / (eta * eta);

  if(costheta <= 0.0f)
  {
    return 1.0f;
  }

  costheta = sqrt(costheta);  // refracted angle cosine

  const float n1t1 = kh;
  const float n1t2 = costheta;
  const float n2t1 = kh * eta;
  const float n2t2 = costheta * eta;
  const float r_p  = (n1t2 - n2t1) / (n1t2 + n2t1);
  const float r_o  = (n1t1 - n2t2) / (n1t1 + n2t2);

  const float fres = 0.5f * (r_p * r_p + r_o * r_o);

  return clamp(fres, 0.0f, 1.0f);
}

// Evaluate anisotropic sheen half-vector distribution on the non-projected hemisphere.
float hvd_sheen_eval(const float invRoughness,
                     const float nh)  // dot(shading_normal, h)
{
  const float sinTheta2 = max(0.0f, 1.0f - nh * nh);
  const float sinTheta  = sqrt(sinTheta2);

  return (invRoughness + 2.0f) * pow(sinTheta, invRoughness) * 0.5f * M_1_PI * nh;
}


// Cook-Torrance style v-cavities masking term.
float vcavities_mask(const float nh,  // abs(dot(normal, half))
                     const float kh,  // abs(dot(dir, half))
                     const float nk)  // abs(dot(normal, dir))
{
  return min(2.0f * nh * nk / kh, 1.0f);
}


float vcavities_shadow_mask(OUT_TYPE(float) G1, OUT_TYPE(float) G2, const float nh, const float3 k1, const float k1h, const float3 k2, const float k2h)
{
  G1 = vcavities_mask(nh, k1h, k1.z);  // In my renderer the z-coordinate is the normal!
  G2 = vcavities_mask(nh, k2h, k2.z);

  //return (refraction) ? fmaxf(0.0f, G1 + G2 - 1.0f) : fminf(G1, G2);
  return min(G1, G2);  // PERF Need reflection only.
}


// Sample half-vector according to anisotropic sheen distribution.
float3 hvd_sheen_sample(const float2 xi, const float invRoughness)
{
  const float phi = 2.0f * M_PI * xi.x;

  float sinPhi = sin(phi);
  float cosPhi = cos(phi);

  const float sinTheta = pow(1.0f - xi.y, 1.0f / (invRoughness + 2.0f));
  const float cosTheta = sqrt(1.0f - sinTheta * sinTheta);

  return normalize(float3(cosPhi * sinTheta, sinPhi * sinTheta,
                        cosTheta));  // In my renderer the z-coordinate is the normal!
}

float3 flip(const float3 h, const float3 k, float xi)
{
  const float a = h.z * k.z;
  const float b = h.x * k.x + h.y * k.y;

  const float kh   = max(0.0f, a + b);
  const float kh_f = max(0.0f, a - b);

  const float p_flip = kh_f / (kh + kh_f);

  // PERF xi is not used after this operation by the only caller brdf_sheen_sample(),
  // so there is no need to scale the sample.
  //if (xi < p_flip)
  //{
  //  xi /= p_flip;
  //  return make_float3(-h.x, -h.y, h.z);
  //}
  //else
  //{
  //  xi = (xi - p_flip) / (1.0f - p_flip);
  //  return h;
  //}

  return (xi < p_flip) ? float3(-h.x, -h.y, h.z) : h;
}

//////////////////////////////////////////////////////////////////////////


#define LOBE_DIFFUSE_REFLECTION 0
#define LOBE_SPECULAR_TRANSMISSION 1
#define LOBE_SPECULAR_REFLECTION 2
#define LOBE_METAL_REFLECTION 3
#define LOBE_SHEEN_REFLECTION 4
#define LOBE_CLEARCOAT_REFLECTION 5
#define LOBE_COUNT 6

// The Fresnel factor depends on the cosine between the view vector k1 and the
// half vector, H = normalize(k1 + k2). But during sampling, we don't have k2
// until we sample a microfacet. So instead, we approximate it.
// For a mirror surface, we have H == N. For a perfectly diffuse surface, k2
// is sampled in a cosine distribution around N, so H ~ normalize(k1 + N).
// We ad-hoc interpolate between them using the roughness.
float fresnelCosineApproximation(float VdotN, float roughness)
{
  return lerp(VdotN, sqrt(0.5F + 0.5F * VdotN), sqrt(roughness));
}

// Calculate the weights of the individual lobes inside the standard PBR material.
void computeLobeWeights(PbrMaterial mat, float VdotN, INOUT_TYPE(float3) tint, out float weightLobe[LOBE_COUNT])
{
  float frCoat = 0.0F;
  if(mat.clearcoat > 0.0f)
  {
    float frCosineClearcoat = fresnelCosineApproximation(VdotN, mat.clearcoatRoughness);
    frCoat                  = mat.clearcoat * ior_fresnel(1.5f / mat.ior1, frCosineClearcoat);
  }

  // This Fresnel value defines the weighting between dielectric specular reflection and
  // the base dielectric BXDFs (diffuse reflection and specular transmission).
  float frDielectric = 0;
  if(mat.specular > 0)
  {
    float frCosineDielectric = fresnelCosineApproximation(VdotN, (mat.roughness.x + mat.roughness.y) * 0.5F);
    frDielectric             = ior_fresnel(mat.ior2 / mat.ior1, frCosineDielectric);
    frDielectric *= mat.specular;
  }

  // Estimate the iridescence Fresnel factor with the angle to the normal, and
  // blend it in. That's good enough for specular reflections.
  if(mat.iridescence > 0.0f)
  {
    // When there is iridescence enabled, use the maximum of the estimated iridescence factor. (Estimated with VdotN, no half-vector H here.)
    // With the thinfilm decision this handles the mix between non-iridescence and iridescence strength automatically.
    float3 frIridescence = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, VdotN);
    frDielectric = lerp(frDielectric, max(frIridescence.x, max(frIridescence.y, frIridescence.z)), mat.iridescence);
    // Modulate the dielectric base lobe (diffuse, transmission) colors by the inverse of the iridescence factor,
    // though use the maximum component to not actually generate inverse colors.
    tint = mix_rgb(tint, mat.specularColor, frIridescence * mat.iridescence);
  }

  float sheen = 0.0f;
  if((mat.sheenColor.r != 0.0F || mat.sheenColor.g != 0.0F || mat.sheenColor.b != 0.0F))
  {
    sheen = pow(1.0f - abs(VdotN), mat.sheenRoughness);  // * luminance(mat.sheenColor);
    sheen = sheen / (sheen + 0.5F);
  }

  /*
  Lobe weights:

    - Clearcoat       : clearcoat * schlickFresnel(1.5, VdotN)
    - Sheen           : sheen
    - Metal           : metallic
    - Specular        : specular * schlickFresnel(ior, VdotN)
    - Transmission    : transmission
    - Diffuse         : 1.0 - clearcoat - sheen - metallic - specular - transmission
  */

  weightLobe[LOBE_CLEARCOAT_REFLECTION]  = 0;
  weightLobe[LOBE_SHEEN_REFLECTION]      = 0;
  weightLobe[LOBE_METAL_REFLECTION]      = 0;
  weightLobe[LOBE_SPECULAR_REFLECTION]   = 0;
  weightLobe[LOBE_SPECULAR_TRANSMISSION] = 0;
  weightLobe[LOBE_DIFFUSE_REFLECTION]    = 0;

  float weightBase = 1.0F;

  weightLobe[LOBE_CLEARCOAT_REFLECTION] = frCoat;  // BRDF clearcoat reflection (GGX-Smith)
  weightBase *= 1.0f - frCoat;

  weightLobe[LOBE_SHEEN_REFLECTION] = weightBase * sheen;  // BRDF sheen reflection (Lambert)
  weightBase *= 1.0f - sheen;

  weightLobe[LOBE_METAL_REFLECTION] = weightBase * mat.metallic;  // BRDF metal (GGX-Smith)
  weightBase *= 1.0f - mat.metallic;

  weightLobe[LOBE_SPECULAR_REFLECTION] = weightBase * frDielectric;  // BRDF dielectric specular reflection (GGX-Smith)
  weightBase *= 1.0f - frDielectric;

  weightLobe[LOBE_SPECULAR_TRANSMISSION] = weightBase * mat.transmission;  // BTDF dielectric specular transmission (GGX-Smith)
  weightLobe[LOBE_DIFFUSE_REFLECTION] = weightBase * (1.0f - mat.transmission);  // BRDF diffuse dielectric reflection (Lambert). // PERF Currently not referenced below.
  
}

// Calculate the weights of the individual lobes inside the standard PBR material
// and randomly select one.
int findLobe(PbrMaterial mat, float VdotN, float rndVal, INOUT_TYPE(float3) tint)
{
  float weightLobe[LOBE_COUNT]; 
  computeLobeWeights(mat, VdotN, tint, weightLobe);

  int   lobe   = LOBE_COUNT;
  float weight = 0.0f;
  while(--lobe > 0)  // Stops when lobe reaches 0!
  {
    weight += weightLobe[lobe];
    if(rndVal < weight)
    {
      break;  // Sample and evaluate this lobe!
    }
  }

  return lobe;  // Last one is the diffuse reflection
}

void brdf_diffuse_eval(INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, float3 tint)
{
  // If the incoming light direction is on the backside, there is nothing to evaluate for a BRDF.
  // Note that the state normals have been flipped to the ray side by the caller.
  // Include edge-on (== 0.0f) as "no light" case.
  if(dot(data.k2, mat.Ng) <= 0.0f)  // if (backside)
  {
    return;  // absorb
  }

  data.pdf = max(0.0f, dot(data.k2, mat.N) * M_1_PI);

  // For a white Lambert material, the bxdf components match the evaluation pdf. (See MDL_renderer.)
  data.bsdf_diffuse = tint * data.pdf;
}

void brdf_diffuse_sample(INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, float3 tint)
{
  data.k2  = cosineSampleHemisphere(data.xi.x, data.xi.y);
  data.k2  = mat.T * data.k2.x + mat.B * data.k2.y + mat.N * data.k2.z;
  data.k2  = normalize(data.k2);
  data.pdf = dot(data.k2, mat.N) * M_1_PI;

  data.bsdf_over_pdf = tint;  // bsdf * dot(wi, normal) / pdf;
  data.event_type    = (0.0f < dot(data.k2, mat.Ng)) ? BSDF_EVENT_DIFFUSE_REFLECTION : BSDF_EVENT_ABSORB;
}


void brdf_ggx_smith_eval(INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, const int lobe, float3 tint)
{
  // BRDF or BTDF eval?
  // If the incoming light direction is on the backface.
  // Include edge-on (== 0.0f) as "no light" case.
  const bool backside = (dot(data.k2, mat.Ng) <= 0.0f);
  // Nothing to evaluate for given directions?
  if(backside && false)  // && scatter_reflect
  {
    data.pdf         = 0.0f;
    data.bsdf_glossy = float3(0.0f,0.0f,0.0f);
    return;
  }

  const float nk1 = abs(dot(data.k1, mat.N));
  const float nk2 = abs(dot(data.k2, mat.N));

  // compute_half_vector() for scatter_reflect.
  const float3 h = normalize(data.k1 + data.k2);

  // Invalid for reflection / refraction?
  const float nh  = dot(mat.N, h);
  const float k1h = dot(data.k1, h);
  const float k2h = dot(data.k2, h);

  if(nh < 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    data.pdf         = 0.0f;
    data.bsdf_glossy = float3(0.0f,0.0f,0.0f);
    return;
  }

  // Compute BSDF and pdf.
  const float3 h0 = float3(dot(mat.T, h), dot(mat.B, h), nh);

  data.pdf = hvd_ggx_eval(1.0f / mat.roughness, h0);
  float G1;
  float G2;

  float G12;
  G12 = ggx_smith_shadow_mask(G1, G2, float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1),
                              float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), mat.roughness);

  data.pdf *= 0.25f / (nk1 * nh);

  float3 bsdf = make_float3(G12 * data.pdf);

  data.pdf *= G1;

  if(mat.iridescence > 0.0f)
  {
    const float3 factor = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, k1h);

    switch(lobe)
    {
      case LOBE_SPECULAR_REFLECTION:
        tint *= lerp(float3(1.f,1.f,1.f), factor, mat.iridescence);
        break;

      case LOBE_METAL_REFLECTION:
        tint = mix_rgb(tint, mat.specularColor, factor * mat.iridescence);
        break;
    }
  }

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  data.bsdf_glossy = bsdf * tint;
}

void brdf_ggx_smith_sample(INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, const int lobe, float3 tint)
{
  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  data.bsdf_over_pdf = float3(0.0f,0.0f,0.0f);
  data.pdf           = 0.0f;

  const float nk1 = abs(dot(data.k1, mat.N));

  const float3 k10 = float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);

  // Sample half-vector, microfacet normal.
  const float3 h0 = hvd_ggx_sample_vndf(k10, mat.roughness, data.xi.xy);


  if(abs(h0.z) == 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3  h  = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;
  const float kh = dot(data.k1, h);

  if(kh <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // BRDF: reflect
  data.k2            = (2.0f * kh) * h - data.k1;
  data.bsdf_over_pdf = float3(1.0f, 1.0f, 1.0f);  // PERF Always white with the original setup.
  data.event_type    = BSDF_EVENT_GLOSSY_REFLECTION;

  // Check if the resulting direction is on the correct side of the actual geometry
  const float gnk2 = dot(data.k2, mat.Ng);  // * ((data.typeEvent == BSDF_EVENT_GLOSSY_REFLECTION) ? 1.0f : -1.0f);

  if(gnk2 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  const float nk2 = abs(dot(data.k2, mat.N));
  const float k2h = abs(dot(data.k2, h));

  float G1;
  float G2;

  float G12;
  G12 = ggx_smith_shadow_mask(G1, G2, k10, float3(dot(data.k2, mat.T), dot(data.k2, mat.B), nk2), mat.roughness);

  if(G12 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  data.bsdf_over_pdf *= G12 / G1;

  // Compute pdf
  data.pdf = hvd_ggx_eval(1.0f / mat.roughness, h0) * G1;
  data.pdf *= 0.25f / (nk1 * h0.z);

  if(mat.iridescence > 0.0f)
  {
    const float3 factor = thin_film_factor(mat.iridescenceThickness, mat.iridescenceIor, mat.ior2, mat.ior1, kh);

    switch(lobe)
    {
      case LOBE_SPECULAR_REFLECTION:
        tint *= lerp(float3(1.f,1.f,1.f), factor, mat.iridescence);
        break;

      case LOBE_METAL_REFLECTION:
        tint = mix_rgb(tint, mat.specularColor, factor * mat.iridescence);
        break;
    }
  }

  data.bsdf_over_pdf *= tint;
}


void btdf_ggx_smith_eval(INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat, const float3 tint)
{
  bool isThinWalled = (mat.thickness == 0.0f);

  const float2 ior = float2(mat.ior1, mat.ior2);

  const float nk1 = abs(dot(data.k1, mat.N));
  const float nk2 = abs(dot(data.k2, mat.N));

  // BRDF or BTDF eval?
  // If the incoming light direction is on the backface.
  // Do NOT include edge-on (== 0.0f) as backside here to take the reflection path.
  const bool backside = (dot(data.k2, mat.Ng) < 0.0f);

  const float3 h = compute_half_vector(data.k1, data.k2, mat.N, ior, nk2, backside, isThinWalled);

  // Invalid for reflection / refraction?
  const float nh  = dot(mat.N, h);
  const float k1h = dot(data.k1, h);
  const float k2h = dot(data.k2, h) * (backside ? -1.0f : 1.0f);

  if(nh < 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    data.pdf         = 0.0f;  // absorb
    data.bsdf_glossy = float3(0.0f,0.0f,0.0f);
    return;
  }

  float fr;

  if(!backside)
  {
    // For scatter_transmit: Only allow TIR with BRDF eval.
    if(!isTIR(ior, k1h))
    {
      data.pdf         = 0.0f;  // absorb
      data.bsdf_glossy = float3(0.0f,0.0f,0.0f);
      return;
    }
    else
    {
      fr = 1.0f;
    }
  }
  else
  {
    fr = 0.0f;
  }

  // Compute BSDF and pdf
  const float3 h0 = float3(dot(mat.T, h), dot(mat.B, h), nh);

  data.pdf = hvd_ggx_eval(1.0f / mat.roughness, h0);

  float G1;
  float G2;
  float G12;
  G12 = ggx_smith_shadow_mask(G1, G2, float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1),
                              float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), mat.roughness);

  if(!isThinWalled && backside)  // Refraction?
  {
    // Refraction pdf and BTDF
    const float tmp = k1h * ior.x - k2h * ior.y;

    data.pdf *= k1h * k2h / (nk1 * nh * tmp * tmp);
  }
  else
  {
    // Reflection pdf and BRDF (and pseudo-BTDF for thin-walled)
    data.pdf *= 0.25f / (nk1 * nh);
  }

  const float prob = (backside) ? 1.0f - fr : fr;

  const float3 bsdf = make_float3(prob * G12 * data.pdf);

  data.pdf *= prob * G1;

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  data.bsdf_glossy = bsdf * tint;
}

void btdf_ggx_smith_sample(INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat, const float3 tint)
{
  bool isThinWalled = (mat.thickness == 0.0f);

  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  data.bsdf_over_pdf = float3(0.0f,0.0f,0.0f);
  data.pdf           = 0.0f;

  const float2 ior = float2(mat.ior1, mat.ior2);

  const float nk1 = abs(dot(data.k1, mat.N));

  const float3 k10 = float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);

  // Sample half-vector, microfacet normal.
  const float3 h0 = hvd_ggx_sample_vndf(k10, mat.roughness, data.xi.xy);

  if(abs(h0.z) == 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;

  const float kh = dot(data.k1, h);

  if(kh <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Case scatter_transmit
  bool tir = false;
  if(isThinWalled)  // No refraction!
  {
    // pseudo-BTDF: flip a reflected reflection direction to the back side
    data.k2 = (2.0f * kh) * h - data.k1;
    data.k2 = normalize(data.k2 - 2.0f * mat.N * dot(data.k2, mat.N));
  }
  else
  {
    // BTDF: refract
    data.k2 = refract(data.k1, h, ior.x / ior.y, kh, tir);
  }

  data.bsdf_over_pdf = float3(1.0f,1.0f,1.0f);  // Was: (float3(1.0f) - fr) / prob; // PERF Always white with the original setup.
  data.event_type    = (tir) ? BSDF_EVENT_GLOSSY_REFLECTION : BSDF_EVENT_GLOSSY_TRANSMISSION;

  // Check if the resulting direction is on the correct side of the actual geometry
  const float gnk2 = dot(data.k2, mat.Ng) * ((data.event_type == BSDF_EVENT_GLOSSY_REFLECTION) ? 1.0f : -1.0f);

  if(gnk2 <= 0.0f || isnan(data.k2.x))
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }


  const float nk2 = abs(dot(data.k2, mat.N));
  const float k2h = abs(dot(data.k2, h));

  float G1;
  float G2;
  float G12;
  G12 = ggx_smith_shadow_mask(G1, G2, k10, float3(dot(data.k2, mat.T), dot(data.k1, mat.B), nk2), mat.roughness);

  if(G12 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  data.bsdf_over_pdf *= G12 / G1;

  // Compute pdf
  data.pdf = hvd_ggx_eval(1.0f / mat.roughness, h0) * G1;  // * prob;

  if(!isThinWalled && (data.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION))  // if (refraction)
  {
    const float tmp = kh * ior.x - k2h * ior.y;

    data.pdf *= kh * k2h / (nk1 * h0.z * tmp * tmp);
  }
  else
  {
    data.pdf *= 0.25f / (nk1 * h0.z);
  }

  data.bsdf_over_pdf *= tint;
}

void brdf_sheen_eval(INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  // BRDF or BTDF eval?
  // If the incoming light direction is on the backface.
  // Include edge-on (== 0.0f) as "no light" case.
  const bool backside = (dot(data.k2, mat.Ng) <= 0.0f);
  // Nothing to evaluate for given directions?
  if(backside)  // && scatter_reflect
  {
    return;  // absorb
  }

  const float nk1 = abs(dot(data.k1, mat.N));
  const float nk2 = abs(dot(data.k2, mat.N));

  // compute_half_vector() for scatter_reflect.
  const float3 h = normalize(data.k1 + data.k2);

  // Invalid for reflection / refraction?
  const float nh  = dot(mat.N, h);
  const float k1h = dot(data.k1, h);
  const float k2h = dot(data.k2, h);

  if(nh < 0.0f || k1h < 0.0f || k2h < 0.0f)
  {
    return;  // absorb
  }

  const float invRoughness = 1.0f / (mat.sheenRoughness * mat.sheenRoughness);  // Perceptual roughness to alpha G.

  // Compute BSDF and pdf
  const float3 h0 = float3(dot(mat.T, h), dot(mat.B, h), nh);

  data.pdf = hvd_sheen_eval(invRoughness, h0.z);

  float G1;
  float G2;

  const float G12 = vcavities_shadow_mask(G1, G2, h0.z, float3(dot(mat.T, data.k1), dot(mat.B, data.k1), nk1), k1h,
                                          float3(dot(mat.T, data.k2), dot(mat.B, data.k2), nk2), k2h);
  data.pdf *= 0.25f / (nk1 * nh);

  const float3 bsdf = make_float3(G12 * data.pdf);

  data.pdf *= G1;

  // eval output: (glossy part of the) bsdf * dot(k2, normal)
  data.bsdf_glossy = bsdf * mat.sheenColor;
}

void brdf_sheen_sample(INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  // When the sampling returns eventType = BSDF_EVENT_ABSORB, the path ends inside the ray generation program.
  // Make sure the returned values are valid numbers when manipulating the PRD.
  data.bsdf_over_pdf = float3(0.0f,0.0f,0.0f);
  data.pdf           = 0.0f;

  const float invRoughness = 1.0f / (mat.sheenRoughness * mat.sheenRoughness);  // Perceptual roughness to alpha G.

  const float nk1 = abs(dot(data.k1, mat.N));

  const float3 k10 = float3(dot(data.k1, mat.T), dot(data.k1, mat.B), nk1);

  float      xiFlip = data.xi.z;
  const float3 h0     = flip(hvd_sheen_sample(data.xi.xy, invRoughness), k10, xiFlip);

  if(abs(h0.z) == 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // Transform to world
  const float3 h = h0.x * mat.T + h0.y * mat.B + h0.z * mat.N;

  const float k1h = dot(data.k1, h);

  if(k1h <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  // BRDF: reflect
  data.k2            = (2.0f * k1h) * h - data.k1;
  data.bsdf_over_pdf = float3(1.0f,1.0f,1.0f);  // PERF Always white with the original setup.
  data.event_type    = BSDF_EVENT_GLOSSY_REFLECTION;

  // Check if the resulting reflection direction is on the correct side of the actual geometry.
  const float gnk2 = dot(data.k2, mat.Ng);

  if(gnk2 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  const float nk2 = abs(dot(data.k2, mat.N));
  const float k2h = abs(dot(data.k2, h));

  float G1;
  float G2;

  const float G12 = vcavities_shadow_mask(G1, G2, h0.z, k10, k1h, float3(dot(data.k2, mat.T), dot(data.k1, mat.B), nk2), k2h);
  if(G12 <= 0.0f)
  {
    data.event_type = BSDF_EVENT_ABSORB;
    return;
  }

  data.bsdf_over_pdf *= G12 / G1;

  // Compute pdf.
  data.pdf = hvd_sheen_eval(invRoughness, h0.z) * G1;

  data.pdf *= 0.25f / (nk1 * h0.z);

  data.bsdf_over_pdf *= mat.sheenColor;
}


/** @DOC_START
# Function bsdfEvaluate
>  Evaluate the BSDF for the given material.
@DOC_END */
void bsdfEvaluate(INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  float3  tint        = mat.baseColor;
  float VdotN       = dot(data.k1, mat.N);
  int   lobe        = findLobe(mat, VdotN, data.xi.z, tint);
  data.bsdf_diffuse = float3(0, 0, 0);
  data.bsdf_glossy  = float3(0, 0, 0);
  data.pdf          = 0.0;

  if(lobe == LOBE_DIFFUSE_REFLECTION)
  {
    brdf_diffuse_eval(data, mat, tint);
  }
  else if(lobe == LOBE_SPECULAR_REFLECTION)
  {
    brdf_ggx_smith_eval(data, mat, LOBE_SPECULAR_REFLECTION, mat.specularColor);
  }
  else if(lobe == LOBE_SPECULAR_TRANSMISSION)
  {
    btdf_ggx_smith_eval(data, mat, tint);
  }
  else if(lobe == LOBE_METAL_REFLECTION)
  {
    brdf_ggx_smith_eval(data, mat, LOBE_METAL_REFLECTION, mat.baseColor);
  }
  else if(lobe == LOBE_CLEARCOAT_REFLECTION)
  {
    float r2        = mat.clearcoatRoughness * mat.clearcoatRoughness;
    mat.roughness   = float2(r2, r2);
    mat.N           = mat.Nc;
    mat.iridescence = 0.0f;
    brdf_ggx_smith_eval(data, mat, LOBE_CLEARCOAT_REFLECTION, float3(1, 1, 1));
  }
  else if(lobe == LOBE_SHEEN_REFLECTION)
  {
    brdf_sheen_eval(data, mat);
  }
}

/** @DOC_START
# Function bsdfSample
>  Sample the BSDF for the given material
@DOC_END */
void bsdfSample(INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  float3  tint         = mat.baseColor;
  float VdotN        = dot(data.k1, mat.N);
  int   lobe         = findLobe(mat, VdotN, data.xi.z, tint);
  data.pdf           = 0;
  data.bsdf_over_pdf = float3(0.0F,0.0F,0.0F);
  data.event_type    = BSDF_EVENT_ABSORB;

  if(lobe == LOBE_DIFFUSE_REFLECTION)
  {
    brdf_diffuse_sample(data, mat, tint);
  }
  else if(lobe == LOBE_SPECULAR_REFLECTION)
  {
    brdf_ggx_smith_sample(data, mat, LOBE_SPECULAR_REFLECTION, mat.specularColor);
    data.event_type = BSDF_EVENT_SPECULAR;
  }
  else if(lobe == LOBE_SPECULAR_TRANSMISSION)
  {
    btdf_ggx_smith_sample(data, mat, tint);
  }
  else if(lobe == LOBE_METAL_REFLECTION)
  {
    brdf_ggx_smith_sample(data, mat, LOBE_METAL_REFLECTION, mat.baseColor);
  }
  else if(lobe == LOBE_CLEARCOAT_REFLECTION)
  {
    float r2        = mat.clearcoatRoughness * mat.clearcoatRoughness;
    mat.roughness   = float2(r2, r2);
    mat.N           = mat.Nc;
    mat.B           = normalize(cross(mat.N, mat.T));  // Assumes Nc and Tc are not collinear!
    mat.T           = cross(mat.B, mat.N);
    mat.iridescence = 0.0f;
    brdf_ggx_smith_sample(data, mat, LOBE_CLEARCOAT_REFLECTION, float3(1, 1, 1));
  }
  else if(lobe == LOBE_SHEEN_REFLECTION)
  {
    // Sheen is using the state.sheenColor and state.sheenInvRoughness values directly.
    // Only brdf_sheen_sample needs a third random sample for the v-cavities flip. Put this as argument.
    brdf_sheen_sample(data, mat);
  }

  // Avoid internal reflection
  if(data.pdf <= 0.00001F || any(isnan(data.bsdf_over_pdf)))
    data.event_type = BSDF_EVENT_ABSORB;
}

//--------------------------------------------------------------------------------------------------
// Those functions are used to evaluate and sample the BSDF for a simple PBR material.
// without any additional lobes like clearcoat, sheen, etc. and without the need of random numbers.
// In the simple sampling, there is no diffuse sampling, and for pure reflection use xi == float2(0,0)
//--------------------------------------------------------------------------------------------------
void bsdfEvaluateSimple(INOUT_TYPE(BsdfEvaluateData) data, PbrMaterial mat)
{
  // Specular reflection
  float3  H              = normalize(data.k1 + data.k2);
  float alphaRoughness = mat.roughness.x;
  float NdotV          = clampedDot(mat.N, data.k1);
  float NdotL          = clampedDot(mat.N, data.k2);
  float VdotH          = clampedDot(data.k1, H);
  float NdotH          = clampedDot(mat.N, H);
  float LdotH          = clampedDot(data.k2, H);

  float3  c_min_reflectance = float3(0.04F,0.04F,0.04F);
  float3  f0                = lerp(c_min_reflectance, mat.baseColor, mat.metallic);
  float3  f90               = float3(1.0F,1.0F,1.0F);
  float3  f                 = schlickFresnel(f0, f90, VdotH);
  float vis               = ggxSmithVisibility(NdotL, NdotV, alphaRoughness);  // Vis = G / (4 * NdotL * NdotV)
  float d                 = ggxDistribution(NdotH, alphaRoughness);

  data.bsdf_glossy  = mat.metallic * (f * vis * d) * NdotL;                      // GGX-Smith
  data.bsdf_diffuse = (1.0F - mat.metallic) * (mat.baseColor * M_1_PI) * NdotL;  // Lambertian
  float diffusePdf  = M_1_PI * NdotL;
  float specularPdf = d * NdotH / (4.0F * LdotH);
  data.pdf          = lerp(diffusePdf, specularPdf, mat.metallic);
}

void bsdfSampleSimple(INOUT_TYPE(BsdfSampleData) data, PbrMaterial mat)
{
  float3 tint          = mat.baseColor;
  data.bsdf_over_pdf = float3(0.0F,0.0F,0.0F);
  data.pdf           = 0.0;
  data.event_type    = BSDF_EVENT_GLOSSY_REFLECTION;

  float3 halfVector      = ggxSampling(max(mat.roughness.x, mat.roughness.y), data.xi.xy);  // Glossy
  halfVector           = mat.T * halfVector.x + mat.B * halfVector.y + mat.N * halfVector.z;
  float3 sampleDirection = reflect(-data.k1, halfVector);

  data.k2 = sampleDirection;
  BsdfEvaluateData evalData;
  evalData.k1 = data.k1;
  evalData.k2 = data.k2;
  bsdfEvaluateSimple(evalData, mat);
  data.pdf           = evalData.pdf;
  data.bsdf_over_pdf = (evalData.bsdf_glossy) / data.pdf;  // Because we don't care about diffuse reflection

  if(data.pdf <= 0.00001F || any(isnan(data.bsdf_over_pdf)))
    data.event_type = BSDF_EVENT_ABSORB;
}

float3 ggxEvaluate(float3 V, float3 L, PbrMaterial mat)
{
  BsdfEvaluateData data;
  data.k1 = V;
  data.k2 = L;

  bsdfEvaluateSimple(data, mat);

  return data.bsdf_glossy + data.bsdf_diffuse;
}

#endif
