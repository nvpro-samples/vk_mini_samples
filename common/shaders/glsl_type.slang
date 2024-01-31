//
// From the GLSL spec, section 4.1. 'asic Types'
// Took from source/slang/glsl.meta.slang
//

public typealias vec2 = vector<float, 2>;
public typealias vec3 = vector<float, 3>;
public typealias vec4 = vector<float, 4>;

public typealias dvec2 = vector<double, 2>;
public typealias dvec3 = vector<double, 3>;
public typealias dvec4 = vector<double, 4>;

public typealias bvec2 = vector<bool, 2>;
public typealias bvec3 = vector<bool, 3>;
public typealias bvec4 = vector<bool, 4>;

public typealias ivec2 = vector<int, 2>;
public typealias ivec3 = vector<int, 3>;
public typealias ivec4 = vector<int, 4>;

public typealias uvec2 = vector<uint, 2>;
public typealias uvec3 = vector<uint, 3>;
public typealias uvec4 = vector<uint, 4>;

public typealias i8vec2 = vector<int8_t, 2>;
public typealias i8vec3 = vector<int8_t, 3>;
public typealias i8vec4 = vector<int8_t, 4>;

public typealias u8vec2 = vector<uint8_t, 2>;
public typealias u8vec3 = vector<uint8_t, 3>;
public typealias u8vec4 = vector<uint8_t, 4>;

public typealias i16vec2 = vector<int16_t, 2>;
public typealias i16vec3 = vector<int16_t, 3>;
public typealias i16vec4 = vector<int16_t, 4>;

public typealias u16vec2 = vector<uint16_t, 2>;
public typealias u16vec3 = vector<uint16_t, 3>;
public typealias u16vec4 = vector<uint16_t, 4>;

public typealias i64vec2 = vector<int64_t, 2>;
public typealias i64vec3 = vector<int64_t, 3>;
public typealias i64vec4 = vector<int64_t, 4>;

public typealias u64vec2 = vector<uint64_t, 2>;
public typealias u64vec3 = vector<uint64_t, 3>;
public typealias u64vec4 = vector<uint64_t, 4>;

public typealias mat2 = matrix<float, 2, 2>;
public typealias mat3 = matrix<float, 3, 3>;
public typealias mat4 = matrix<float, 4, 4>;

public typealias mat2x2 = matrix<float, 2, 2>;
public typealias mat2x3 = matrix<float, 3, 2>;
public typealias mat2x4 = matrix<float, 4, 2>;

public typealias mat3x2 = matrix<float, 2, 3>;
public typealias mat3x3 = matrix<float, 3, 3>;
public typealias mat3x4 = matrix<float, 4, 3>;

public typealias mat4x2 = matrix<float, 2, 4>;
public typealias mat4x3 = matrix<float, 3, 4>;
public typealias mat4x4 = matrix<float, 4, 4>;

public typealias dmat2 = matrix<double, 2, 2>;
public typealias dmat3 = matrix<double, 3, 3>;
public typealias dmat4 = matrix<double, 4, 4>;

public typealias dmat2x2 = matrix<double, 2, 2>;
public typealias dmat2x3 = matrix<double, 3, 2>;
public typealias dmat2x4 = matrix<double, 4, 2>;

public typealias dmat3x2 = matrix<double, 2, 3>;
public typealias dmat3x3 = matrix<double, 3, 3>;
public typealias dmat3x4 = matrix<double, 4, 3>;

public typealias dmat4x2 = matrix<double, 2, 4>;
public typealias dmat4x3 = matrix<double, 3, 4>;
public typealias dmat4x4 = matrix<double, 4, 4>;

public typealias usampler1D = Sampler1D<uint4>;
public typealias isampler1D = Sampler1D<int4>;
public typealias sampler1D = Sampler1D<float4>;

public typealias usampler2D = Sampler2D<uint4>;
public typealias isampler2D = Sampler2D<int4>;
public typealias sampler2D = Sampler2D<float4>;

public typealias usampler3D = Sampler3D<uint4>;
public typealias isampler3D = Sampler3D<int4>;
public typealias sampler3D = Sampler3D<float4>;

public typealias usamplerCube = SamplerCube<uint4>;
public typealias isamplerCube = SamplerCube<int4>;
public typealias samplerCube = SamplerCube<float4>;

public typealias Sampler1DShadow<T=float> = __TextureImpl<T, __Shape1D, /*isArray:*/ 0, /*isMS:*/ 0, /*sampleCount:*/ 0, /*access:*/ 0, /*isShadow: */ 1, /*isCombined: */ 1, /*format*/ 0>;
public typealias usampler1DShadow = Sampler1DShadow<uint>;
public typealias isampler1DShadow = Sampler1DShadow<int>;
public typealias sampler1DShadow = Sampler1DShadow<float>;

public typealias Sampler2DShadow<T=float> = __TextureImpl<T, __Shape2D, /*isArray:*/ 0, /*isMS:*/ 0, /*sampleCount:*/ 0, /*access:*/ 0, /*isShadow: */ 1, /*isCombined: */ 1, /*format*/ 0>;
public typealias usampler2DShadow = Sampler2DShadow<uint>;
public typealias isampler2DShadow = Sampler2DShadow<int>;
public typealias sampler2DShadow = Sampler2DShadow<float>;

public typealias SamplerCubeShadow<T=float> = __TextureImpl<T, __ShapeCube, /*isArray:*/ 0, /*isMS:*/ 0, /*sampleCount:*/ 0, /*access:*/ 0, /*isShadow: */ 1, /*isCombined: */ 1, /*format*/ 0>;
public typealias usamplerCubeShadow = SamplerCubeShadow<uint>;
public typealias isamplerCubeShadow = SamplerCubeShadow<int>;
public typealias samplerCubeShadow = SamplerCubeShadow<float>;

public typealias usampler1DArray = Sampler1DArray<uint4>;
public typealias isampler1DArray = Sampler1DArray<int4>;
public typealias sampler1DArray = Sampler1DArray<float4>;

public typealias usampler2DArray = Sampler2DArray<uint4>;
public typealias isampler2DArray = Sampler2DArray<int4>;
public typealias sampler2DArray = Sampler2DArray<float4>;

public typealias usamplerCubeArray = SamplerCubeArray<uint4>;
public typealias isamplerCubeArray = SamplerCubeArray<int4>;
public typealias samplerCubeArray = SamplerCubeArray<float4>;

public typealias Sampler1DArrayShadow<T=float> = __TextureImpl<T, __Shape1D, /*isArray:*/ 1, /*isMS:*/ 0, /*sampleCount:*/ 0, /*access:*/ 0, /*isShadow: */ 1, /*isCombined: */ 1, /*format*/ 0>;
public typealias usampler1DArrayShadow = Sampler1DArrayShadow<uint>;
public typealias isampler1DArrayShadow = Sampler1DArrayShadow<int>;
public typealias sampler1DArrayShadow = Sampler1DArrayShadow<float>;

public typealias Sampler2DArrayShadow<T=float> = __TextureImpl<T, __Shape2D, /*isArray:*/ 1, /*isMS:*/ 0, /*sampleCount:*/ 0, /*access:*/ 0, /*isShadow: */ 1, /*isCombined: */ 1, /*format*/ 0>;
public typealias usampler2DArrayShadow = Sampler2DArrayShadow<uint>;
public typealias isampler2DArrayShadow = Sampler2DArrayShadow<int>;
public typealias sampler2DArrayShadow = Sampler2DArrayShadow<float>;

public typealias SamplerCubeArrayShadow<T=float> = __TextureImpl<T, __ShapeCube, /*isArray:*/ 1, /*isMS:*/ 0, /*sampleCount:*/ 0, /*access:*/ 0, /*isShadow: */ 1, /*isCombined: */ 1, /*format*/ 0>;
public typealias usamplerCubeArrayShadow = SamplerCubeArrayShadow<uint>;
public typealias isamplerCubeArrayShadow = SamplerCubeArrayShadow<int>;
public typealias samplerCubeArrayShadow = SamplerCubeArrayShadow<float>;

static /*out*/ float4 gl_Position : SV_Position;
static /*out*/ float gl_PointSize : SV_PointSize;
static /*out*/ vec4 gl_FragCoord : SV_Position;
static /*out*/ float gl_FragDepth : SV_Depth;
static /*out*/ int gl_FragStencilRef : SV_StencilRef;

static uvec3 gl_NumWorkGroups;
static /*public out*/ uvec3 gl_GlobalInvocationID : SV_DispatchThreadID;
static /*public out*/ uvec3 gl_WorkGroupID : SV_GroupID;
static /*public out*/ uvec3 gl_LocalInvocationIndex : SV_GroupIndex;
static /*public out*/ uvec3 gl_LocalInvocationID : SV_GroupThreadID;    


// TODO: define overload for tessellation control stage.
static /*in*/ int gl_InvocationID : SV_GSInstanceID;

static /*in*/ int gl_InstanceIndex : SV_InstanceID;
static /*in*/ bool gl_FrontFacing : SV_IsFrontFace;

// TODO: define overload for geometry stage.
static /*in*/ int gl_Layer : SV_RenderTargetArrayIndex;

static /*in*/ int gl_SampleID : SV_SampleIndex;
static /*in*/ int gl_VertexIndex : SV_VertexID;
static /*in*/ int gl_ViewIndex : SV_ViewID;
static /*in*/ int gl_ViewportIndex : SV_ViewportArrayIndex;

// TODO KHR_shader_subgroup
static /*in*/ uint gl_NumSubgroups;
static /*in*/ uint gl_SubgroupID;
static /*in*/ uint gl_SubgroupSize;
static /*in*/ uint gl_SubgroupInvocationID;
