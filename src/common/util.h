#pragma once

#include <cuda_runtime.h>

namespace lfm {
__host__ __device__ __forceinline__ int TileIjkToIdx(int3 _dim, int3 _ijk)
{
    return _ijk.x * _dim.y * _dim.z + _ijk.y * _dim.z + _ijk.z;
}
__host__ __device__ __forceinline__ int3 TileIdxToIjk(int3 _dim, int _idx)
{
    return { _idx / (_dim.y * _dim.z), (_idx / _dim.z) % _dim.y, _idx % _dim.z };
}

__host__ __device__ __forceinline__ int VoxelIjkToIdx(int3 _ijk)
{
    return _ijk.x * 64 + _ijk.y * 8 + _ijk.z;
}
__host__ __device__ __forceinline__ int3 VoxelIdxToIjk(int _idx)
{
    return { _idx / 64, (_idx / 8) % 8, _idx % 8 };
}

__host__ __device__ __forceinline__ int3 IjkToTileIjk(int3 _ijk)
{
    return { _ijk.x >> 3, _ijk.y >> 3, _ijk.z >> 3 };
}

__host__ __device__ __forceinline__ int IjkToIdx(int3 _tile_dim, int3 _ijk)
{
    int3 tile_ijk  = { _ijk.x >> 3, _ijk.y >> 3, _ijk.z >> 3 };
    int3 voxel_ijk = { _ijk.x - tile_ijk.x * 8, _ijk.y - tile_ijk.y * 8, _ijk.z - tile_ijk.z * 8 };
    int tile_idx   = tile_ijk.x * _tile_dim.y * _tile_dim.z + tile_ijk.y * _tile_dim.z + tile_ijk.z;
    int voxel_idx  = voxel_ijk.x * 64 + voxel_ijk.y * 8 + voxel_ijk.z;
    return tile_idx * 512 + voxel_idx;
}

__host__ __device__ __forceinline__ int Prod(int3 _v)
{
    return _v.x * _v.y * _v.z;
}

__host__ __device__ __forceinline__ float Clamp(float _v, float _min, float _max)
{
    float _ret = _v;
    if (_v < _min)
        _ret = _min;
    if (_v > _max)
        _ret = _max;
    return _ret;
}

__host__ __device__ __forceinline__ float3 Clamp(float3 _v, float3 _min, float3 _max)
{
    return { Clamp(_v.x, _min.x, _max.x), Clamp(_v.y, _min.y, _max.y), Clamp(_v.z, _min.z, _max.z) };
}

__host__ __device__ __forceinline__ float Abs(float _v)
{
    return _v >= 0.0f ? _v : -_v;
}

__host__ __device__ __forceinline__ bool InRange(int3 _coord, int3 _min, int3 _max)
{
    return _coord.x >= _min.x && _coord.x <= _max.x && _coord.y >= _min.y && _coord.y <= _max.y && _coord.z >= _min.z && _coord.z <= _max.z;
}

__device__ __forceinline__ float Interp(float3 _ijk, const float* _data, int3 _tile_dim, int3 _max_ijk)
{
    float eps            = 0.0001f;
    float3 float_min_ijk = { 0.0f, 0.0f, 0.0f };
    float3 float_max_ijk = { float(_max_ijk.x) - eps, float(_max_ijk.y) - eps, float(_max_ijk.z) - eps };
    float3 clamped_ijk   = Clamp(_ijk, float_min_ijk, float_max_ijk);
    int3 base_ijk        = { int(clamped_ijk.x), int(clamped_ijk.y), int(clamped_ijk.z) };

    float ret = 0.0f;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++) {
                int3 target_ijk = { base_ijk.x + i, base_ijk.y + j, base_ijk.z + k };
                int target_idx  = IjkToIdx(_tile_dim, target_ijk);
                float weight    = (1.0f - Abs(clamped_ijk.x - float(target_ijk.x)));
                weight *= 1.0f - Abs(clamped_ijk.y - float(target_ijk.y));
                weight *= 1.0f - Abs(clamped_ijk.z - float(target_ijk.z));
                ret += _data[target_idx] * weight;
            }

    return ret;
}

template <typename T>
void DevToDevCpyAsync(T* _dst, const T* _src, int _size, cudaStream_t _stream);

struct float3x3 {
    float xx, xy, xz;
    float yx, yy, yz;
    float zx, zy, zz;
};

__host__ __device__ __forceinline__ float3x3 MatTranspose(float3x3 _m)
{
    float3x3 ret;
    ret.xx = _m.xx;
    ret.xy = _m.yx;
    ret.xz = _m.zx;
    ret.yx = _m.xy;
    ret.yy = _m.yy;
    ret.yz = _m.zy;
    ret.zx = _m.xz;
    ret.zy = _m.yz;
    ret.zz = _m.zz;
    return ret;
}

__host__ __device__ __forceinline__ float3 MatMulVec(float3x3 _m, float3 _v)
{
    float3 ret;
    ret.x = _m.xx * _v.x + _m.xy * _v.y + _m.xz * _v.z;
    ret.y = _m.yx * _v.x + _m.yy * _v.y + _m.yz * _v.z;
    ret.z = _m.zx * _v.x + _m.zy * _v.y + _m.zz * _v.z;
    return ret;
}

__host__ __device__ __forceinline__ float3x3 MatMulMat(float3x3 _m1, float3x3 _m2)
{
    float3x3 ret;
    ret.xx = _m1.xx * _m2.xx + _m1.xy * _m2.yx + _m1.xz * _m2.zx;
    ret.xy = _m1.xx * _m2.xy + _m1.xy * _m2.yy + _m1.xz * _m2.zy;
    ret.xz = _m1.xx * _m2.xz + _m1.xy * _m2.yz + _m1.xz * _m2.zz;
    ret.yx = _m1.yx * _m2.xx + _m1.yy * _m2.yx + _m1.yz * _m2.zx;
    ret.yy = _m1.yx * _m2.xy + _m1.yy * _m2.yy + _m1.yz * _m2.zy;
    ret.yz = _m1.yx * _m2.xz + _m1.yy * _m2.yz + _m1.yz * _m2.zz;
    ret.zx = _m1.zx * _m2.xx + _m1.zy * _m2.yx + _m1.zz * _m2.zx;
    ret.zy = _m1.zx * _m2.xy + _m1.zy * _m2.yy + _m1.zz * _m2.zy;
    ret.zz = _m1.zx * _m2.xz + _m1.zy * _m2.yz + _m1.zz * _m2.zz;
    return ret;
}

__host__ __device__ __forceinline__ float4 QuatMulQuat(float4 _q1, float4 _q2)
{
    float4 ret;
    ret.w = _q1.w * _q2.w - _q1.x * _q2.x - _q1.y * _q2.y - _q1.z * _q2.z;
    ret.x = _q1.w * _q2.x + _q1.x * _q2.w + _q1.y * _q2.z - _q1.z * _q2.y;
    ret.y = _q1.w * _q2.y - _q1.x * _q2.z + _q1.y * _q2.w + _q1.z * _q2.x;
    ret.z = _q1.w * _q2.z + _q1.x * _q2.y - _q1.y * _q2.x + _q1.z * _q2.w;
    return ret;
}

__host__ __device__ __forceinline__ float4 QuatConj(float4 _q)
{
    float4 ret;
    ret.w = _q.w;
    ret.x = -_q.x;
    ret.y = -_q.y;
    ret.z = -_q.z;
    return ret;
}

__host__ __device__ __forceinline__ float3x3 QuatToRot(float4 _q)
{
    float3x3 ret;
    ret.xx = 1.0f - 2.0f * (_q.y * _q.y + _q.z * _q.z);
    ret.xy = 2.0f * (_q.x * _q.y - _q.z * _q.w);
    ret.xz = 2.0f * (_q.x * _q.z + _q.y * _q.w);
    ret.yx = 2.0f * (_q.x * _q.y + _q.z * _q.w);
    ret.yy = 1.0f - 2.0f * (_q.x * _q.x + _q.z * _q.z);
    ret.yz = 2.0f * (_q.y * _q.z - _q.x * _q.w);
    ret.zx = 2.0f * (_q.x * _q.z - _q.y * _q.w);
    ret.zy = 2.0f * (_q.y * _q.z + _q.x * _q.w);
    ret.zz = 1.0f - 2.0f * (_q.x * _q.x + _q.y * _q.y);
    return ret;
}

__host__ __device__ __forceinline__ float3 Cross(float3 _v1, float3 _v2)
{
    float3 ret;
    ret.x = _v1.y * _v2.z - _v1.z * _v2.y;
    ret.y = _v1.z * _v2.x - _v1.x * _v2.z;
    ret.z = _v1.x * _v2.y - _v1.y * _v2.x;
    return ret;
}
};