#include "cooperative_groups.h"
#include "timer.h"
#include "trim_poisson.h"
#include <cub/cub.cuh>

namespace lfm {
TrimPoisson::TrimPoisson(int3 _tile_dim)
{
    Alloc(_tile_dim);
}

void TrimPoisson::Alloc(int3 _tile_dim)
{
    tile_dim_ = _tile_dim;

    int voxel_num = Prod(_tile_dim) * 512;
    x_            = std::make_shared<DHMemory<float>>(voxel_num);
    is_dof_       = std::make_shared<DHMemory<char>>(voxel_num);
    a_diag_       = std::make_shared<DHMemory<float>>(voxel_num);
    a_x_          = std::make_shared<DHMemory<float>>(voxel_num);
    a_y_          = std::make_shared<DHMemory<float>>(voxel_num);
    a_z_          = std::make_shared<DHMemory<float>>(voxel_num);
    b_            = std::make_shared<DHMemory<float>>(voxel_num);
    buffer_       = std::make_shared<DHMemory<float>>(voxel_num);

    int tile_num  = Prod(_tile_dim);
    tile_trimmed_ = std::make_shared<DHMemory<char>>(tile_num);
    cudaMemset((void*)(tile_trimmed_->dev_ptr_), 0x00, tile_num);
}

__global__ void TrimEmptyKernel(char* _tile_trimmed, int3 _tile_dim, const char* _is_dof)
{
    int tile_idx = blockIdx.x;
    int t_id     = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int idx = tile_idx * 512 + t_id + i * 128;
        if (_is_dof[idx]) {
            _tile_trimmed[tile_idx] = 1;
            return;
        }
    }
}
__global__ void TrimTrivialKernel(char* _tile_trimmed, int3 _tile_dim, const char* _is_dof, const float* _a_diag, const float* _a_x, const float* _a_y, const float* _a_z, float _default_a_diag, float _default_a_off_diag)
{
    int tile_idx = blockIdx.x;
    int t_id     = threadIdx.x;
    if (_tile_trimmed[tile_idx] == 0)
        return;
    // check current tile
    for (int i = 0; i < 4; i++) {
        int idx = tile_idx * 512 + t_id + i * 128;
        if (_is_dof[idx] == 0) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_diag[idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[idx] != _default_a_off_diag || _a_y[idx] != _default_a_off_diag || _a_z[idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
    }
    // check boundary tile
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    if (tile_ijk.x == 0 || tile_ijk.y == 0 || tile_ijk.z == 0) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    if (tile_ijk.x == _tile_dim.x - 1 || tile_ijk.y == _tile_dim.y - 1 || tile_ijk.z == _tile_dim.z - 1) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    // check nb face
    // x-
    int3 nb_tile_ijk  = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
    int3 nb_voxel_ijk = { 6 + t_id / 64, (t_id / 8) % 8, t_id % 8 };
    int nb_tile_idx   = TileIjkToIdx(_tile_dim, nb_tile_ijk);
    int nb_voxel_idx  = VoxelIjkToIdx(nb_voxel_ijk);
    int nb_idx        = nb_tile_idx * 512 + nb_voxel_idx;
    if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    // x+
    nb_tile_ijk  = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
    nb_voxel_ijk = { t_id / 64, (t_id / 8) % 8, t_id % 8 };
    nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
    nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
    nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
    if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    // y-
    nb_tile_ijk  = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
    nb_voxel_ijk = { t_id / 16, 6 + (t_id / 8) % 2, t_id % 8 };
    nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
    nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
    nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
    if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    // y+
    nb_tile_ijk  = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
    nb_voxel_ijk = { t_id / 16, (t_id / 8) % 2, t_id % 8 };
    nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
    nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
    nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
    if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    // z-
    nb_tile_ijk  = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
    nb_voxel_ijk = { t_id / 16, (t_id / 2) % 8, 6 + t_id % 2 };
    nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
    nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
    nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
    if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    // z+
    nb_tile_ijk  = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
    nb_voxel_ijk = { t_id / 16, (t_id / 2) % 8, t_id % 2 };
    nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
    nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
    nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
    if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
        _tile_trimmed[tile_idx] = 2;
        return;
    }
    // check nb edge
    int warp_id = t_id / 32;
    int lane_id = t_id % 32;
    if (warp_id == 0 && lane_id < 8) {
        // x- y-
        nb_tile_ijk  = { tile_ijk.x - 1, tile_ijk.y - 1, tile_ijk.z };
        nb_voxel_ijk = { 7, 7, lane_id };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        // x- y+
        nb_tile_ijk  = { tile_ijk.x - 1, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk = { 7, 0, lane_id };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        // x- z-
        nb_tile_ijk  = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z - 1 };
        nb_voxel_ijk = { 7, lane_id, 7 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
    }
    if (warp_id == 1 && lane_id < 8) {
        // y- z+
        nb_tile_ijk  = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z + 1 };
        nb_voxel_ijk = { lane_id, 7, 0 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        // y+ z+
        nb_tile_ijk  = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z + 1 };
        nb_voxel_ijk = { lane_id, 0, 0 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        // x- z+
        nb_tile_ijk  = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk = { 7, lane_id, 0 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
    }
    if (warp_id == 2 && lane_id < 8) {
        // x+ y-
        nb_tile_ijk  = { tile_ijk.x + 1, tile_ijk.y - 1, tile_ijk.z };
        nb_voxel_ijk = { 0, 7, lane_id };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        // x+ y+
        nb_tile_ijk  = { tile_ijk.x + 1, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk = { 0, 0, lane_id };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        // x+ z+
        nb_tile_ijk  = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk = { 0, lane_id, 0 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
    }
    if (warp_id == 3 && lane_id < 8) {
        // y- z-
        nb_tile_ijk  = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z - 1 };
        nb_voxel_ijk = { lane_id, 7, 7 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        // y+ z-
        nb_tile_ijk  = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z - 1 };
        nb_voxel_ijk = { lane_id, 0, 7 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        // x+ z-
        nb_tile_ijk  = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z - 1 };
        nb_voxel_ijk = { 0, lane_id, 7 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if (_is_dof[nb_idx] == 0 || _a_diag[nb_idx] != _default_a_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
        if (_a_x[nb_idx] != _default_a_off_diag || _a_y[nb_idx] != _default_a_off_diag || _a_z[nb_idx] != _default_a_off_diag) {
            _tile_trimmed[tile_idx] = 2;
            return;
        }
    }
}

void TrimPoisson::TrimAsync(float _default_a_diag, float _default_a_off_diag, cudaStream_t _stream)
{
    default_a_diag_     = _default_a_diag;
    default_a_off_diag_ = _default_a_off_diag;
    int tile_num        = Prod(tile_dim_);
    char* tile_trimmed  = tile_trimmed_->dev_ptr_;
    cudaMemsetAsync((void*)tile_trimmed, 0x00, tile_num, _stream);

    const char* is_dof  = is_dof_->dev_ptr_;
    const float* a_diag = a_diag_->dev_ptr_;
    const float* a_x    = a_x_->dev_ptr_;
    const float* a_y    = a_y_->dev_ptr_;
    const float* a_z    = a_z_->dev_ptr_;
    TrimEmptyKernel<<<tile_num, 128, 0, _stream>>>(tile_trimmed, tile_dim_, is_dof);
    TrimTrivialKernel<<<tile_num, 128, 0, _stream>>>(tile_trimmed, tile_dim_, is_dof, a_diag, a_x, a_y, a_z, default_a_diag_, default_a_off_diag_);
}

__global__ void LaplacianDotTrivialKernel(float* _result, int3 _tile_dim, float* _dot_buffer, const float* _x, const char* _tile_trimmed, float _default_a_diag, float _default_a_off_diag)
{
    __shared__ float shared_x[10][10][10];
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id      = threadIdx.x;

    char tile_trimmed = _tile_trimmed[tile_idx];
    if (tile_trimmed != 1)
        return;
    for (int i = 0; i < 4; i++) {
        int voxel_idx                                               = i * 128 + t_id;
        int3 voxel_ijk                                              = VoxelIdxToIjk(voxel_idx);
        int idx                                                     = tile_idx * 512 + voxel_idx;
        shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _x[idx];
    }
    int warp_id = t_id / 32;

    // load nb tile data
    if (warp_id < 2) {
        int a                                               = t_id / 8;
        int b                                               = t_id % 8;
        // x-
        int3 nb_tile_ijk                                    = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
        int3 nb_voxel_ijk                                   = { 7, a, b };
        int nb_tile_idx                                     = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx                                    = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx                                          = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[nb_idx];

        // y-
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
        nb_voxel_ijk                                        = { a, 7, b };
        nb_tile_idx                                         = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx                                        = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx                                              = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = _x[nb_idx];

        // z-
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
        nb_voxel_ijk                                        = { a, b, 7 };
        nb_tile_idx                                         = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx                                        = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx                                              = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = _x[nb_idx];
    } else {
        int group_id                                        = t_id - 64;
        int a                                               = group_id / 8;
        int b                                               = group_id % 8;
        // x+
        int3 nb_tile_ijk                                    = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
        int3 nb_voxel_ijk                                   = { 0, a, b };
        int nb_tile_idx                                     = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx                                    = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx                                          = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[nb_idx];
        // y+
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk                                        = { a, 0, b };
        nb_tile_idx                                         = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx                                        = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx                                              = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = _x[nb_idx];
        // z+
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk                                        = { a, b, 0 };
        nb_tile_idx                                         = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx                                        = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx                                              = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = _x[nb_idx];
    }
    __syncthreads();
    float mul[4];
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int idx        = tile_idx * 512 + voxel_idx;
        float Ax       = _default_a_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1];
        float nb_sum   = shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1] + shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
        nb_sum += shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1] + shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
        nb_sum += shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z] + shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];
        Ax += _default_a_off_diag * nb_sum;
        _result[idx] = Ax;
        mul[i]       = Ax * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1];
    }
    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float sum = BlockReduce(temp_storage).Sum(mul);
    if (t_id == 0)
        _dot_buffer[tile_idx] = sum;
}

__global__ void LaplacianDotNonTrivialKernel(float* _result, int3 _tile_dim, float* _dot_buffer, const float* _x, const char* _tile_trimmed, const char* _is_dof, const float* _a_diag, const float* _a_x, const float* _a_y, const float* _a_z)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id      = threadIdx.x;

    char tile_trimmed = _tile_trimmed[tile_idx];
    if (tile_trimmed != 2)
        return;
    float mul[4];
    for (int i = 0; i < 4; i++) {
        float Ax       = 0.0f;
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int idx        = tile_idx * 512 + voxel_idx;
        if (_is_dof[idx]) {
            mul[i] = _x[idx];
            Ax += _a_diag[idx] * mul[i];
            // x+
            int3 nb_tile_ijk  = tile_ijk;
            int3 nb_voxel_ijk = { voxel_ijk.x + 1, voxel_ijk.y, voxel_ijk.z };
            if (nb_voxel_ijk.x == 8) {
                nb_tile_ijk.x++;
                nb_voxel_ijk.x = 0;
            }
            if (nb_tile_ijk.x < _tile_dim.x) {
                int nb_idx = TileIjkToIdx(_tile_dim, nb_tile_ijk) * 512 + VoxelIjkToIdx(nb_voxel_ijk);
                if (_is_dof[nb_idx])
                    Ax += _a_x[idx] * _x[nb_idx];
            }
            // y+
            nb_tile_ijk  = tile_ijk;
            nb_voxel_ijk = { voxel_ijk.x, voxel_ijk.y + 1, voxel_ijk.z };
            if (nb_voxel_ijk.y == 8) {
                nb_tile_ijk.y++;
                nb_voxel_ijk.y = 0;
            }
            if (nb_tile_ijk.y < _tile_dim.y) {
                int nb_idx = TileIjkToIdx(_tile_dim, nb_tile_ijk) * 512 + VoxelIjkToIdx(nb_voxel_ijk);
                if (_is_dof[nb_idx])
                    Ax += _a_y[idx] * _x[nb_idx];
            }
            // z+
            nb_tile_ijk  = tile_ijk;
            nb_voxel_ijk = { voxel_ijk.x, voxel_ijk.y, voxel_ijk.z + 1 };
            if (nb_voxel_ijk.z == 8) {
                nb_tile_ijk.z++;
                nb_voxel_ijk.z = 0;
            }
            if (nb_tile_ijk.z < _tile_dim.z) {
                int nb_idx = TileIjkToIdx(_tile_dim, nb_tile_ijk) * 512 + VoxelIjkToIdx(nb_voxel_ijk);
                if (_is_dof[nb_idx])
                    Ax += _a_z[idx] * _x[nb_idx];
            }
            // x-
            nb_tile_ijk  = tile_ijk;
            nb_voxel_ijk = { voxel_ijk.x - 1, voxel_ijk.y, voxel_ijk.z };
            if (nb_voxel_ijk.x == -1) {
                nb_tile_ijk.x--;
                nb_voxel_ijk.x = 7;
            }
            if (nb_tile_ijk.x >= 0) {
                int nb_idx = TileIjkToIdx(_tile_dim, nb_tile_ijk) * 512 + VoxelIjkToIdx(nb_voxel_ijk);
                if (_is_dof[nb_idx])
                    Ax += _a_x[nb_idx] * _x[nb_idx];
            }
            // y-
            nb_tile_ijk  = tile_ijk;
            nb_voxel_ijk = { voxel_ijk.x, voxel_ijk.y - 1, voxel_ijk.z };
            if (nb_voxel_ijk.y == -1) {
                nb_tile_ijk.y--;
                nb_voxel_ijk.y = 7;
            }
            if (nb_tile_ijk.y >= 0) {
                int nb_idx = TileIjkToIdx(_tile_dim, nb_tile_ijk) * 512 + VoxelIjkToIdx(nb_voxel_ijk);
                if (_is_dof[nb_idx])
                    Ax += _a_y[nb_idx] * _x[nb_idx];
            }
            // z-
            nb_tile_ijk  = tile_ijk;
            nb_voxel_ijk = { voxel_ijk.x, voxel_ijk.y, voxel_ijk.z - 1 };
            if (nb_voxel_ijk.z == -1) {
                nb_tile_ijk.z--;
                nb_voxel_ijk.z = 7;
            }
            if (nb_tile_ijk.z >= 0) {
                int nb_idx = TileIjkToIdx(_tile_dim, nb_tile_ijk) * 512 + VoxelIjkToIdx(nb_voxel_ijk);
                if (_is_dof[nb_idx])
                    Ax += _a_z[nb_idx] * _x[nb_idx];
            }
            mul[i] *= Ax;
        } else
            mul[i] = 0.0f;
        _result[idx] = Ax;
    }
    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float sum = BlockReduce(temp_storage).Sum(mul);
    if (t_id == 0)
        _dot_buffer[tile_idx] = sum;
}

void TrimPoisson::LaplacianDotAsync(std::shared_ptr<DHMemory<float>> _output, std::shared_ptr<DHMemory<float>> _dot_buffer, const std::shared_ptr<DHMemory<float>> _input, cudaStream_t _stream) const
{
    int tile_num             = Prod(tile_dim_);
    float* result            = _output->dev_ptr_;
    float* dot_buffer        = _dot_buffer->dev_ptr_;
    const float* x           = _input->dev_ptr_;
    const char* tile_trimmed = tile_trimmed_->dev_ptr_;
    const char* is_dof       = is_dof_->dev_ptr_;
    const float* a_diag      = a_diag_->dev_ptr_;
    const float* a_x         = a_x_->dev_ptr_;
    const float* a_y         = a_y_->dev_ptr_;
    const float* a_z         = a_z_->dev_ptr_;
    LaplacianDotTrivialKernel<<<tile_num, 128, 0, _stream>>>(result, tile_dim_, dot_buffer, x, tile_trimmed, default_a_diag_, default_a_off_diag_);
    LaplacianDotNonTrivialKernel<<<tile_num, 128, 0, _stream>>>(result, tile_dim_, dot_buffer, x, tile_trimmed, is_dof, a_diag, a_x, a_y, a_z);
}

__global__ void GaussSeidelRestrictTrivialKernel(float* _x, int3 _tile_dim, float* _coarse_b, const char* _tile_trimmed, float _default_a_diag, float _default_inv_a_diag, float _default_a_off_diag, const float* _b)
{
    __shared__ float shared_x[10][10][10];
    __shared__ float shared_b[8][8][8];
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id      = threadIdx.x;

    char tile_trimmed = _tile_trimmed[tile_idx];
    if (tile_trimmed != 1)
        return;
    // load current tile and do phase 0
    int phase = 0;
    for (int i = 0; i < 4; i++) {
        int voxel_idx                                   = i * 128 + t_id;
        int3 voxel_ijk                                  = VoxelIdxToIjk(voxel_idx);
        int idx                                         = tile_idx * 512 + voxel_idx;
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] = _b[idx];
        if ((voxel_ijk.x + voxel_ijk.y + voxel_ijk.z) % 2 == phase)
            shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * _default_inv_a_diag;
    }
    int warp_id = t_id / 32;
    int lane_id = t_id % 32;
    // faces
    if (warp_id == 0) {
        // x-
        int3 nb_tile_ijk                                    = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
        int3 nb_voxel_ijk                                   = { 7, lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        int nb_tile_idx                                     = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx                                    = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx                                          = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _b[nb_idx] * _default_inv_a_diag;
        // x+
        nb_tile_ijk                                         = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
        nb_voxel_ijk                                        = { 0, lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        nb_tile_idx                                         = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx                                        = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx                                              = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _b[nb_idx] * _default_inv_a_diag;
    }
    if (warp_id == 1) {
        // y-
        int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
        int3 nb_voxel_ijk                                   = { lane_id / 4, 7, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        int nb_tile_idx                                     = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx                                    = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx                                          = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = _b[nb_idx] * _default_inv_a_diag;
        // y+
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk                                        = { lane_id / 4, 0, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        nb_tile_idx                                         = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx                                        = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx                                              = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = _b[nb_idx] * _default_inv_a_diag;
    }
    if (warp_id == 2) {
        // z-
        int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
        int3 nb_voxel_ijk                                   = { lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2), 7 };
        int nb_tile_idx                                     = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx                                    = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx                                          = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = _b[nb_idx] * _default_inv_a_diag;
        // z+
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk                                        = { lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2, 0 };
        nb_tile_idx                                         = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx                                        = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx                                              = nb_tile_idx * 512 + nb_voxel_idx;
        shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = _b[nb_idx] * _default_inv_a_diag;
    }
    // edges
    if (warp_id == 0 && lane_id < 8) {
        // x- y-
        int3 nb_tile_ijk  = { tile_ijk.x - 1, tile_ijk.y - 1, tile_ijk.z };
        int3 nb_voxel_ijk = { 7, 7, lane_id };
        int nb_tile_idx   = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx  = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx        = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[0][0][lane_id + 1] = _b[nb_idx] * _default_inv_a_diag;
        // x- y+
        nb_tile_ijk  = { tile_ijk.x - 1, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk = { 7, 0, lane_id };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[0][9][lane_id + 1] = _b[nb_idx] * _default_inv_a_diag;
        // x- z-
        nb_tile_ijk  = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z - 1 };
        nb_voxel_ijk = { 7, lane_id, 7 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[0][lane_id + 1][0] = _b[nb_idx] * _default_inv_a_diag;
    }
    if (warp_id == 1 && lane_id < 8) {
        // y- z+
        int3 nb_tile_ijk  = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z + 1 };
        int3 nb_voxel_ijk = { lane_id, 7, 0 };
        int nb_tile_idx   = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx  = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx        = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[lane_id + 1][0][9] = _b[nb_idx] * _default_inv_a_diag;
        // y+ z+
        nb_tile_ijk  = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z + 1 };
        nb_voxel_ijk = { lane_id, 0, 0 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[lane_id + 1][9][9] = _b[nb_idx] * _default_inv_a_diag;
        // x- z+
        nb_tile_ijk  = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk = { 7, lane_id, 0 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[0][lane_id + 1][9] = _b[nb_idx] * _default_inv_a_diag;
    }
    if (warp_id == 2 && lane_id < 8) {
        // x+ y-
        int3 nb_tile_ijk  = { tile_ijk.x + 1, tile_ijk.y - 1, tile_ijk.z };
        int3 nb_voxel_ijk = { 0, 7, lane_id };
        int nb_tile_idx   = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx  = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx        = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[9][0][lane_id + 1] = _b[nb_idx] * _default_inv_a_diag;
        // x+ y+
        nb_tile_ijk  = { tile_ijk.x + 1, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk = { 0, 0, lane_id };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[9][9][lane_id + 1] = _b[nb_idx] * _default_inv_a_diag;
        // x+ z+
        nb_tile_ijk  = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk = { 0, lane_id, 0 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[9][lane_id + 1][9] = _b[nb_idx] * _default_inv_a_diag;
    }
    if (warp_id == 3 && lane_id < 8) {
        // y- z-
        int3 nb_tile_ijk  = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z - 1 };
        int3 nb_voxel_ijk = { lane_id, 7, 7 };
        int nb_tile_idx   = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx  = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx        = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[lane_id + 1][0][0] = _b[nb_idx] * _default_inv_a_diag;
        // y+ z-
        nb_tile_ijk  = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z - 1 };
        nb_voxel_ijk = { lane_id, 0, 7 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[lane_id + 1][9][0] = _b[nb_idx] * _default_inv_a_diag;
        // x+ z-
        nb_tile_ijk  = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z - 1 };
        nb_voxel_ijk = { 0, lane_id, 7 };
        nb_tile_idx  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx       = nb_tile_idx * 512 + nb_voxel_idx;
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
            shared_x[9][lane_id + 1][0] = _b[nb_idx] * _default_inv_a_diag;
    }

    __syncthreads();
    // phase 1
    for (int i = 0; i < 2; i++) {
        int id         = i * 128 + t_id;
        int a          = id / 32;
        int b          = id % 32;
        int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + !((b / 4 + a) % 2) };
        float val      = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];
        shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _default_inv_a_diag * val;
    }
    if (warp_id == 0) {
        // x-
        int3 nb_tile_ijk  = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
        int3 nb_voxel_ijk = { 7, lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        int nb_tile_idx   = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx  = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx        = nb_tile_idx * 512 + nb_voxel_idx;
        float val         = _b[nb_idx];
        val -= _default_a_off_diag * _default_inv_a_diag * _b[nb_idx - 64];
        val -= _default_a_off_diag * shared_x[1][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[0][nb_voxel_ijk.y][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[0][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2];
        shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _default_inv_a_diag * val;
        // x+
        nb_tile_ijk                                         = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
        nb_voxel_ijk                                        = { 0, lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        nb_tile_idx                                         = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx                                        = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx                                              = nb_tile_idx * 512 + nb_voxel_idx;
        val                                                 = _b[nb_idx];
        val -= _default_a_off_diag * shared_x[8][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * _default_inv_a_diag * _b[nb_idx + 64];
        val -= _default_a_off_diag * shared_x[9][nb_voxel_ijk.y][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[9][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2];
        shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _default_inv_a_diag * val;
    }
    if (warp_id == 1) {
        // y-
        int3 nb_tile_ijk  = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
        int3 nb_voxel_ijk = { lane_id / 4, 7, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        int nb_tile_idx   = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx  = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx        = nb_tile_idx * 512 + nb_voxel_idx;
        float val         = _b[nb_idx];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x][0][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 2][0][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * _default_inv_a_diag * _b[nb_idx - 8];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][1][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 2];
        shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = _default_inv_a_diag * val;
        // y+
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk                                        = { lane_id / 4, 0, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        nb_tile_idx                                         = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx                                        = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx                                              = nb_tile_idx * 512 + nb_voxel_idx;
        val                                                 = _b[nb_idx];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x][9][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 2][9][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][8][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * _default_inv_a_diag * _b[nb_idx + 8];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 2];
        shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = _default_inv_a_diag * val;
    }
    if (warp_id == 2) {
        // z-
        int3 nb_tile_ijk  = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
        int3 nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2, 7 };
        int nb_tile_idx   = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        int nb_voxel_idx  = VoxelIjkToIdx(nb_voxel_ijk);
        int nb_idx        = nb_tile_idx * 512 + nb_voxel_idx;
        float val         = _b[nb_idx];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][0];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][0];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][0];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][0];
        val -= _default_a_off_diag * _default_inv_a_diag * _b[nb_idx - 1];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][1];
        shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = _default_inv_a_diag * val;
        // z+
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk                                        = { lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2), 0 };
        nb_tile_idx                                         = TileIjkToIdx(_tile_dim, nb_tile_ijk);
        nb_voxel_idx                                        = VoxelIjkToIdx(nb_voxel_ijk);
        nb_idx                                              = nb_tile_idx * 512 + nb_voxel_idx;
        val                                                 = _b[nb_idx];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][9];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][9];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][9];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][9];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][8];
        val -= _default_a_off_diag * _default_inv_a_diag * _b[nb_idx + 1];
        shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = _default_inv_a_diag * val;
    }
    __syncthreads();
    // write back x
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int idx        = tile_idx * 512 + voxel_idx;
        _x[idx]        = shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1];
    }
    // residual
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] -= shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] * _default_a_diag;
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] -= shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1] * _default_a_off_diag;
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] -= shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1] * _default_a_off_diag;
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] -= shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1] * _default_a_off_diag;
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] -= shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1] * _default_a_off_diag;
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] -= shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z] * _default_a_off_diag;
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] -= shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2] * _default_a_off_diag;
    }
    __syncthreads();
    // restrict
    if (t_id < 64) {
        int3 rel_voxel_ijk = { t_id / 16, (t_id / 4) % 4, t_id % 4 };
        float val          = 0.0f;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
                for (int c = 0; c < 2; c++) {
                    int3 fine_voxel_ijk = { rel_voxel_ijk.x * 2 + a, rel_voxel_ijk.y * 2 + b, rel_voxel_ijk.z * 2 + c };
                    val += shared_b[fine_voxel_ijk.x][fine_voxel_ijk.y][fine_voxel_ijk.z];
                }
        int3 coarse_tile_ijk     = { tile_ijk.x / 2, tile_ijk.y / 2, tile_ijk.z / 2 };
        int3 coarse_voxel_offset = { 4 * (tile_ijk.x % 2), 4 * (tile_ijk.y % 2), 4 * (tile_ijk.z % 2) };
        int3 coarse_voxel_ijk    = { coarse_voxel_offset.x + rel_voxel_ijk.x, coarse_voxel_offset.y + rel_voxel_ijk.y, coarse_voxel_offset.z + rel_voxel_ijk.z };
        int3 coarse_ijk          = { coarse_tile_ijk.x * 8 + coarse_voxel_ijk.x, coarse_tile_ijk.y * 8 + coarse_voxel_ijk.y, coarse_tile_ijk.z * 8 + coarse_voxel_ijk.z };
        int3 coarse_tile_dim     = { _tile_dim.x / 2, _tile_dim.y / 2, _tile_dim.z / 2 };
        int coarse_idx           = IjkToIdx(coarse_tile_dim, coarse_ijk);
        _coarse_b[coarse_idx]    = val * 0.125f;
    }
}

__global__ void GaussSeidelRestrictNonTrivialKernel(float* _x, int3 _tile_dim, float* _coarse_b, const char* _tile_trimmed, const char* _is_dof, const float* _a_diag, const float* _a_x, const float* _a_y, const float* _a_z, const float* _b)
{
    __shared__ float shared_x[10][10][10];
    __shared__ float shared_b[8][8][8];
    __shared__ char shared_is_dof[10][10][10];
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id      = threadIdx.x;

    char tile_trimmed = _tile_trimmed[tile_idx];
    if (tile_trimmed != 2)
        return;

    int phase = 0;
    // load current tile and do phase 0
    for (int i = 0; i < 4; i++) {
        int voxel_idx                                                    = i * 128 + t_id;
        int3 voxel_ijk                                                   = VoxelIdxToIjk(voxel_idx);
        int idx                                                          = tile_idx * 512 + voxel_idx;
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z]                  = _b[idx];
        shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _is_dof[idx];
        if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] && (voxel_ijk.x + voxel_ijk.y + voxel_ijk.z) % 2 == phase)
            shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] / _a_diag[idx];
        else
            shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = 0.0f;
    }
    int warp_id = t_id / 32;
    int lane_id = t_id % 32;
    // face
    if (warp_id < 2) {
        int u            = t_id / 8;
        int v            = t_id % 8;
        // x-
        int3 nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
        if (nb_tile_ijk.x >= 0) {
            int3 nb_voxel_ijk              = { 7, u, v };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[0][u + 1][v + 1] = _is_dof[nb_idx];
            if (shared_is_dof[0][u + 1][v + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[0][u + 1][v + 1] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[0][u + 1][v + 1] = 0;
        // y-
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
        if (nb_tile_ijk.y >= 0) {
            int3 nb_voxel_ijk              = { u, 7, v };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[u + 1][0][v + 1] = _is_dof[nb_idx];
            if (shared_is_dof[u + 1][0][v + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[u + 1][0][v + 1] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[u + 1][0][v + 1] = 0;
        // z-
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
        if (nb_tile_ijk.z >= 0) {
            int3 nb_voxel_ijk              = { u, v, 7 };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[u + 1][v + 1][0] = _is_dof[nb_idx];
            if (shared_is_dof[u + 1][v + 1][0] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[u + 1][v + 1][0] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[u + 1][v + 1][0] = 0;
    } else {
        int group_id     = t_id - 64;
        int u            = group_id / 8;
        int v            = group_id % 8;
        // x+
        int3 nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
        if (nb_tile_ijk.x < _tile_dim.x) {
            int3 nb_voxel_ijk              = { 0, u, v };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[9][u + 1][v + 1] = _is_dof[nb_idx];
            if (shared_is_dof[9][u + 1][v + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[9][u + 1][v + 1] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[9][u + 1][v + 1] = 0;
        // y+
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
        if (nb_tile_ijk.y < _tile_dim.y) {
            int3 nb_voxel_ijk              = { u, 0, v };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[u + 1][9][v + 1] = _is_dof[nb_idx];
            if (shared_is_dof[u + 1][9][v + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[u + 1][9][v + 1] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[u + 1][9][v + 1] = 0;
        // z+
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
        if (nb_tile_ijk.z < _tile_dim.z) {
            int3 nb_voxel_ijk              = { u, v, 0 };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[u + 1][v + 1][9] = _is_dof[nb_idx];
            if (shared_is_dof[u + 1][v + 1][9] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[u + 1][v + 1][9] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[u + 1][v + 1][9] = 0;
    }
    // edge
    if (warp_id == 0 && lane_id < 8) {
        // x- y-
        int3 nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y - 1, tile_ijk.z };
        if (nb_tile_ijk.x >= 0 && nb_tile_ijk.y >= 0) {
            int3 nb_voxel_ijk                = { 7, 7, lane_id };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[0][0][lane_id + 1] = _is_dof[nb_idx];
            if (shared_is_dof[0][0][lane_id + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[0][0][lane_id + 1] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[0][0][lane_id + 1] = 0;
        // x- y+
        nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y + 1, tile_ijk.z };
        if (nb_tile_ijk.x >= 0 && nb_tile_ijk.y < _tile_dim.y) {
            int3 nb_voxel_ijk                = { 7, 0, lane_id };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[0][9][lane_id + 1] = _is_dof[nb_idx];
            if (shared_is_dof[0][9][lane_id + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[0][9][lane_id + 1] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[0][9][lane_id + 1] = 0;
        // x- z-
        nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z - 1 };
        if (nb_tile_ijk.x >= 0 && nb_tile_ijk.z >= 0) {
            int3 nb_voxel_ijk                = { 7, lane_id, 7 };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[0][lane_id + 1][0] = _is_dof[nb_idx];
            if (shared_is_dof[0][lane_id + 1][0] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[0][lane_id + 1][0] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[0][lane_id + 1][0] = 0;
    }
    if (warp_id == 1 && lane_id < 8) {
        // y- z+
        int3 nb_tile_ijk = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z + 1 };
        if (nb_tile_ijk.y >= 0 && nb_tile_ijk.z < _tile_dim.z) {
            int3 nb_voxel_ijk                = { lane_id, 7, 0 };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[lane_id + 1][0][9] = _is_dof[nb_idx];
            if (shared_is_dof[lane_id + 1][0][9] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[lane_id + 1][0][9] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[lane_id + 1][0][9] = 0;
        // y+ z+
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z + 1 };
        if (nb_tile_ijk.y < _tile_dim.y && nb_tile_ijk.z < _tile_dim.z) {
            int3 nb_voxel_ijk                = { lane_id, 0, 0 };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[lane_id + 1][9][9] = _is_dof[nb_idx];
            if (shared_is_dof[lane_id + 1][9][9] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[lane_id + 1][9][9] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[lane_id + 1][9][9] = 0;
        // x- z+
        nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z + 1 };
        if (nb_tile_ijk.x >= 0 && nb_tile_ijk.z < _tile_dim.z) {
            int3 nb_voxel_ijk                = { 7, lane_id, 0 };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[0][lane_id + 1][9] = _is_dof[nb_idx];
            if (shared_is_dof[0][lane_id + 1][9] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[0][lane_id + 1][9] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[0][lane_id + 1][9] = 0;
    }
    if (warp_id == 2 && lane_id < 8) {
        // x+ y-
        int3 nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y - 1, tile_ijk.z };
        if (nb_tile_ijk.x < _tile_dim.x && nb_tile_ijk.y >= 0) {
            int3 nb_voxel_ijk                = { 0, 7, lane_id };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[9][0][lane_id + 1] = _is_dof[nb_idx];
            if (shared_is_dof[9][0][lane_id + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[9][0][lane_id + 1] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[9][0][lane_id + 1] = 0;
        // x+ y+
        nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y + 1, tile_ijk.z };
        if (nb_tile_ijk.x < _tile_dim.x && nb_tile_ijk.y < _tile_dim.y) {
            int3 nb_voxel_ijk                = { 0, 0, lane_id };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[9][9][lane_id + 1] = _is_dof[nb_idx];
            if (shared_is_dof[9][9][lane_id + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[9][9][lane_id + 1] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[9][9][lane_id + 1] = 0;

        // x+ z+
        nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z + 1 };
        if (nb_tile_ijk.x < _tile_dim.x && nb_tile_ijk.z < _tile_dim.z) {
            int3 nb_voxel_ijk                = { 0, lane_id, 0 };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[9][lane_id + 1][9] = _is_dof[nb_idx];
            if (shared_is_dof[9][lane_id + 1][9] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[9][lane_id + 1][9] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[9][lane_id + 1][9] = 0;
    }
    if (warp_id == 3 && lane_id < 8) {
        // y- z-
        int3 nb_tile_ijk = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z - 1 };
        if (nb_tile_ijk.y >= 0 && nb_tile_ijk.z >= 0) {
            int3 nb_voxel_ijk                = { lane_id, 7, 7 };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[lane_id + 1][0][0] = _is_dof[nb_idx];
            if (shared_is_dof[lane_id + 1][0][0] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[lane_id + 1][0][0] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[lane_id + 1][0][0] = 0;
        // y+ z-
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z - 1 };
        if (nb_tile_ijk.y < _tile_dim.y && nb_tile_ijk.z >= 0) {
            int3 nb_voxel_ijk                = { lane_id, 0, 7 };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[lane_id + 1][9][0] = _is_dof[nb_idx];
            if (shared_is_dof[lane_id + 1][9][0] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[lane_id + 1][9][0] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[lane_id + 1][9][0] = 0;
        // x+ z-
        nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z - 1 };
        if (nb_tile_ijk.x < _tile_dim.x && nb_tile_ijk.z >= 0) {
            int3 nb_voxel_ijk                = { 0, lane_id, 7 };
            int nb_tile_idx                  = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx                 = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                       = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[9][lane_id + 1][0] = _is_dof[nb_idx];
            if (shared_is_dof[9][lane_id + 1][0] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == phase)
                shared_x[9][lane_id + 1][0] = _b[nb_idx] / _a_diag[nb_idx];
        } else
            shared_is_dof[9][lane_id + 1][0] = 0;
    }
    __syncthreads();
    // phase 1
    phase = 1;
    for (int i = 0; i < 2; i++) {
        int id         = i * 128 + t_id;
        int a          = id / 32;
        int b          = id % 32;
        int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + !((b / 4 + a) % 2) };
        if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1]) {
            float val = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
            int3 ijk  = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
            int idx   = IjkToIdx(_tile_dim, ijk);
            if (shared_is_dof[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1])
                val -= _a_x[IjkToIdx(_tile_dim, { ijk.x - 1, ijk.y, ijk.z })] * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1])
                val -= _a_x[idx] * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1])
                val -= _a_y[IjkToIdx(_tile_dim, { ijk.x, ijk.y - 1, ijk.z })] * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1])
                val -= _a_y[idx] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { ijk.x, ijk.y, ijk.z - 1 })] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2])
                val -= _a_z[idx] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];
            shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = val / _a_diag[idx];
        }
    }
    if (warp_id == 0) {
        // x-
        int3 nb_voxel_ijk = { 7, lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        if (shared_is_dof[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
            int3 nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
            int3 nb_ijk      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx       = IjkToIdx(_tile_dim, nb_ijk);
            float val        = _b[nb_idx];
            if (_is_dof[nb_idx - 64])
                val -= _a_x[nb_idx - 64] * (_b[nb_idx - 64] / _a_diag[nb_idx - 64]);
            if (shared_is_dof[1][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1])
                val -= _a_x[nb_idx] * shared_x[1][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1];
            if (shared_is_dof[0][nb_voxel_ijk.y][nb_voxel_ijk.z + 1]) {
                val -= _a_y[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y - 1, nb_ijk.z })] * shared_x[0][nb_voxel_ijk.y][nb_voxel_ijk.z + 1];
            }
            if (shared_is_dof[0][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1])
                val -= _a_y[nb_idx] * shared_x[0][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1];
            if (shared_is_dof[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y, nb_ijk.z - 1 })] * shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z];
            if (shared_is_dof[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2])
                val -= _a_z[nb_idx] * shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2];
            shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = val / _a_diag[nb_idx];
        }
        // x+
        nb_voxel_ijk = { 0, lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        if (shared_is_dof[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
            int3 nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
            int3 nb_ijk      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx       = IjkToIdx(_tile_dim, nb_ijk);
            float val        = _b[nb_idx];
            if (shared_is_dof[8][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1])
                val -= _a_x[IjkToIdx(_tile_dim, { nb_ijk.x - 1, nb_ijk.y, nb_ijk.z })] * shared_x[8][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1];
            if (_is_dof[nb_idx + 64])
                val -= _a_x[nb_idx] * (_b[nb_idx + 64] / _a_diag[nb_idx + 64]);
            if (shared_is_dof[9][nb_voxel_ijk.y][nb_voxel_ijk.z + 1])
                val -= _a_y[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y - 1, nb_ijk.z })] * shared_x[9][nb_voxel_ijk.y][nb_voxel_ijk.z + 1];
            if (shared_is_dof[9][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1])
                val -= _a_y[nb_idx] * shared_x[9][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1];
            if (shared_is_dof[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y, nb_ijk.z - 1 })] * shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z];
            if (shared_is_dof[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2])
                val -= _a_z[nb_idx] * shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2];
            shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = val / _a_diag[nb_idx];
        }
    }
    if (warp_id == 1) {
        // y-
        int3 nb_voxel_ijk = { lane_id / 4, 7, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        if (shared_is_dof[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1]) {
            int3 nb_tile_ijk = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
            int3 nb_ijk      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx       = IjkToIdx(_tile_dim, nb_ijk);
            float val        = _b[nb_idx];
            if (shared_is_dof[nb_voxel_ijk.x][0][nb_voxel_ijk.z + 1])
                val -= _a_x[IjkToIdx(_tile_dim, { nb_ijk.x - 1, nb_ijk.y, nb_ijk.z })] * shared_x[nb_voxel_ijk.x][0][nb_voxel_ijk.z + 1];
            if (shared_is_dof[nb_voxel_ijk.x + 2][0][nb_voxel_ijk.z + 1])
                val -= _a_x[nb_idx] * shared_x[nb_voxel_ijk.x + 2][0][nb_voxel_ijk.z + 1];
            if (_is_dof[nb_idx - 8])
                val -= _a_y[nb_idx - 8] * (_b[nb_idx - 8] / _a_diag[nb_idx - 8]);
            if (shared_is_dof[nb_voxel_ijk.x + 1][1][nb_voxel_ijk.z + 1])
                val -= _a_y[nb_idx] * shared_x[nb_voxel_ijk.x + 1][1][nb_voxel_ijk.z + 1];
            if (shared_is_dof[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y, nb_ijk.z - 1 })] * shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z];
            if (shared_is_dof[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 2])
                val -= _a_z[nb_idx] * shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 2];
            shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = val / _a_diag[nb_idx];
        }
        // y+
        nb_voxel_ijk = { lane_id / 4, 0, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        if (shared_is_dof[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1]) {
            int3 nb_tile_ijk = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
            int3 nb_ijk      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx       = IjkToIdx(_tile_dim, nb_ijk);
            float val        = _b[nb_idx];
            if (shared_is_dof[nb_voxel_ijk.x][9][nb_voxel_ijk.z + 1])
                val -= _a_x[IjkToIdx(_tile_dim, { nb_ijk.x - 1, nb_ijk.y, nb_ijk.z })] * shared_x[nb_voxel_ijk.x][9][nb_voxel_ijk.z + 1];
            if (shared_is_dof[nb_voxel_ijk.x + 2][9][nb_voxel_ijk.z + 1])
                val -= _a_x[nb_idx] * shared_x[nb_voxel_ijk.x + 2][9][nb_voxel_ijk.z + 1];
            if (shared_is_dof[nb_voxel_ijk.x + 1][8][nb_voxel_ijk.z + 1])
                val -= _a_y[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y - 1, nb_ijk.z })] * shared_x[nb_voxel_ijk.x + 1][8][nb_voxel_ijk.z + 1];
            if (_is_dof[nb_idx + 8])
                val -= _a_y[nb_idx] * (_b[nb_idx + 8] / _a_diag[nb_idx + 8]);
            if (shared_is_dof[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y, nb_ijk.z - 1 })] * shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z];
            if (shared_is_dof[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 2])
                val -= _a_z[nb_idx] * shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 2];
            shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = val / _a_diag[nb_idx];
        }
    }
    if (warp_id == 2) {
        // z-
        int3 nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2, 7 };
        if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0]) {
            int3 nb_tile_ijk = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
            int3 nb_ijk      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx       = IjkToIdx(_tile_dim, nb_ijk);
            float val        = _b[nb_idx];
            if (shared_is_dof[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][0])
                val -= _a_x[IjkToIdx(_tile_dim, { nb_ijk.x - 1, nb_ijk.y, nb_ijk.z })] * shared_x[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][0];
            if (shared_is_dof[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][0])
                val -= _a_x[nb_idx] * shared_x[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][0];
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][0])
                val -= _a_y[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y - 1, nb_ijk.z })] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][0];
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][0])
                val -= _a_y[nb_idx] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][0];
            if (_is_dof[nb_idx - 1])
                val -= _a_z[nb_idx - 1] * (_b[nb_idx - 1] / _a_diag[nb_idx - 1]);
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][1])
                val -= _a_z[nb_idx] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][1];
            shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = val / _a_diag[nb_idx];
        }
        // z+
        nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2), 0 };
        if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9]) {
            int3 nb_tile_ijk = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
            int3 nb_ijk      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx       = IjkToIdx(_tile_dim, nb_ijk);
            float val        = _b[nb_idx];
            if (shared_is_dof[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][9])
                val -= _a_x[IjkToIdx(_tile_dim, { nb_ijk.x - 1, nb_ijk.y, nb_ijk.z })] * shared_x[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][9];
            if (shared_is_dof[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][9])
                val -= _a_x[nb_idx] * shared_x[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][9];
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][9])
                val -= _a_y[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y - 1, nb_ijk.z })] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][9];
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][9])
                val -= _a_y[nb_idx] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][9];
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][8])
                val -= _a_z[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y, nb_ijk.z - 1 })] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][8];
            if (_is_dof[nb_idx + 1])
                val -= _a_z[nb_idx] * (_b[nb_idx + 1] / _a_diag[nb_idx + 1]);
            shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = val / _a_diag[nb_idx];
        }
    }
    __syncthreads();
    // write back x
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int idx        = tile_idx * 512 + voxel_idx;
        _x[idx]        = shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1];
    }
    // residual
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1]) {
            int3 ijk  = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
            int idx   = IjkToIdx(_tile_dim, ijk);
            float val = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] - _a_diag[idx] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1])
                val -= _a_x[IjkToIdx(_tile_dim, { ijk.x - 1, ijk.y, ijk.z })] * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1])
                val -= _a_x[idx] * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1])
                val -= _a_y[IjkToIdx(_tile_dim, { ijk.x, ijk.y - 1, ijk.z })] * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1])
                val -= _a_y[idx] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { ijk.x, ijk.y, ijk.z - 1 })] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2])
                val -= _a_z[idx] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];
            shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] = val;
        } else
            shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] = 0.0f;
    }
    __syncthreads();
    // restrict
    if (t_id < 64) {
        int3 rel_voxel_ijk = { t_id / 16, (t_id / 4) % 4, t_id % 4 };
        float val          = 0.0f;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
                for (int c = 0; c < 2; c++) {
                    int3 fine_voxel_ijk = { rel_voxel_ijk.x * 2 + a, rel_voxel_ijk.y * 2 + b, rel_voxel_ijk.z * 2 + c };
                    val += shared_b[fine_voxel_ijk.x][fine_voxel_ijk.y][fine_voxel_ijk.z];
                }
        int3 coarse_tile_ijk     = { tile_ijk.x / 2, tile_ijk.y / 2, tile_ijk.z / 2 };
        int3 coarse_voxel_offset = { 4 * (tile_ijk.x % 2), 4 * (tile_ijk.y % 2), 4 * (tile_ijk.z % 2) };
        int3 coarse_voxel_ijk    = { coarse_voxel_offset.x + rel_voxel_ijk.x, coarse_voxel_offset.y + rel_voxel_ijk.y, coarse_voxel_offset.z + rel_voxel_ijk.z };
        int3 coarse_ijk          = { coarse_tile_ijk.x * 8 + coarse_voxel_ijk.x, coarse_tile_ijk.y * 8 + coarse_voxel_ijk.y, coarse_tile_ijk.z * 8 + coarse_voxel_ijk.z };
        int3 coarse_tile_dim     = { _tile_dim.x / 2, _tile_dim.y / 2, _tile_dim.z / 2 };
        int coarse_idx           = IjkToIdx(coarse_tile_dim, coarse_ijk);
        _coarse_b[coarse_idx]    = val * 0.125f;
    }
}
void TrimPoisson::GaussSeidelRestrictAsync(std::shared_ptr<DHMemory<float>> _coarse_b, cudaStream_t _stream)
{
    float* x        = x_->dev_ptr_;
    int tile_num    = Prod(tile_dim_);
    float* coarse_b = _coarse_b->dev_ptr_;
    const float* b  = b_->dev_ptr_;

    float uniform_coef = -default_a_off_diag_;
    float inv_coef     = 1.0f / uniform_coef;

    const char* is_dof  = is_dof_->dev_ptr_;
    const float* a_diag = a_diag_->dev_ptr_;
    const float* a_x    = a_x_->dev_ptr_;
    const float* a_y    = a_y_->dev_ptr_;
    const float* a_z    = a_z_->dev_ptr_;

    const char* tile_trimmed = tile_trimmed_->dev_ptr_;
    float default_inv_a_diag = 1.0f / default_a_diag_;
    GaussSeidelRestrictTrivialKernel<<<tile_num, 128, 0, _stream>>>(x, tile_dim_, coarse_b, tile_trimmed, default_a_diag_, default_inv_a_diag, default_a_off_diag_, b);
    GaussSeidelRestrictNonTrivialKernel<<<tile_num, 128, 0, _stream>>>(x, tile_dim_, coarse_b, tile_trimmed, is_dof, a_diag, a_x, a_y, a_z, b);
}

__global__ void ProlongGaussSeidelDotTrivialKernel(float* _dst_x, int3 _tile_dim, const float* _x, const float* _coarse_x, const char* _tile_trimmed, float _default_inv_a_diag, float _default_a_off_diag, const float* _b, float* _dot_buffer, bool _do_dot)
{
    __shared__ float shared_x[10][10][10];
    __shared__ float shared_b[8][8][8];
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id      = threadIdx.x;

    char tile_trimmed = _tile_trimmed[tile_idx];
    if (tile_trimmed != 1)
        return;

    // prolong
    int3 coarse_tile_dim = { _tile_dim.x / 2, _tile_dim.y / 2, _tile_dim.z / 2 };
    int3 coarse_tile_ijk = { tile_ijk.x / 2, tile_ijk.y / 2, tile_ijk.z / 2 };
    int coarse_tile_idx  = TileIjkToIdx(coarse_tile_dim, coarse_tile_ijk);
    for (int i = 0; i < 4; i++) {
        int voxel_idx                                   = i * 128 + t_id;
        int3 voxel_ijk                                  = VoxelIdxToIjk(voxel_idx);
        int idx                                         = tile_idx * 512 + voxel_idx;
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] = _b[idx];
        if ((voxel_ijk.x + voxel_ijk.y + voxel_ijk.z) % 2 == 0) {
            int3 coarse_voxel_ijk                                       = { (tile_ijk.x % 2) * 4 + voxel_ijk.x / 2, (tile_ijk.y % 2) * 4 + voxel_ijk.y / 2, (tile_ijk.z % 2) * 4 + voxel_ijk.z / 2 };
            int coarse_voxel_idx                                        = VoxelIjkToIdx(coarse_voxel_ijk);
            int coarse_idx                                              = coarse_tile_idx * 512 + coarse_voxel_idx;
            shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _x[idx] + 2.0f * _coarse_x[coarse_idx];
        }
    }
    int warp_id = t_id / 32;
    int lane_id = t_id % 32;
    // faces
    if (warp_id == 0) {
        // x-
        int3 nb_tile_ijk                                    = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
        int3 nb_voxel_ijk                                   = { 7, lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        int3 coarse_nb_ijk                                  = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // x+
        nb_tile_ijk                                         = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
        nb_voxel_ijk                                        = { 0, lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        nb_ijk                                              = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk                                       = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
    }
    if (warp_id == 1) {
        // y-
        int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
        int3 nb_voxel_ijk                                   = { lane_id / 4, 7, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        int3 coarse_nb_ijk                                  = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // y+
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk                                        = { lane_id / 4, 0, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        nb_ijk                                              = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk                                       = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
    }
    if (warp_id == 2) {
        // z-
        int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
        int3 nb_voxel_ijk                                   = { lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2), 7 };
        int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        int3 coarse_nb_ijk                                  = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // z+
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk                                        = { lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2, 0 };
        nb_ijk                                              = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk                                       = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
    }
    // edges
    if (warp_id == 0 && lane_id < 8) {
        // x- y-
        int3 nb_tile_ijk   = { tile_ijk.x - 1, tile_ijk.y - 1, tile_ijk.z };
        int3 nb_voxel_ijk  = { 7, 7, lane_id };
        int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[0][0][lane_id + 1] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // x- y+
        nb_tile_ijk   = { tile_ijk.x - 1, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk  = { 7, 0, lane_id };
        nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[0][9][lane_id + 1] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // x- z-
        nb_tile_ijk   = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z - 1 };
        nb_voxel_ijk  = { 7, lane_id, 7 };
        nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[0][lane_id + 1][0] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
    }
    if (warp_id == 1 && lane_id < 8) {
        // y- z+
        int3 nb_tile_ijk   = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z + 1 };
        int3 nb_voxel_ijk  = { lane_id, 7, 0 };
        int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[lane_id + 1][0][9] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // y+ z+
        nb_tile_ijk   = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z + 1 };
        nb_voxel_ijk  = { lane_id, 0, 0 };
        nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[lane_id + 1][9][9] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // x- z+
        nb_tile_ijk   = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk  = { 7, lane_id, 0 };
        nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[0][lane_id + 1][9] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
    }
    if (warp_id == 2 && lane_id < 8) {
        // x+ y-
        int3 nb_tile_ijk   = { tile_ijk.x + 1, tile_ijk.y - 1, tile_ijk.z };
        int3 nb_voxel_ijk  = { 0, 7, lane_id };
        int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[9][0][lane_id + 1] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // x+ y+
        nb_tile_ijk   = { tile_ijk.x + 1, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk  = { 0, 0, lane_id };
        nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[9][9][lane_id + 1] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // x+ z+
        nb_tile_ijk   = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk  = { 0, lane_id, 0 };
        nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[9][lane_id + 1][9] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
    }
    if (warp_id == 3 && lane_id < 8) {
        // y- z-
        int3 nb_tile_ijk   = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z - 1 };
        int3 nb_voxel_ijk  = { lane_id, 7, 7 };
        int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[lane_id + 1][0][0] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // y+ z-
        nb_tile_ijk   = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z - 1 };
        nb_voxel_ijk  = { lane_id, 0, 7 };
        nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[lane_id + 1][9][0] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
        // x+ z-
        nb_tile_ijk   = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z - 1 };
        nb_voxel_ijk  = { 0, lane_id, 7 };
        nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        if ((nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0)
            shared_x[9][lane_id + 1][0] = _x[IjkToIdx(_tile_dim, nb_ijk)] + 2.0f * _coarse_x[IjkToIdx(coarse_tile_dim, coarse_nb_ijk)];
    }
    __syncthreads();
    // phase 1
    for (int i = 0; i < 2; i++) {
        int id         = i * 128 + t_id;
        int a          = id / 32;
        int b          = id % 32;
        int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + !((b / 4 + a) % 2) };
        float val      = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];
        shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _default_inv_a_diag * val;
    }
    if (warp_id == 0) {
        // x-
        int3 nb_tile_ijk   = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
        int3 nb_voxel_ijk  = { 7, lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        int nb_idx         = IjkToIdx(_tile_dim, nb_ijk);
        int coarse_nb_idx  = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
        float val          = _b[nb_idx];
        val -= _default_a_off_diag * (_x[nb_idx - 64] + 2.0f * _coarse_x[coarse_nb_idx]);
        val -= _default_a_off_diag * shared_x[1][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[0][nb_voxel_ijk.y][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[0][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2];
        shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _default_inv_a_diag * val;
        // x+
        nb_tile_ijk                                         = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
        nb_voxel_ijk                                        = { 0, lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        nb_ijk                                              = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk                                       = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        nb_idx                                              = IjkToIdx(_tile_dim, nb_ijk);
        coarse_nb_idx                                       = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
        val                                                 = _b[nb_idx];
        val -= _default_a_off_diag * shared_x[8][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * (_x[nb_idx + 64] + 2.0f * _coarse_x[coarse_nb_idx]);
        val -= _default_a_off_diag * shared_x[9][nb_voxel_ijk.y][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[9][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2];
        shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _default_inv_a_diag * val;
    }
    if (warp_id == 1) {
        // y-
        int3 nb_tile_ijk   = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
        int3 nb_voxel_ijk  = { lane_id / 4, 7, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        int nb_idx         = IjkToIdx(_tile_dim, nb_ijk);
        int coarse_nb_idx  = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
        float val          = _b[nb_idx];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x][0][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 2][0][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * (_x[nb_idx - 8] + 2.0f * _coarse_x[coarse_nb_idx]);
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][1][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 2];
        shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = _default_inv_a_diag * val;
        // y+
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
        nb_voxel_ijk                                        = { lane_id / 4, 0, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        nb_ijk                                              = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk                                       = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        nb_idx                                              = IjkToIdx(_tile_dim, nb_ijk);
        coarse_nb_idx                                       = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
        val                                                 = _b[nb_idx];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x][9][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 2][9][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][8][nb_voxel_ijk.z + 1];
        val -= _default_a_off_diag * (_x[nb_idx + 8] + 2.0f * _coarse_x[coarse_nb_idx]);
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 2];
        shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = _default_inv_a_diag * val;
    }
    if (warp_id == 2) {
        // z-
        int3 nb_tile_ijk   = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
        int3 nb_voxel_ijk  = { lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2, 7 };
        int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        int nb_idx         = IjkToIdx(_tile_dim, nb_ijk);
        int coarse_nb_idx  = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
        float val          = _b[nb_idx];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][0];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][0];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][0];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][0];
        val -= _default_a_off_diag * (_x[nb_idx - 1] + 2.0f * _coarse_x[coarse_nb_idx]);
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][1];
        shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = _default_inv_a_diag * val;
        // z+
        nb_tile_ijk                                         = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
        nb_voxel_ijk                                        = { lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2), 0 };
        nb_ijk                                              = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
        coarse_nb_ijk                                       = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
        nb_idx                                              = IjkToIdx(_tile_dim, nb_ijk);
        coarse_nb_idx                                       = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
        val                                                 = _b[nb_idx];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][9];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][9];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][9];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][9];
        val -= _default_a_off_diag * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][8];
        val -= _default_a_off_diag * (_x[nb_idx + 1] + 2.0f * _coarse_x[coarse_nb_idx]);
        shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = _default_inv_a_diag * val;
    }
    __syncthreads();
    // phase 0
    for (int i = 0; i < 2; i++) {
        int id         = i * 128 + t_id;
        int a          = id / 32;
        int b          = id % 32;
        int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + (b / 4 + a) % 2 };
        float val      = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
        val -= _default_a_off_diag * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];
        shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _default_inv_a_diag * val;
    }
    __syncthreads();
    // write back x
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int idx        = tile_idx * 512 + voxel_idx;
        _dst_x[idx]    = shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1];
    }
    // dot
    if (_do_dot) {
        float mul[4];
        for (int i = 0; i < 4; i++) {
            int voxel_idx  = i * 128 + t_id;
            int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
            mul[i]         = shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] * shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
        }
        using BlockReduce = cub::BlockReduce<float, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        float sum = BlockReduce(temp_storage).Sum(mul);
        if (t_id == 0)
            _dot_buffer[tile_idx] = sum;
    }
}

__global__ void ProlongGaussSeidelDotNonTrivialKernel(float* _dst_x, int3 _tile_dim, const float* _x, const float* _coarse_x, const char* _tile_trimmed, const char* _is_dof, const float* _a_diag, const float* _a_x, const float* _a_y, const float* _a_z, const float* _b, float* _dot_buffer, bool _do_dot)
{
    __shared__ float shared_x[10][10][10];
    __shared__ float shared_b[8][8][8];
    __shared__ char shared_is_dof[10][10][10];
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id      = threadIdx.x;

    char tile_trimmed = _tile_trimmed[tile_idx];
    if (tile_trimmed != 2)
        return;

    // prolong
    int3 coarse_tile_dim = { _tile_dim.x / 2, _tile_dim.y / 2, _tile_dim.z / 2 };
    int3 coarse_tile_ijk = { tile_ijk.x / 2, tile_ijk.y / 2, tile_ijk.z / 2 };
    int coarse_tile_idx  = TileIjkToIdx(coarse_tile_dim, coarse_tile_ijk);
    for (int i = 0; i < 4; i++) {
        int voxel_idx                                                    = i * 128 + t_id;
        int3 voxel_ijk                                                   = VoxelIdxToIjk(voxel_idx);
        int idx                                                          = tile_idx * 512 + voxel_idx;
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z]                  = _b[idx];
        shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _is_dof[idx];
        if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] && (voxel_ijk.x + voxel_ijk.y + voxel_ijk.z) % 2 == 0) {
            int3 coarse_voxel_ijk                                       = { (tile_ijk.x % 2) * 4 + voxel_ijk.x / 2, (tile_ijk.y % 2) * 4 + voxel_ijk.y / 2, (tile_ijk.z % 2) * 4 + voxel_ijk.z / 2 };
            int coarse_voxel_idx                                        = VoxelIjkToIdx(coarse_voxel_ijk);
            int coarse_idx                                              = coarse_tile_idx * 512 + coarse_voxel_idx;
            shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _x[idx] + 2.0f * _coarse_x[coarse_idx];
        } else
            shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = 0.0f;
    }
    int warp_id = t_id / 32;
    int lane_id = t_id % 32;
    // faces
    if (warp_id < 2) {
        int u            = t_id / 8;
        int v            = t_id % 8;
        // x-
        int3 nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
        if (nb_tile_ijk.x >= 0) {
            int3 nb_voxel_ijk              = { 7, u, v };
            int3 nb_ijk                    = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                     = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[0][u + 1][v + 1] = _is_dof[nb_idx];
            if (shared_is_dof[0][u + 1][v + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk        = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx         = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[0][u + 1][v + 1] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[0][u + 1][v + 1] = 0;
        // y-
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
        if (nb_tile_ijk.y >= 0) {
            int3 nb_voxel_ijk              = { u, 7, v };
            int3 nb_ijk                    = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                     = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[u + 1][0][v + 1] = _is_dof[nb_idx];
            if (shared_is_dof[u + 1][0][v + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk        = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx         = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[u + 1][0][v + 1] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[u + 1][0][v + 1] = 0;
        // z-
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
        if (nb_tile_ijk.z >= 0) {
            int3 nb_voxel_ijk              = { u, v, 7 };
            int3 nb_ijk                    = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                     = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[u + 1][v + 1][0] = _is_dof[nb_idx];
            if (shared_is_dof[u + 1][v + 1][0] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk        = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx         = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[u + 1][v + 1][0] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[u + 1][v + 1][0] = 0;
    } else {
        int group_id     = t_id - 64;
        int u            = group_id / 8;
        int v            = group_id % 8;
        // x+
        int3 nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
        if (nb_tile_ijk.x < _tile_dim.x) {
            int3 nb_voxel_ijk              = { 0, u, v };
            int3 nb_ijk                    = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                     = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[9][u + 1][v + 1] = _is_dof[nb_idx];
            if (shared_is_dof[9][u + 1][v + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk        = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx         = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[9][u + 1][v + 1] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[9][u + 1][v + 1] = 0;
        // y+
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
        if (nb_tile_ijk.y < _tile_dim.y) {
            int3 nb_voxel_ijk              = { u, 0, v };
            int3 nb_ijk                    = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                     = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[u + 1][9][v + 1] = _is_dof[nb_idx];
            if (shared_is_dof[u + 1][9][v + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk        = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx         = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[u + 1][9][v + 1] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[u + 1][9][v + 1] = 0;
        // z+
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
        if (nb_tile_ijk.z < _tile_dim.z) {
            int3 nb_voxel_ijk              = { u, v, 0 };
            int3 nb_ijk                    = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                     = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[u + 1][v + 1][9] = _is_dof[nb_idx];
            if (shared_is_dof[u + 1][v + 1][9] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk        = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx         = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[u + 1][v + 1][9] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[u + 1][v + 1][9] = 0;
    }
    // edges
    if (warp_id == 0 && lane_id < 8) {
        // x- y-
        int3 nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y - 1, tile_ijk.z };
        if (nb_tile_ijk.x >= 0 && nb_tile_ijk.y >= 0) {
            int3 nb_voxel_ijk                = { 7, 7, lane_id };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[0][0][lane_id + 1] = _is_dof[nb_idx];
            if (shared_is_dof[0][0][lane_id + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[0][0][lane_id + 1] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[0][0][lane_id + 1] = 0;
        // x- y+
        nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y + 1, tile_ijk.z };
        if (nb_tile_ijk.x >= 0 && nb_tile_ijk.y < _tile_dim.y) {
            int3 nb_voxel_ijk                = { 7, 0, lane_id };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[0][9][lane_id + 1] = _is_dof[nb_idx];
            if (shared_is_dof[0][9][lane_id + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[0][9][lane_id + 1] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[0][9][lane_id + 1] = 0;
        // x- z-
        nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z - 1 };
        if (nb_tile_ijk.x >= 0 && nb_tile_ijk.z >= 0) {
            int3 nb_voxel_ijk                = { 7, lane_id, 7 };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[0][lane_id + 1][0] = _is_dof[nb_idx];
            if (shared_is_dof[0][lane_id + 1][0] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[0][lane_id + 1][0] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[0][lane_id + 1][0] = 0;
    }
    if (warp_id == 1 && lane_id < 8) {
        // y- z+
        int3 nb_tile_ijk = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z + 1 };
        if (nb_tile_ijk.y >= 0 && nb_tile_ijk.z < _tile_dim.z) {
            int3 nb_voxel_ijk                = { lane_id, 7, 0 };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[lane_id + 1][0][9] = _is_dof[nb_idx];
            if (shared_is_dof[lane_id + 1][0][9] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[lane_id + 1][0][9] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[lane_id + 1][0][9] = 0;
        // y+ z+
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z + 1 };
        if (nb_tile_ijk.y < _tile_dim.y && nb_tile_ijk.z < _tile_dim.z) {
            int3 nb_voxel_ijk                = { lane_id, 0, 0 };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[lane_id + 1][9][9] = _is_dof[nb_idx];
            if (shared_is_dof[lane_id + 1][9][9] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[lane_id + 1][9][9] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[lane_id + 1][9][9] = 0;
        // x- z+
        nb_tile_ijk = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z + 1 };
        if (nb_tile_ijk.x >= 0 && nb_tile_ijk.z < _tile_dim.z) {
            int3 nb_voxel_ijk                = { 7, lane_id, 0 };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[0][lane_id + 1][9] = _is_dof[nb_idx];
            if (shared_is_dof[0][lane_id + 1][9] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[0][lane_id + 1][9] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[0][lane_id + 1][9] = 0;
    }
    if (warp_id == 2 && lane_id < 8) {
        // x+ y-
        int3 nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y - 1, tile_ijk.z };
        if (nb_tile_ijk.x < _tile_dim.x && nb_tile_ijk.y >= 0) {
            int3 nb_voxel_ijk                = { 0, 7, lane_id };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[9][0][lane_id + 1] = _is_dof[nb_idx];
            if (shared_is_dof[9][0][lane_id + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[9][0][lane_id + 1] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[9][0][lane_id + 1] = 0;
        // x+ y+
        nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y + 1, tile_ijk.z };
        if (nb_tile_ijk.x < _tile_dim.x && nb_tile_ijk.y < _tile_dim.y) {
            int3 nb_voxel_ijk                = { 0, 0, lane_id };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[9][9][lane_id + 1] = _is_dof[nb_idx];
            if (shared_is_dof[9][9][lane_id + 1] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[9][9][lane_id + 1] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[9][9][lane_id + 1] = 0;
        // x+ z+
        nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z + 1 };
        if (nb_tile_ijk.x < _tile_dim.x && nb_tile_ijk.z < _tile_dim.z) {
            int3 nb_voxel_ijk                = { 0, lane_id, 0 };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[9][lane_id + 1][9] = _is_dof[nb_idx];
            if (shared_is_dof[9][lane_id + 1][9] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[9][lane_id + 1][9] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[9][lane_id + 1][9] = 0;
    }
    if (warp_id == 3 && lane_id < 8) {
        // y- z-
        int3 nb_tile_ijk = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z - 1 };
        if (nb_tile_ijk.y >= 0 && nb_tile_ijk.z >= 0) {
            int3 nb_voxel_ijk                = { lane_id, 7, 7 };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[lane_id + 1][0][0] = _is_dof[nb_idx];
            if (shared_is_dof[lane_id + 1][0][0] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[lane_id + 1][0][0] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[lane_id + 1][0][0] = 0;
        // y+ z-
        nb_tile_ijk = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z - 1 };
        if (nb_tile_ijk.y < _tile_dim.y && nb_tile_ijk.z >= 0) {
            int3 nb_voxel_ijk                = { lane_id, 0, 7 };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[lane_id + 1][9][0] = _is_dof[nb_idx];
            if (shared_is_dof[lane_id + 1][9][0] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[lane_id + 1][9][0] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[lane_id + 1][9][0] = 0;
        // x+ z-
        nb_tile_ijk = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z - 1 };
        if (nb_tile_ijk.x < _tile_dim.x && nb_tile_ijk.z >= 0) {
            int3 nb_voxel_ijk                = { 0, lane_id, 7 };
            int3 nb_ijk                      = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int nb_idx                       = IjkToIdx(_tile_dim, nb_ijk);
            shared_is_dof[9][lane_id + 1][0] = _is_dof[nb_idx];
            if (shared_is_dof[9][lane_id + 1][0] && (nb_voxel_ijk.x + nb_voxel_ijk.y + nb_voxel_ijk.z) % 2 == 0) {
                int3 coarse_nb_ijk          = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
                int coarse_nb_idx           = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
                shared_x[9][lane_id + 1][0] = _x[nb_idx] + 2.0f * _coarse_x[coarse_nb_idx];
            }
        } else
            shared_is_dof[9][lane_id + 1][0] = 0;
    }
    __syncthreads();
    // phase 1
    for (int i = 0; i < 2; i++) {
        int id         = i * 128 + t_id;
        int a          = id / 32;
        int b          = id % 32;
        int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + !((b / 4 + a) % 2) };
        if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1]) {
            float val = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
            int3 ijk  = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
            int idx   = IjkToIdx(_tile_dim, ijk);
            if (shared_is_dof[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1])
                val -= _a_x[IjkToIdx(_tile_dim, { ijk.x - 1, ijk.y, ijk.z })] * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1])
                val -= _a_x[idx] * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1])
                val -= _a_y[IjkToIdx(_tile_dim, { ijk.x, ijk.y - 1, ijk.z })] * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1])
                val -= _a_y[idx] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { ijk.x, ijk.y, ijk.z - 1 })] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2])
                val -= _a_z[idx] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];
            shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = val / _a_diag[idx];
        }
    }
    if (warp_id == 0) {
        // x-
        int3 nb_voxel_ijk = { 7, lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        if (shared_is_dof[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
            int3 nb_tile_ijk   = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
            int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
            int nb_idx         = IjkToIdx(_tile_dim, nb_ijk);
            int coarse_nb_idx  = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
            float val          = _b[nb_idx];
            if (_is_dof[nb_idx - 64])
                val -= _a_x[nb_idx - 64] * (_x[nb_idx - 64] + 2.0f * _coarse_x[coarse_nb_idx]);
            if (shared_is_dof[1][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1])
                val -= _a_x[nb_idx] * shared_x[1][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1];
            if (shared_is_dof[0][nb_voxel_ijk.y][nb_voxel_ijk.z + 1]) {
                val -= _a_y[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y - 1, nb_ijk.z })] * shared_x[0][nb_voxel_ijk.y][nb_voxel_ijk.z + 1];
            }
            if (shared_is_dof[0][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1])
                val -= _a_y[nb_idx] * shared_x[0][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1];
            if (shared_is_dof[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y, nb_ijk.z - 1 })] * shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z];
            if (shared_is_dof[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2])
                val -= _a_z[nb_idx] * shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2];
            shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = val / _a_diag[nb_idx];
        }
        // x+
        nb_voxel_ijk = { 0, lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        if (shared_is_dof[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
            int3 nb_tile_ijk   = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
            int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
            int nb_idx         = IjkToIdx(_tile_dim, nb_ijk);
            int coarse_nb_idx  = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
            float val          = _b[nb_idx];
            if (shared_is_dof[8][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1])
                val -= _a_x[IjkToIdx(_tile_dim, { nb_ijk.x - 1, nb_ijk.y, nb_ijk.z })] * shared_x[8][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1];
            if (_is_dof[nb_idx + 64])
                val -= _a_x[nb_idx] * (_x[nb_idx + 64] + 2.0f * _coarse_x[coarse_nb_idx]);
            if (shared_is_dof[9][nb_voxel_ijk.y][nb_voxel_ijk.z + 1])
                val -= _a_y[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y - 1, nb_ijk.z })] * shared_x[9][nb_voxel_ijk.y][nb_voxel_ijk.z + 1];
            if (shared_is_dof[9][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1])
                val -= _a_y[nb_idx] * shared_x[9][nb_voxel_ijk.y + 2][nb_voxel_ijk.z + 1];
            if (shared_is_dof[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y, nb_ijk.z - 1 })] * shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z];
            if (shared_is_dof[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2])
                val -= _a_z[nb_idx] * shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 2];
            shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = val / _a_diag[nb_idx];
        }
    }
    if (warp_id == 1) {
        // y-
        int3 nb_voxel_ijk = { lane_id / 4, 7, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
        if (shared_is_dof[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1]) {
            int3 nb_tile_ijk   = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
            int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
            int nb_idx         = IjkToIdx(_tile_dim, nb_ijk);
            int coarse_nb_idx  = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
            float val          = _b[nb_idx];
            if (shared_is_dof[nb_voxel_ijk.x][0][nb_voxel_ijk.z + 1])
                val -= _a_x[IjkToIdx(_tile_dim, { nb_ijk.x - 1, nb_ijk.y, nb_ijk.z })] * shared_x[nb_voxel_ijk.x][0][nb_voxel_ijk.z + 1];
            if (shared_is_dof[nb_voxel_ijk.x + 2][0][nb_voxel_ijk.z + 1])
                val -= _a_x[nb_idx] * shared_x[nb_voxel_ijk.x + 2][0][nb_voxel_ijk.z + 1];
            if (_is_dof[nb_idx - 8])
                val -= _a_y[nb_idx - 8] * (_x[nb_idx - 8] + 2.0f * _coarse_x[coarse_nb_idx]);
            if (shared_is_dof[nb_voxel_ijk.x + 1][1][nb_voxel_ijk.z + 1])
                val -= _a_y[nb_idx] * shared_x[nb_voxel_ijk.x + 1][1][nb_voxel_ijk.z + 1];
            if (shared_is_dof[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y, nb_ijk.z - 1 })] * shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z];
            if (shared_is_dof[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 2])
                val -= _a_z[nb_idx] * shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 2];
            shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = val / _a_diag[nb_idx];
        }
        // y+
        nb_voxel_ijk = { lane_id / 4, 0, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
        if (shared_is_dof[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1]) {
            int3 nb_tile_ijk   = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
            int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
            int nb_idx         = IjkToIdx(_tile_dim, nb_ijk);
            int coarse_nb_idx  = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
            float val          = _b[nb_idx];
            if (shared_is_dof[nb_voxel_ijk.x][9][nb_voxel_ijk.z + 1])
                val -= _a_x[IjkToIdx(_tile_dim, { nb_ijk.x - 1, nb_ijk.y, nb_ijk.z })] * shared_x[nb_voxel_ijk.x][9][nb_voxel_ijk.z + 1];
            if (shared_is_dof[nb_voxel_ijk.x + 2][9][nb_voxel_ijk.z + 1])
                val -= _a_x[nb_idx] * shared_x[nb_voxel_ijk.x + 2][9][nb_voxel_ijk.z + 1];
            if (shared_is_dof[nb_voxel_ijk.x + 1][8][nb_voxel_ijk.z + 1])
                val -= _a_y[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y - 1, nb_ijk.z })] * shared_x[nb_voxel_ijk.x + 1][8][nb_voxel_ijk.z + 1];
            if (_is_dof[nb_idx + 8])
                val -= _a_y[nb_idx] * (_x[nb_idx + 8] + 2.0f * _coarse_x[coarse_nb_idx]);
            if (shared_is_dof[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y, nb_ijk.z - 1 })] * shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z];
            if (shared_is_dof[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 2])
                val -= _a_z[nb_idx] * shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 2];
            shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = val / _a_diag[nb_idx];
        }
    }
    if (warp_id == 2) {
        // z-
        int3 nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2, 7 };
        if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0]) {
            int3 nb_tile_ijk   = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
            int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
            int nb_idx         = IjkToIdx(_tile_dim, nb_ijk);
            int coarse_nb_idx  = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
            float val          = _b[nb_idx];
            if (shared_is_dof[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][0])
                val -= _a_x[IjkToIdx(_tile_dim, { nb_ijk.x - 1, nb_ijk.y, nb_ijk.z })] * shared_x[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][0];
            if (shared_is_dof[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][0])
                val -= _a_x[nb_idx] * shared_x[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][0];
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][0])
                val -= _a_y[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y - 1, nb_ijk.z })] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][0];
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][0])
                val -= _a_y[nb_idx] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][0];
            if (_is_dof[nb_idx - 1])
                val -= _a_z[nb_idx - 1] * (_x[nb_idx - 1] + 2.0f * _coarse_x[coarse_nb_idx]);
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][1])
                val -= _a_z[nb_idx] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][1];
            shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = val / _a_diag[nb_idx];
        }
        // z+
        nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2), 0 };
        if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9]) {
            int3 nb_tile_ijk   = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
            int3 nb_ijk        = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
            int3 coarse_nb_ijk = { nb_ijk.x / 2, nb_ijk.y / 2, nb_ijk.z / 2 };
            int nb_idx         = IjkToIdx(_tile_dim, nb_ijk);
            int coarse_nb_idx  = IjkToIdx(coarse_tile_dim, coarse_nb_ijk);
            float val          = _b[nb_idx];
            if (shared_is_dof[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][9])
                val -= _a_x[IjkToIdx(_tile_dim, { nb_ijk.x - 1, nb_ijk.y, nb_ijk.z })] * shared_x[nb_voxel_ijk.x][nb_voxel_ijk.y + 1][9];
            if (shared_is_dof[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][9])
                val -= _a_x[nb_idx] * shared_x[nb_voxel_ijk.x + 2][nb_voxel_ijk.y + 1][9];
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][9])
                val -= _a_y[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y - 1, nb_ijk.z })] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y][9];
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][9])
                val -= _a_y[nb_idx] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 2][9];
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][8])
                val -= _a_z[IjkToIdx(_tile_dim, { nb_ijk.x, nb_ijk.y, nb_ijk.z - 1 })] * shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][8];
            if (_is_dof[nb_idx + 1])
                val -= _a_z[nb_idx] * (_x[nb_idx + 1] + 2.0f * _coarse_x[coarse_nb_idx]);
            shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = val / _a_diag[nb_idx];
        }
    }
    __syncthreads();
    // phase 0
    for (int i = 0; i < 2; i++) {
        int id         = i * 128 + t_id;
        int a          = id / 32;
        int b          = id % 32;
        int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + (b / 4 + a) % 2 };
        if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1]) {
            float val = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
            int3 ijk  = { tile_ijk.x * 8 + voxel_ijk.x, tile_ijk.y * 8 + voxel_ijk.y, tile_ijk.z * 8 + voxel_ijk.z };
            int idx   = IjkToIdx(_tile_dim, ijk);
            if (shared_is_dof[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1])
                val -= _a_x[IjkToIdx(_tile_dim, { ijk.x - 1, ijk.y, ijk.z })] * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1])
                val -= _a_x[idx] * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1])
                val -= _a_y[IjkToIdx(_tile_dim, { ijk.x, ijk.y - 1, ijk.z })] * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1])
                val -= _a_y[idx] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z])
                val -= _a_z[IjkToIdx(_tile_dim, { ijk.x, ijk.y, ijk.z - 1 })] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2])
                val -= _a_z[idx] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];
            shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = val / _a_diag[idx];
        }
    }
    __syncthreads();
    // write back x
    for (int i = 0; i < 4; i++) {
        int voxel_idx  = i * 128 + t_id;
        int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
        int idx        = tile_idx * 512 + voxel_idx;
        _dst_x[idx]    = shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1];
    }
    // dot
    if (_do_dot) {
        float mul[4];
        for (int i = 0; i < 4; i++) {
            int voxel_idx  = i * 128 + t_id;
            int3 voxel_ijk = VoxelIdxToIjk(voxel_idx);
            mul[i]         = shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] * shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
        }
        using BlockReduce = cub::BlockReduce<float, 128>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        float sum = BlockReduce(temp_storage).Sum(mul);
        if (t_id == 0)
            _dot_buffer[tile_idx] = sum;
    }
}
void TrimPoisson::ProlongGaussSeidelDotAsync(const std::shared_ptr<DHMemory<float>> _coarse_x, std::shared_ptr<DHMemory<float>> _dot_buffer, bool _do_dot, cudaStream_t _stream)
{
    int tile_num             = Prod(tile_dim_);
    float* dst_x             = buffer_->dev_ptr_;
    const float* x           = x_->dev_ptr_;
    const float* b           = b_->dev_ptr_;
    const float* coarse_x    = _coarse_x->dev_ptr_;
    const char* is_dof       = is_dof_->dev_ptr_;
    const float* a_diag      = a_diag_->dev_ptr_;
    const float* a_x         = a_x_->dev_ptr_;
    const float* a_y         = a_y_->dev_ptr_;
    const float* a_z         = a_z_->dev_ptr_;
    const char* tile_trimmed = tile_trimmed_->dev_ptr_;
    float default_inv_a_diag = 1.0f / default_a_diag_;
    float* dot_buffer        = _dot_buffer->dev_ptr_;
    ProlongGaussSeidelDotTrivialKernel<<<tile_num, 128, 0, _stream>>>(dst_x, tile_dim_, x, coarse_x, tile_trimmed, default_inv_a_diag, default_a_off_diag_, b, dot_buffer, _do_dot);
    ProlongGaussSeidelDotNonTrivialKernel<<<tile_num, 128, 0, _stream>>>(dst_x, tile_dim_, x, coarse_x, tile_trimmed, is_dof, a_diag, a_x, a_y, a_z, b, dot_buffer, _do_dot);
    x_.swap(buffer_);
}

__global__ void CoarsestGaussSeidelKernel(float* _x, int3 _tile_dim, const char* _is_dof, const float* _a_diag, const float* _a_x, const float* _a_y, const float* _a_z, const float* _b, int _iter_num)
{
    int tile_idx  = blockIdx.x;
    int3 tile_ijk = TileIdxToIjk(_tile_dim, tile_idx);
    int t_id      = threadIdx.x;
    __shared__ float shared_x[10][10][10];
    __shared__ char shared_is_dof[10][10][10];
    __shared__ float shared_inv_a_diag[8][8][8];
    __shared__ float shared_a_x[9][8][8];
    __shared__ float shared_a_y[8][9][8];
    __shared__ float shared_a_z[8][8][9];
    __shared__ float shared_b[8][8][8];
    // load current tile
    for (int i = 0; i < 4; i++) {
        int voxel_idx                                                    = i * 128 + t_id;
        int3 voxel_ijk                                                   = VoxelIdxToIjk(voxel_idx);
        int idx                                                          = tile_idx * 512 + voxel_idx;
        shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _is_dof[idx];
        if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1])
            shared_inv_a_diag[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] = 1.0f / _a_diag[idx];
        else
            shared_inv_a_diag[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] = 0.0f;
        shared_a_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z] = _a_x[idx];
        shared_a_y[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z] = _a_y[idx];
        shared_a_z[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z + 1] = _a_z[idx];
        shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z]       = _b[idx];
        _x[idx]                                               = 0.0f;
    }
    // load neighbor tiles
    int warp_id = t_id / 32;
    int lane_id = t_id % 32;
    if (warp_id < 2) {
        int a = t_id / 8;
        int b = t_id % 8;
        // x-
        if (tile_ijk.x > 0) {
            int3 nb_tile_ijk               = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
            int3 nb_voxel_ijk              = { 7, a, b };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[0][a + 1][b + 1] = _is_dof[nb_idx];
            shared_a_x[0][a][b]            = _a_x[nb_idx];
        } else
            shared_is_dof[0][a + 1][b + 1] = 0;
        // y-
        if (tile_ijk.y > 0) {
            int3 nb_tile_ijk               = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
            int3 nb_voxel_ijk              = { a, 7, b };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[a + 1][0][b + 1] = _is_dof[nb_idx];
            shared_a_y[a][0][b]            = _a_y[nb_idx];
        } else
            shared_is_dof[a + 1][0][b + 1] = 0;
        // z-
        if (tile_ijk.z > 0) {
            int3 nb_tile_ijk               = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
            int3 nb_voxel_ijk              = { a, b, 7 };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[a + 1][b + 1][0] = _is_dof[nb_idx];
            shared_a_z[a][b][0]            = _a_z[nb_idx];
        } else
            shared_is_dof[a + 1][b + 1][0] = 0;
    } else {
        int group_id = t_id - 64;
        int a        = group_id / 8;
        int b        = group_id % 8;
        // x+
        if (tile_ijk.x < _tile_dim.x - 1) {
            int3 nb_tile_ijk               = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
            int3 nb_voxel_ijk              = { 0, a, b };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[9][a + 1][b + 1] = _is_dof[nb_idx];
        } else
            shared_is_dof[9][a + 1][b + 1] = 0;
        // y+
        if (tile_ijk.y < _tile_dim.y - 1) {
            int3 nb_tile_ijk               = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
            int3 nb_voxel_ijk              = { a, 0, b };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[a + 1][9][b + 1] = _is_dof[nb_idx];
        } else
            shared_is_dof[a + 1][9][b + 1] = 0;
        // z+
        if (tile_ijk.z < _tile_dim.z - 1) {
            int3 nb_tile_ijk               = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
            int3 nb_voxel_ijk              = { a, b, 0 };
            int nb_tile_idx                = TileIjkToIdx(_tile_dim, nb_tile_ijk);
            int nb_voxel_idx               = VoxelIjkToIdx(nb_voxel_ijk);
            int nb_idx                     = nb_tile_idx * 512 + nb_voxel_idx;
            shared_is_dof[a + 1][b + 1][9] = _is_dof[nb_idx];
        } else
            shared_is_dof[a + 1][b + 1][9] = 0;
    }
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    grid.sync();

    int half_iter_num = _iter_num / 2;
    for (int iter = 0; iter < half_iter_num; iter++) {
        // phase 0
        for (int i = 0; i < 2; i++) {
            int id         = i * 128 + t_id;
            int a          = id / 32;
            int b          = id % 32;
            int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + !((b / 4 + a) % 2) };
            int voxel_idx  = VoxelIjkToIdx(voxel_ijk);
            int idx        = tile_idx * 512 + voxel_idx;
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1])
                shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _x[idx];
            else
                shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 0) {
            // x-
            int3 nb_voxel_ijk = { 7, lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
            if (shared_is_dof[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[nb_idx];

            } else
                shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = 0.0f;
            // x+
            nb_voxel_ijk = { 0, lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
            if (shared_is_dof[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[nb_idx];

            } else
                shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 1) {
            // y-
            int3 nb_voxel_ijk = { lane_id / 4, 7, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = 0.0f;
            // y+
            nb_voxel_ijk = { lane_id / 4, 0, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
            if (shared_is_dof[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 2) {
            // z-
            int3 nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2, 7 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = 0.0f;
            // z+
            nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2), 0 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < 2; i++) {
            int id         = i * 128 + t_id;
            int a          = id / 32;
            int b          = id % 32;
            int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + (b / 4 + a) % 2 };
            float val      = 0.0f;
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1]) {
                val = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
                val -= shared_a_x[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
                val -= shared_a_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
                val -= shared_a_y[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
                val -= shared_a_y[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
                val -= shared_a_z[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
                val -= shared_a_z[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z + 1] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];
                val *= shared_inv_a_diag[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
            }
            int voxel_idx = VoxelIjkToIdx(voxel_ijk);
            int idx       = tile_idx * 512 + voxel_idx;
            _x[idx]       = val;
        }
        grid.sync();

        // phase 1
        for (int i = 0; i < 2; i++) {
            int id         = i * 128 + t_id;
            int a          = id / 32;
            int b          = id % 32;
            int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + (b / 4 + a) % 2 };
            int voxel_idx  = VoxelIjkToIdx(voxel_ijk);
            int idx        = tile_idx * 512 + voxel_idx;
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1])
                shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _x[idx];
            else
                shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 0) {
            // x-
            int3 nb_voxel_ijk = { 7, lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
            if (shared_is_dof[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[nb_idx];

            } else
                shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = 0.0f;
            // x+
            nb_voxel_ijk = { 0, lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
            if (shared_is_dof[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[nb_idx];

            } else
                shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 1) {
            // y-
            int3 nb_voxel_ijk = { lane_id / 4, 7, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
            if (shared_is_dof[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = 0.0f;
            // y+
            nb_voxel_ijk = { lane_id / 4, 0, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 2) {
            // z-
            int3 nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2), 7 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = 0.0f;
            // z+
            nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2, 0 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < 2; i++) {
            int id         = i * 128 + t_id;
            int a          = id / 32;
            int b          = id % 32;
            int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + !((b / 4 + a) % 2) };
            float val      = 0.0f;
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1]) {
                val = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
                val -= shared_a_x[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
                val -= shared_a_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
                val -= shared_a_y[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
                val -= shared_a_y[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
                val -= shared_a_z[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
                val -= shared_a_z[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z + 1] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];

                val *= shared_inv_a_diag[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
            }
            int voxel_idx = VoxelIjkToIdx(voxel_ijk);
            int idx       = tile_idx * 512 + voxel_idx;
            _x[idx]       = val;
        }
        grid.sync();
    }

    for (int iter = 0; iter < half_iter_num; iter++) {
        // phase 1
        for (int i = 0; i < 2; i++) {
            int id         = i * 128 + t_id;
            int a          = id / 32;
            int b          = id % 32;
            int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + (b / 4 + a) % 2 };
            int voxel_idx  = VoxelIjkToIdx(voxel_ijk);
            int idx        = tile_idx * 512 + voxel_idx;
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1])
                shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _x[idx];
            else
                shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 0) {
            // x-
            int3 nb_voxel_ijk = { 7, lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
            if (shared_is_dof[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[nb_idx];

            } else
                shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = 0.0f;
            // x+
            nb_voxel_ijk = { 0, lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
            if (shared_is_dof[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[nb_idx];

            } else
                shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 1) {
            // y-
            int3 nb_voxel_ijk = { lane_id / 4, 7, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
            if (shared_is_dof[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = 0.0f;
            // y+
            nb_voxel_ijk = { lane_id / 4, 0, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 2) {
            // z-
            int3 nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2), 7 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = 0.0f;
            // z+
            nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2, 0 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < 2; i++) {
            int id         = i * 128 + t_id;
            int a          = id / 32;
            int b          = id % 32;
            int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + !((b / 4 + a) % 2) };
            float val      = 0.0f;
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1]) {
                val = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
                val -= shared_a_x[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
                val -= shared_a_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
                val -= shared_a_y[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
                val -= shared_a_y[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
                val -= shared_a_z[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
                val -= shared_a_z[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z + 1] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];

                val *= shared_inv_a_diag[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
            }
            int voxel_idx = VoxelIjkToIdx(voxel_ijk);
            int idx       = tile_idx * 512 + voxel_idx;
            _x[idx]       = val;
        }
        grid.sync();
        // phase 0
        for (int i = 0; i < 2; i++) {
            int id         = i * 128 + t_id;
            int a          = id / 32;
            int b          = id % 32;
            int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + !((b / 4 + a) % 2) };
            int voxel_idx  = VoxelIjkToIdx(voxel_ijk);
            int idx        = tile_idx * 512 + voxel_idx;
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1])
                shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = _x[idx];
            else
                shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 0) {
            // x-
            int3 nb_voxel_ijk = { 7, lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
            if (shared_is_dof[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x - 1, tile_ijk.y, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[nb_idx];

            } else
                shared_x[0][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = 0.0f;
            // x+
            nb_voxel_ijk = { 0, lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
            if (shared_is_dof[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x + 1, tile_ijk.y, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = _x[nb_idx];

            } else
                shared_x[9][nb_voxel_ijk.y + 1][nb_voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 1) {
            // y-
            int3 nb_voxel_ijk = { lane_id / 4, 7, 2 * (lane_id % 4) + (lane_id / 4) % 2 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y - 1, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][0][nb_voxel_ijk.z + 1] = 0.0f;
            // y+
            nb_voxel_ijk = { lane_id / 4, 0, 2 * (lane_id % 4) + !((lane_id / 4) % 2) };
            if (shared_is_dof[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y + 1, tile_ijk.z };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][9][nb_voxel_ijk.z + 1] = 0.0f;
        }
        if (warp_id == 2) {
            // z-
            int3 nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + (lane_id / 4) % 2, 7 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y, tile_ijk.z - 1 };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][0] = 0.0f;
            // z+
            nb_voxel_ijk = { lane_id / 4, 2 * (lane_id % 4) + !((lane_id / 4) % 2), 0 };
            if (shared_is_dof[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9]) {
                int3 nb_tile_ijk                                    = { tile_ijk.x, tile_ijk.y, tile_ijk.z + 1 };
                int3 nb_ijk                                         = { nb_tile_ijk.x * 8 + nb_voxel_ijk.x, nb_tile_ijk.y * 8 + nb_voxel_ijk.y, nb_tile_ijk.z * 8 + nb_voxel_ijk.z };
                int nb_idx                                          = IjkToIdx(_tile_dim, nb_ijk);
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = _x[nb_idx];
            } else
                shared_x[nb_voxel_ijk.x + 1][nb_voxel_ijk.y + 1][9] = 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < 2; i++) {
            int id         = i * 128 + t_id;
            int a          = id / 32;
            int b          = id % 32;
            int3 voxel_ijk = { a, b / 4, 2 * (b % 4) + (b / 4 + a) % 2 };
            float val      = 0.0f;
            if (shared_is_dof[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 1]) {
                val = shared_b[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
                val -= shared_a_x[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z + 1];
                val -= shared_a_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 2][voxel_ijk.y + 1][voxel_ijk.z + 1];
                val -= shared_a_y[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y][voxel_ijk.z + 1];
                val -= shared_a_y[voxel_ijk.x][voxel_ijk.y + 1][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 2][voxel_ijk.z + 1];
                val -= shared_a_z[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z];
                val -= shared_a_z[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z + 1] * shared_x[voxel_ijk.x + 1][voxel_ijk.y + 1][voxel_ijk.z + 2];
                val *= shared_inv_a_diag[voxel_ijk.x][voxel_ijk.y][voxel_ijk.z];
            }
            int voxel_idx = VoxelIjkToIdx(voxel_ijk);
            int idx       = tile_idx * 512 + voxel_idx;
            _x[idx]       = val;
        }
        grid.sync();
    }
}

void TrimPoisson::CoarsestGaussSeidelAsync(int _num_iter, cudaStream_t _stream)
{
    dim3 blocks_per_grid(Prod(tile_dim_));
    dim3 threads_per_block(128);
    float* x            = x_->dev_ptr_;
    const char* is_dof  = is_dof_->dev_ptr_;
    const float* a_diag = a_diag_->dev_ptr_;
    const float* a_x    = a_x_->dev_ptr_;
    const float* a_y    = a_y_->dev_ptr_;
    const float* a_z    = a_z_->dev_ptr_;
    const float* b      = b_->dev_ptr_;
    void* args[]        = { (void*)&x, (void*)&tile_dim_, (void*)&is_dof, (void*)&a_diag, (void*)&a_x, (void*)&a_y, (void*)&a_z, (void*)&b, (void*)&_num_iter };
    cudaLaunchCooperativeKernel((void*)CoarsestGaussSeidelKernel, blocks_per_grid, threads_per_block, args, 0, _stream);
}
}