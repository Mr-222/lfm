#include "data_io.h"
#include "lfm.h"
#include "lfm_util.h"
#include "timer.h"
#include <cub/cub.cuh>

namespace lfm {
LFM::LFM(int3 _tile_dim, int _reinit_every, int _num_smoke)
{
    Alloc(_tile_dim, _reinit_every, _num_smoke);
}

void LFM::Alloc(int3 _tile_dim, int _reinit_every, int _num_smoke)
{
    tile_dim_     = _tile_dim;
    reinit_every_ = _reinit_every;

    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    int voxel_num   = Prod(tile_dim_) * 512;
    int x_voxel_num = Prod(x_tile_dim) * 512;
    int y_voxel_num = Prod(y_tile_dim) * 512;
    int z_voxel_num = Prod(z_tile_dim) * 512;

    // boundary
    is_bc_x_  = std::make_shared<DHMemory<uint8_t>>(x_voxel_num);
    is_bc_y_  = std::make_shared<DHMemory<uint8_t>>(y_voxel_num);
    is_bc_z_  = std::make_shared<DHMemory<uint8_t>>(z_voxel_num);
    bc_val_x_ = std::make_shared<DHMemory<float>>(x_voxel_num);
    bc_val_y_ = std::make_shared<DHMemory<float>>(y_voxel_num);
    bc_val_z_ = std::make_shared<DHMemory<float>>(z_voxel_num);

    // backward flow map
    T_x_   = std::make_shared<DHMemory<float3>>(x_voxel_num);
    T_y_   = std::make_shared<DHMemory<float3>>(y_voxel_num);
    T_z_   = std::make_shared<DHMemory<float3>>(z_voxel_num);
    psi_x_ = std::make_shared<DHMemory<float3>>(x_voxel_num);
    psi_y_ = std::make_shared<DHMemory<float3>>(y_voxel_num);
    psi_z_ = std::make_shared<DHMemory<float3>>(z_voxel_num);

    // forward flow map
    F_x_   = std::make_shared<DHMemory<float3>>(x_voxel_num);
    F_y_   = std::make_shared<DHMemory<float3>>(y_voxel_num);
    F_z_   = std::make_shared<DHMemory<float3>>(z_voxel_num);
    phi_x_ = std::make_shared<DHMemory<float3>>(x_voxel_num);
    phi_y_ = std::make_shared<DHMemory<float3>>(y_voxel_num);
    phi_z_ = std::make_shared<DHMemory<float3>>(z_voxel_num);

    // velocity storage
    u_        = std::make_shared<DHMemory<float3>>(voxel_num);
    u_x_      = std::make_shared<DHMemory<float>>(x_voxel_num);
    u_y_      = std::make_shared<DHMemory<float>>(y_voxel_num);
    u_z_      = std::make_shared<DHMemory<float>>(z_voxel_num);
    init_u_x_ = std::make_shared<DHMemory<float>>(x_voxel_num);
    init_u_y_ = std::make_shared<DHMemory<float>>(y_voxel_num);
    init_u_z_ = std::make_shared<DHMemory<float>>(z_voxel_num);
    tmp_u_x_  = std::make_shared<DHMemory<float>>(x_voxel_num);
    tmp_u_y_  = std::make_shared<DHMemory<float>>(y_voxel_num);
    tmp_u_z_  = std::make_shared<DHMemory<float>>(z_voxel_num);
    err_u_x_  = std::make_shared<DHMemory<float>>(x_voxel_num);
    err_u_y_  = std::make_shared<DHMemory<float>>(y_voxel_num);
    err_u_z_  = std::make_shared<DHMemory<float>>(z_voxel_num);

    mid_u_x_.resize(reinit_every_);
    mid_u_y_.resize(reinit_every_);
    mid_u_z_.resize(reinit_every_);
    for (int i = 0; i < reinit_every_; i++) {
        mid_u_x_[i] = std::make_shared<DHMemory<float>>(x_voxel_num);
        mid_u_y_[i] = std::make_shared<DHMemory<float>>(y_voxel_num);
        mid_u_z_[i] = std::make_shared<DHMemory<float>>(z_voxel_num);
    }

    // vorcity
    vor_norm_ = std::make_shared<DHMemory<float>>(voxel_num);

    // solver
    int min_dim   = tile_dim_.x;
    min_dim       = min_dim > tile_dim_.y ? tile_dim_.y : min_dim;
    min_dim       = min_dim > tile_dim_.z ? tile_dim_.z : min_dim;
    int level_num = (int)log2(min_dim) + 1;
    amgpcg_.Alloc(_tile_dim, level_num);

    // smoke
    num_smoke_ = _num_smoke;
    if (num_smoke_ > 0) {
        smoke_.resize(num_smoke_);
        init_smoke_.resize(num_smoke_);
        prev_smoke_.resize(num_smoke_);
        smoke_np_.resize(num_smoke_);
        for (int i = 0; i < num_smoke_; i++) {
            smoke_[i]      = std::make_shared<DHMemory<float>>(voxel_num);
            init_smoke_[i] = std::make_shared<DHMemory<float>>(voxel_num);
            prev_smoke_[i] = std::make_shared<DHMemory<float>>(voxel_num);
            smoke_np_[i]   = std::make_shared<DHMemory<float>>(voxel_num);
        }
        err_smoke_ = std::make_shared<DHMemory<float>>(voxel_num);
        tmp_smoke_ = std::make_shared<DHMemory<float>>(voxel_num);
        psi_c_     = std::make_shared<DHMemory<float3>>(voxel_num);
        phi_c_     = std::make_shared<DHMemory<float3>>(voxel_num);
    }
}

void LFM::AdvanceAsync(float _dt, cudaStream_t _stream)
{
    float mid_dt;
    std::shared_ptr<DHMemory<float>> last_proj_u_x;
    std::shared_ptr<DHMemory<float>> last_proj_u_y;
    std::shared_ptr<DHMemory<float>> last_proj_u_z;
    std::shared_ptr<DHMemory<float>> src_u_x;
    std::shared_ptr<DHMemory<float>> src_u_y;
    std::shared_ptr<DHMemory<float>> src_u_z;
    if (step_ % reinit_every_ == 0) {
        mid_dt        = 0.5f * _dt;
        last_proj_u_x = init_u_x_;
        last_proj_u_y = init_u_y_;
        last_proj_u_z = init_u_z_;
        src_u_x       = init_u_x_;
        src_u_y       = init_u_y_;
        src_u_z       = init_u_z_;
    } else if (step_ % reinit_every_ == 1) {
        mid_dt        = _dt;
        last_proj_u_x = mid_u_x_[0];
        last_proj_u_y = mid_u_y_[0];
        last_proj_u_z = mid_u_z_[0];
        src_u_x       = mid_u_x_[0];
        src_u_y       = mid_u_y_[0];
        src_u_z       = mid_u_z_[0];
    } else {
        mid_dt        = 2.0f * _dt;
        last_proj_u_x = mid_u_x_[step_ % reinit_every_ - 1];
        last_proj_u_y = mid_u_y_[step_ % reinit_every_ - 1];
        last_proj_u_z = mid_u_z_[step_ % reinit_every_ - 1];
        src_u_x       = mid_u_x_[step_ % reinit_every_ - 2];
        src_u_y       = mid_u_y_[step_ % reinit_every_ - 2];
        src_u_z       = mid_u_z_[step_ % reinit_every_ - 2];
    }

    AdvectN2XAsync(*tmp_u_x_, tile_dim_, *src_u_x, *last_proj_u_x, *last_proj_u_y, *last_proj_u_z, dx_, mid_dt, _stream);
    AdvectN2YAsync(*tmp_u_y_, tile_dim_, *src_u_y, *last_proj_u_x, *last_proj_u_y, *last_proj_u_z, dx_, mid_dt, _stream);
    AdvectN2ZAsync(*tmp_u_z_, tile_dim_, *src_u_z, *last_proj_u_x, *last_proj_u_y, *last_proj_u_z, dx_, mid_dt, _stream);
    SetInletAsync(*bc_val_x_, *bc_val_y_, tile_dim_, inlet_norm_, inlet_angle_, _stream);
    ProjectAsync(_stream);
    mid_u_x_[step_ % reinit_every_].swap(tmp_u_x_);
    mid_u_y_[step_ % reinit_every_].swap(tmp_u_y_);
    mid_u_z_[step_ % reinit_every_].swap(tmp_u_z_);

    step_++;
}

void LFM::ReinitAsync(float _dt, cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    ResetForwardFlowMapAsync(_stream);
    ResetBackwardFlowMapAsync(_stream);
    for (int i = reinit_every_ - 1; i >= 0; i--) {
        RKAxisAsync(rk_order_, *psi_x_, *T_x_, tile_dim_, x_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, _dt, _stream);
        RKAxisAsync(rk_order_, *psi_y_, *T_y_, tile_dim_, y_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, _dt, _stream);
        RKAxisAsync(rk_order_, *psi_z_, *T_z_, tile_dim_, z_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, _dt, _stream);
    }
    for (int i = 0; i < reinit_every_; i++) {
        RKAxisAsync(rk_order_, *phi_x_, *F_x_, tile_dim_, x_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -_dt, _stream);
        RKAxisAsync(rk_order_, *phi_y_, *F_y_, tile_dim_, y_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -_dt, _stream);
        RKAxisAsync(rk_order_, *phi_z_, *F_z_, tile_dim_, z_tile_dim, *mid_u_x_[i], *mid_u_y_[i], *mid_u_z_[i], grid_origin_, dx_, -_dt, _stream);
    }
    PullbackAxisAsync(*u_x_, tile_dim_, x_tile_dim, *init_u_x_, *init_u_y_, *init_u_z_, *psi_x_, *T_x_, grid_origin_, dx_, _stream);
    PullbackAxisAsync(*u_y_, tile_dim_, y_tile_dim, *init_u_x_, *init_u_y_, *init_u_z_, *psi_y_, *T_y_, grid_origin_, dx_, _stream);
    PullbackAxisAsync(*u_z_, tile_dim_, z_tile_dim, *init_u_x_, *init_u_y_, *init_u_z_, *psi_z_, *T_z_, grid_origin_, dx_, _stream);
    PullbackAxisAsync(*err_u_x_, tile_dim_, x_tile_dim, *u_x_, *u_y_, *u_z_, *phi_x_, *F_x_, grid_origin_, dx_, _stream);
    PullbackAxisAsync(*err_u_y_, tile_dim_, y_tile_dim, *u_x_, *u_y_, *u_z_, *phi_y_, *F_y_, grid_origin_, dx_, _stream);
    PullbackAxisAsync(*err_u_z_, tile_dim_, z_tile_dim, *u_x_, *u_y_, *u_z_, *phi_z_, *F_z_, grid_origin_, dx_, _stream);
    AddFieldsAsync(*err_u_x_, x_tile_dim, *err_u_x_, *init_u_x_, -1.0f, _stream);
    AddFieldsAsync(*err_u_y_, y_tile_dim, *err_u_y_, *init_u_y_, -1.0f, _stream);
    AddFieldsAsync(*err_u_z_, z_tile_dim, *err_u_z_, *init_u_z_, -1.0f, _stream);
    PullbackAxisAsync(*init_u_x_, tile_dim_, x_tile_dim, *err_u_x_, *err_u_y_, *err_u_z_, *psi_x_, *T_x_, grid_origin_, dx_, _stream);
    PullbackAxisAsync(*init_u_y_, tile_dim_, y_tile_dim, *err_u_x_, *err_u_y_, *err_u_z_, *psi_y_, *T_y_, grid_origin_, dx_, _stream);
    PullbackAxisAsync(*init_u_z_, tile_dim_, z_tile_dim, *err_u_x_, *err_u_y_, *err_u_z_, *psi_z_, *T_z_, grid_origin_, dx_, _stream);
    AddFieldsAsync(*tmp_u_x_, x_tile_dim, *u_x_, *init_u_x_, -0.5f, _stream);
    AddFieldsAsync(*tmp_u_y_, y_tile_dim, *u_y_, *init_u_y_, -0.5f, _stream);
    AddFieldsAsync(*tmp_u_z_, z_tile_dim, *u_z_, *init_u_z_, -0.5f, _stream);
    if (use_bfecc_clamp_) {
        int3 x_max_ijk = { tile_dim_.x * 8, tile_dim_.y * 8 - 1, tile_dim_.z * 8 - 1 };
        int3 y_max_ijk = { tile_dim_.x * 8 - 1, tile_dim_.y * 8, tile_dim_.z * 8 - 1 };
        int3 z_max_ijk = { tile_dim_.x * 8 - 1, tile_dim_.y * 8 - 1, tile_dim_.z * 8 };
        BfeccClampAsync(*tmp_u_x_, x_tile_dim, x_max_ijk, *u_x_, _stream);
        BfeccClampAsync(*tmp_u_y_, y_tile_dim, y_max_ijk, *u_y_, _stream);
        BfeccClampAsync(*tmp_u_z_, z_tile_dim, z_max_ijk, *u_z_, _stream);
    }
    ProjectAsync(_stream);
    init_u_x_.swap(tmp_u_x_);
    init_u_y_.swap(tmp_u_y_);
    init_u_z_.swap(tmp_u_z_);
    if (num_smoke_ > 0) {
        GetCentralPsiAsync(*psi_c_, tile_dim_, *psi_x_, *psi_y_, *psi_z_, _stream);
        GetCentralPsiAsync(*phi_c_, tile_dim_, *phi_x_, *phi_y_, *phi_z_, _stream);
        for (int i = 0; i < num_smoke_; i++) {
            PullbackCenterAsync(*smoke_[i], tile_dim_, *init_smoke_[i], *psi_c_, grid_origin_, dx_, _stream);
            PullbackCenterAsync(*err_smoke_, tile_dim_, *smoke_[i], *phi_c_, grid_origin_, dx_, _stream);
            AddFieldsAsync(*err_smoke_, tile_dim_, *err_smoke_, *init_smoke_[i], -1.0f, _stream);
            PullbackCenterAsync(*tmp_smoke_, tile_dim_, *err_smoke_, *psi_c_, grid_origin_, dx_, _stream);
            AddFieldsAsync(*smoke_[i], tile_dim_, *smoke_[i], *tmp_smoke_, -0.5f, _stream);
            DevToDevCpyAsync(init_smoke_[i]->dev_ptr_, smoke_[i]->dev_ptr_, Prod(tile_dim_) * 512, _stream);
        }
    }
}

void LFM::ResetForwardFlowMapAsync(cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    ResetToIdentityXASync(*phi_x_, *F_x_, x_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityYASync(*phi_y_, *F_y_, y_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityZASync(*phi_z_, *F_z_, z_tile_dim, grid_origin_, dx_, _stream);
}

void LFM::ResetBackwardFlowMapAsync(cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };
    ResetToIdentityXASync(*psi_x_, *T_x_, x_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityYASync(*psi_y_, *T_y_, y_tile_dim, grid_origin_, dx_, _stream);
    ResetToIdentityZASync(*psi_z_, *T_z_, z_tile_dim, grid_origin_, dx_, _stream);
}

void LFM::ProjectAsync(cudaStream_t _stream)
{
    int3 x_tile_dim = { tile_dim_.x + 1, tile_dim_.y, tile_dim_.z };
    int3 y_tile_dim = { tile_dim_.x, tile_dim_.y + 1, tile_dim_.z };
    int3 z_tile_dim = { tile_dim_.x, tile_dim_.y, tile_dim_.z + 1 };

    SetBcAxisAsync(*tmp_u_x_, x_tile_dim, *is_bc_x_, *bc_val_x_, _stream);
    SetBcAxisAsync(*tmp_u_y_, y_tile_dim, *is_bc_y_, *bc_val_y_, _stream);
    SetBcAxisAsync(*tmp_u_z_, z_tile_dim, *is_bc_z_, *bc_val_z_, _stream);

    CalcDivAsync(*(amgpcg_.b_), tile_dim_, *(amgpcg_.poisson_vector_[0].is_dof_), *tmp_u_x_, *tmp_u_y_, *tmp_u_z_, _stream);

    amgpcg_.SolveAsync(_stream);

    ApplyPressureAsync(*tmp_u_x_, *tmp_u_y_, *tmp_u_z_, tile_dim_, *(amgpcg_.x_), *is_bc_x_, *is_bc_y_, *is_bc_z_, _stream);
}
}
