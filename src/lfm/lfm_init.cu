#include "data_io.h"
#include "lfm_init.h"
#include "lfm_util.h"
namespace lfm {
void InitLFMAsync(LFM& _lfm, const LFMConfiguration& _config, cudaStream_t _stream)
{
    // alloc
    int3 tile_dim    = { _config.tile_dim[0], _config.tile_dim[1], _config.tile_dim[2] };
    int reinit_every = _config.reinit_every;
    int num_smoke    = _config.num_smoke;
    _lfm.Alloc(tile_dim, reinit_every, num_smoke);

    // simulation parameter
    _lfm.rk_order_ = _config.rk_order;
    _lfm.step_     = 0;

    // domain
    _lfm.dx_            = _config.len_y / (8 * tile_dim.y);
    _lfm.grid_origin_.x = _config.grid_origin[0];
    _lfm.grid_origin_.y = _config.grid_origin[1];
    _lfm.grid_origin_.z = _config.grid_origin[2];

    // boundary
    _lfm.inlet_norm_   = _config.inlet_norm;
    _lfm.inlet_angle_  = _config.inlet_angle;
    float pi           = 3.1415926f;
    float radian_angle = _lfm.inlet_angle_ / 180.0f * pi;
    float3 neg_bc_val  = { _lfm.inlet_norm_ * cos(radian_angle), _lfm.inlet_norm_ * sin(radian_angle), 0.0f };
    float3 pos_bc_val  = neg_bc_val;
    SetWallBcAsync(*_lfm.is_bc_x_, *_lfm.is_bc_y_, *_lfm.is_bc_z_, *_lfm.bc_val_x_, *_lfm.bc_val_y_, *_lfm.bc_val_z_, tile_dim, neg_bc_val, pos_bc_val, _stream);

    bool use_static_solid = _config.use_static_solid;
    DHMemory<float> solid_sdf(Prod(tile_dim) * 512);

    if (use_static_solid) {
        DHMemory<float> solid_sdf_np(Prod(tile_dim) * 512);
        ReadNpy<float>(_config.solid_sdf_path, solid_sdf_np.host_ptr_);
        solid_sdf_np.HostToDevAsync(_stream);
        ConToTileAsync(solid_sdf, tile_dim, solid_sdf_np, _stream);
        SetBcByPhiAsync(*_lfm.is_bc_x_, *_lfm.is_bc_y_, *_lfm.is_bc_z_, *_lfm.bc_val_x_, *_lfm.bc_val_y_, *_lfm.bc_val_z_, tile_dim, solid_sdf, _stream);
    }

    // init velocity
    DHMemory<float> init_u_x_np((8 * tile_dim.x + 1) * (8 * tile_dim.y) * (8 * tile_dim.z));
    DHMemory<float> init_u_y_np((8 * tile_dim.x) * (8 * tile_dim.y + 1) * (8 * tile_dim.z));
    DHMemory<float> init_u_z_np((8 * tile_dim.x) * (8 * tile_dim.y) * (8 * tile_dim.z + 1));
    ReadNpy<float>(_config.init_u_x_path, init_u_x_np.host_ptr_);
    ReadNpy<float>(_config.init_u_y_path, init_u_y_np.host_ptr_);
    ReadNpy<float>(_config.init_u_z_path, init_u_z_np.host_ptr_);
    init_u_x_np.HostToDevAsync(_stream);
    init_u_y_np.HostToDevAsync(_stream);
    init_u_z_np.HostToDevAsync(_stream);

    StagConToTileAsync(*_lfm.init_u_x_, *_lfm.init_u_y_, *_lfm.init_u_z_, tile_dim, init_u_x_np, init_u_y_np, init_u_z_np, _stream);
    // poisson
    {
        SetCoefByIsBcAsync(*(_lfm.amgpcg_.poisson_vector_[0].is_dof_), *(_lfm.amgpcg_.poisson_vector_[0].a_diag_), *(_lfm.amgpcg_.poisson_vector_[0].a_x_), *(_lfm.amgpcg_.poisson_vector_[0].a_y_),
                           *(_lfm.amgpcg_.poisson_vector_[0].a_z_), tile_dim, *_lfm.is_bc_x_, *_lfm.is_bc_y_, *_lfm.is_bc_z_, _stream);
        _lfm.amgpcg_.BuildAsync(6.0f, -1.0f, _stream);
        _lfm.amgpcg_.solve_by_tol_ = false;
        _lfm.amgpcg_.max_iter_ = 6;
    }

    // smoke
    if (_lfm.num_smoke_ > 0) {
        DHMemory<float> init_smoke_np(512 * Prod(tile_dim));
        for (int i = 0; i < _lfm.num_smoke_; i++) {
            ReadNpy<float>(_config.init_smoke_path_prefix + std::to_string(i) + ".npy", init_smoke_np.host_ptr_);
            init_smoke_np.HostToDevAsync(_stream);
            ConToTileAsync(*_lfm.smoke_[i], tile_dim, init_smoke_np, _stream);
            DevToDevCpyAsync(_lfm.init_smoke_[i]->dev_ptr_, _lfm.smoke_[i]->dev_ptr_, 512 * Prod(tile_dim), _stream);
        }
    }
    // bfecc clamp
    _lfm.use_bfecc_clamp_ = _config.use_bfecc_clamp;
}
}