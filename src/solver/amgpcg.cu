#include "amgpcg.h"
#include "timer.h"
#include <cub/cub.cuh>

namespace lfm {
void AMGPCG::Alloc(int3 _tile_dim, int _level_num)
{
    tile_dim_  = _tile_dim;
    level_num_ = _level_num;

    rTr_     = std::make_shared<DHMemory<float>>(1);
    old_zTr_ = std::make_shared<DHMemory<float>>(1);
    new_zTr_ = std::make_shared<DHMemory<float>>(1);
    pAp_     = std::make_shared<DHMemory<float>>(1);
    alpha_   = std::make_shared<DHMemory<float>>(1);
    beta_    = std::make_shared<DHMemory<float>>(1);
    avg_     = std::make_shared<DHMemory<float>>(1);
    num_dof_ = std::make_shared<DHMemory<int>>(1);

    int voxel_num = Prod(tile_dim_) * 512;
    x_            = std::make_shared<DHMemory<float>>(voxel_num);
    p_            = std::make_shared<DHMemory<float>>(voxel_num);
    Ap_           = std::make_shared<DHMemory<float>>(voxel_num);

    int tile_num = Prod(tile_dim_);
    size_t tmp_size;
    block_num_dof_ = std::make_shared<DHMemory<int>>(tile_num);
    cub::DeviceReduce::Sum(nullptr, tmp_size, block_num_dof_->dev_ptr_, num_dof_->dev_ptr_, tile_num);
    int tmp_size_int   = tmp_size;
    block_num_dof_tmp_ = std::make_shared<DHMemory<char>>(tmp_size_int);

    dot_buffer_ = std::make_shared<DHMemory<float>>(Prod(tile_dim_));
    cub::DeviceReduce::Sum(nullptr, tmp_size, dot_buffer_->dev_ptr_, alpha_->dev_ptr_, tile_num);
    tmp_size_int = tmp_size;
    dot_tmp_     = std::make_shared<DHMemory<char>>(tmp_size_int);

    poisson_vector_.resize(level_num_);
    int3 l_tile_dim = tile_dim_;
    for (int l = 0; l < level_num_; l++) {
        poisson_vector_[l].Alloc(l_tile_dim);
        l_tile_dim = { (l_tile_dim.x + 1) / 2, (l_tile_dim.y + 1) / 2, (l_tile_dim.z + 1) / 2 };
    }

    b_ = poisson_vector_[0].b_;
}

AMGPCG::AMGPCG(int3 _tile_dim, int _level_num)
{
    Alloc(_tile_dim, _level_num);
}

__global__ void CoarsenKernel(char* _coarse_is_dof, float* _coarse_a_diag, float* _coarse_a_x, float* _coarse_a_y, float* _coarse_a_z, int3 _coarse_tile_dim, int3 _fine_tile_dim,
                              const char* _fine_is_dof, const float* _fine_a_diag, const float* _fine_a_x, const float* _fine_a_y, const float* _fine_a_z)
{
    int3 fine_tile_dim = { _coarse_tile_dim.x * 2, _coarse_tile_dim.y * 2, _coarse_tile_dim.z * 2 };
    char fine_is_dof[2][2][2];
    int coarse_tile_idx  = blockIdx.x;
    int3 coarse_tile_ijk = TileIdxToIjk(_coarse_tile_dim, coarse_tile_idx);
    int t_id             = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int coarse_voxel_idx     = i * 128 + t_id;
        int3 coarse_voxel_ijk    = VoxelIdxToIjk(coarse_voxel_idx);
        int3 fine_tile_ijk       = { coarse_tile_ijk.x * 2, coarse_tile_ijk.y * 2, coarse_tile_ijk.z * 2 };
        int3 fine_base_voxel_ijk = { coarse_voxel_ijk.x * 2, coarse_voxel_ijk.y * 2, coarse_voxel_ijk.z * 2 };
        if (fine_base_voxel_ijk.x >= 8) {
            fine_base_voxel_ijk.x -= 8;
            fine_tile_ijk.x++;
        }
        if (fine_base_voxel_ijk.y >= 8) {
            fine_base_voxel_ijk.y -= 8;
            fine_tile_ijk.y++;
        }
        if (fine_base_voxel_ijk.z >= 8) {
            fine_base_voxel_ijk.z -= 8;
            fine_tile_ijk.z++;
        }
        int fine_tile_idx = TileIjkToIdx(fine_tile_dim, fine_tile_ijk);

        // is_dof
        char coarse_is_dof = (char)0;
        if (fine_tile_ijk.x < _fine_tile_dim.x && fine_tile_ijk.y < _fine_tile_dim.y && fine_tile_ijk.z < _fine_tile_dim.z) {
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++) {
                        int3 fine_target_voxel_ijk = { fine_base_voxel_ijk.x + a, fine_base_voxel_ijk.y + b, fine_base_voxel_ijk.z + c };
                        int fine_target_voxel_idx  = VoxelIjkToIdx(fine_target_voxel_ijk);
                        fine_is_dof[a][b][c]       = _fine_is_dof[fine_tile_idx * 512 + fine_target_voxel_idx];
                        if (fine_is_dof[a][b][c])
                            coarse_is_dof = (char)1;
                    }
        }
        _coarse_is_dof[coarse_tile_idx * 512 + coarse_voxel_idx] = coarse_is_dof;

        // a_diag
        float coarse_a_diag = 0.0f;
        if (coarse_is_dof)
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++)
                    for (int c = 0; c < 2; c++)
                        if (fine_is_dof[a][b][c]) {
                            int3 fine_target_voxel_ijk = { fine_base_voxel_ijk.x + a, fine_base_voxel_ijk.y + b, fine_base_voxel_ijk.z + c };
                            int fine_target_voxel_idx  = VoxelIjkToIdx(fine_target_voxel_ijk);
                            coarse_a_diag += _fine_a_diag[fine_tile_idx * 512 + fine_target_voxel_idx];
                            if (a == 0 && fine_is_dof[1][b][c])
                                coarse_a_diag += 2.0f * _fine_a_x[fine_tile_idx * 512 + fine_target_voxel_idx];
                            if (b == 0 && fine_is_dof[a][1][c])
                                coarse_a_diag += 2.0f * _fine_a_y[fine_tile_idx * 512 + fine_target_voxel_idx];
                            if (c == 0 && fine_is_dof[a][b][1])
                                coarse_a_diag += 2.0f * _fine_a_z[fine_tile_idx * 512 + fine_target_voxel_idx];
                        }
        coarse_a_diag *= 0.125f;
        _coarse_a_diag[coarse_tile_idx * 512 + coarse_voxel_idx] = coarse_a_diag;

        // a_x
        float coarse_a_x = 0.0f;
        if (coarse_is_dof)
            for (int b = 0; b < 2; b++)
                for (int c = 0; c < 2; c++)
                    if (fine_is_dof[1][b][c]) {
                        int3 fine_target_voxel_ijk = { fine_base_voxel_ijk.x + 1, fine_base_voxel_ijk.y + b, fine_base_voxel_ijk.z + c };
                        int fine_target_voxel_idx  = VoxelIjkToIdx(fine_target_voxel_ijk);
                        int3 fine_nb_voxel_ijk     = { fine_base_voxel_ijk.x + 2, fine_base_voxel_ijk.y + b, fine_base_voxel_ijk.z + c };
                        int3 fine_nb_tile_ijk      = fine_tile_ijk;
                        if (fine_nb_voxel_ijk.x == 8) {
                            fine_nb_voxel_ijk.x = 0;
                            fine_nb_tile_ijk.x++;
                        }
                        if (fine_nb_tile_ijk.x < fine_tile_dim.x) {
                            int fine_nb_tile_idx  = TileIjkToIdx(fine_tile_dim, fine_nb_tile_ijk);
                            int fine_nb_voxel_idx = VoxelIjkToIdx(fine_nb_voxel_ijk);
                            if (_fine_is_dof[fine_nb_tile_idx * 512 + fine_nb_voxel_idx])
                                coarse_a_x += _fine_a_x[fine_tile_idx * 512 + fine_target_voxel_idx];
                        }
                    }
        coarse_a_x *= 0.125f;
        _coarse_a_x[coarse_tile_idx * 512 + coarse_voxel_idx] = coarse_a_x;

        // a_y
        float coarse_a_y = 0.0f;
        if (coarse_is_dof)
            for (int a = 0; a < 2; a++)
                for (int c = 0; c < 2; c++)
                    if (fine_is_dof[a][1][c]) {
                        int3 fine_target_voxel_ijk = { fine_base_voxel_ijk.x + a, fine_base_voxel_ijk.y + 1, fine_base_voxel_ijk.z + c };
                        int fine_target_voxel_idx  = VoxelIjkToIdx(fine_target_voxel_ijk);
                        int3 fine_nb_voxel_ijk     = { fine_base_voxel_ijk.x + a, fine_base_voxel_ijk.y + 2, fine_base_voxel_ijk.z + c };
                        int3 fine_nb_tile_ijk      = fine_tile_ijk;
                        if (fine_nb_voxel_ijk.y == 8) {
                            fine_nb_voxel_ijk.y = 0;
                            fine_nb_tile_ijk.y++;
                        }
                        if (fine_nb_tile_ijk.y < fine_tile_dim.y) {
                            int fine_nb_tile_idx  = TileIjkToIdx(fine_tile_dim, fine_nb_tile_ijk);
                            int fine_nb_voxel_idx = VoxelIjkToIdx(fine_nb_voxel_ijk);
                            if (_fine_is_dof[fine_nb_tile_idx * 512 + fine_nb_voxel_idx])
                                coarse_a_y += _fine_a_y[fine_tile_idx * 512 + fine_target_voxel_idx];
                        }
                    }
        coarse_a_y *= 0.125f;
        _coarse_a_y[coarse_tile_idx * 512 + coarse_voxel_idx] = coarse_a_y;

        // a_z
        float coarse_a_z = 0.0f;
        if (coarse_is_dof)
            for (int a = 0; a < 2; a++)
                for (int b = 0; b < 2; b++)
                    if (fine_is_dof[a][b][1]) {
                        int3 fine_target_voxel_ijk = { fine_base_voxel_ijk.x + a, fine_base_voxel_ijk.y + b, fine_base_voxel_ijk.z + 1 };
                        int fine_target_voxel_idx  = VoxelIjkToIdx(fine_target_voxel_ijk);
                        int3 fine_nb_voxel_ijk     = { fine_base_voxel_ijk.x + a, fine_base_voxel_ijk.y + b, fine_base_voxel_ijk.z + 2 };
                        int3 fine_nb_tile_ijk      = fine_tile_ijk;
                        if (fine_nb_voxel_ijk.z == 8) {
                            fine_nb_voxel_ijk.z = 0;
                            fine_nb_tile_ijk.z++;
                        }
                        if (fine_nb_tile_ijk.z < fine_tile_dim.z) {
                            int fine_nb_tile_idx  = TileIjkToIdx(fine_tile_dim, fine_nb_tile_ijk);
                            int fine_nb_voxel_idx = VoxelIjkToIdx(fine_nb_voxel_ijk);
                            if (_fine_is_dof[fine_nb_tile_idx * 512 + fine_nb_voxel_idx])
                                coarse_a_z += _fine_a_z[fine_tile_idx * 512 + fine_target_voxel_idx];
                        }
                    }
        coarse_a_z *= 0.125f;
        _coarse_a_z[coarse_tile_idx * 512 + coarse_voxel_idx] = coarse_a_z;
    }
}

void AMGPCG::BuildAsync(float _default_a_diag, float _default_a_off_diag, cudaStream_t _stream)
{
    poisson_vector_[0].TrimAsync(_default_a_diag, _default_a_off_diag, _stream);
    for (int i = 1; i < level_num_; i++) {
        {
            char* coarse_is_dof      = poisson_vector_[i].is_dof_->dev_ptr_;
            float* coarse_a_diag     = poisson_vector_[i].a_diag_->dev_ptr_;
            float* coarse_a_x        = poisson_vector_[i].a_x_->dev_ptr_;
            float* coarse_a_y        = poisson_vector_[i].a_y_->dev_ptr_;
            float* coarse_a_z        = poisson_vector_[i].a_z_->dev_ptr_;
            int3 coarse_tile_dim     = poisson_vector_[i].tile_dim_;
            int3 fine_tile_dim       = poisson_vector_[i - 1].tile_dim_;
            const char* fine_is_dof  = poisson_vector_[i - 1].is_dof_->dev_ptr_;
            const float* fine_a_diag = poisson_vector_[i - 1].a_diag_->dev_ptr_;
            const float* fine_a_x    = poisson_vector_[i - 1].a_x_->dev_ptr_;
            const float* fine_a_y    = poisson_vector_[i - 1].a_y_->dev_ptr_;
            const float* fine_a_z    = poisson_vector_[i - 1].a_z_->dev_ptr_;
            int coarse_tile_num      = Prod(coarse_tile_dim);
            CoarsenKernel<<<coarse_tile_num, 128, 0, _stream>>>(coarse_is_dof, coarse_a_diag, coarse_a_x, coarse_a_y, coarse_a_z, coarse_tile_dim, fine_tile_dim, fine_is_dof, fine_a_diag, fine_a_x, fine_a_y, fine_a_z);
            _default_a_diag *= 0.5f;
            _default_a_off_diag *= 0.5f;
            poisson_vector_[i].TrimAsync(_default_a_diag, _default_a_off_diag, _stream);
        }
    }
    if (pure_neumann_)
        CountDofAsync(_stream);
}

void AMGPCG::VcycleDotAsync(cudaStream_t _stream)
{
    for (int l = 0; l < level_num_ - 1; l++)
        poisson_vector_[l].GaussSeidelRestrictAsync(poisson_vector_[l + 1].b_, _stream);
    poisson_vector_[level_num_ - 1].CoarsestGaussSeidelAsync(bottom_smoothing_, _stream);
    for (int l = level_num_ - 2; l >= 0; l--) {
        bool do_dot = (l == 0);
        poisson_vector_[l].ProlongGaussSeidelDotAsync(poisson_vector_[l + 1].x_, dot_buffer_, do_dot, _stream);
    }
}

__global__ void MulKernel(float* _dst, const char* _is_dof, const float* _src1, const float* _src2, int3 _tile_dim)
{
    int tile_idx = blockIdx.x;
    int t_id     = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx = i * 128 + t_id;
        float val     = 0.0f;
        if (_is_dof[tile_idx * 512 + voxel_idx])
            val = _src1[tile_idx * 512 + voxel_idx] * _src2[tile_idx * 512 + voxel_idx];
        _dst[tile_idx * 512 + voxel_idx] = val;
    }
}

__global__ void DivKernel(float* _dst, const float* _src1, const float* _src2, const float _eps)
{
    _dst[0] = _src1[0] / (_src2[0] + _eps);
}

void DivideAsync(std::shared_ptr<DHMemory<float>> _dst, const std::shared_ptr<DHMemory<float>> _src1,
                 const std::shared_ptr<DHMemory<float>> _src2, const float _eps, cudaStream_t _stream)
{
    float* dst        = _dst->dev_ptr_;
    const float* src1 = _src1->dev_ptr_;
    const float* src2 = _src2->dev_ptr_;
    DivKernel<<<1, 1, 0, _stream>>>(dst, src1, src2, _eps);
}

__global__ void AxpyKernel(float* _dst, int3 _tile_dim, const char* _is_dof, const float* _coef, float* _x, float* _y)
{
    int tile_idx = blockIdx.x;
    int t_id     = threadIdx.x;
    float coef   = _coef[0];
    for (int i = 0; i < 4; i++) {
        int voxel_idx = i * 128 + t_id;
        int idx       = tile_idx * 512 + voxel_idx;
        if (_is_dof[idx])
            _dst[idx] = coef * _x[idx] + _y[idx];
    }
}

void AMGPCG::AxpyAsync(std::shared_ptr<DHMemory<float>> _dst, const std::shared_ptr<DHMemory<float>> _coef,
                       std::shared_ptr<DHMemory<float>> _x, std::shared_ptr<DHMemory<float>> _y, cudaStream_t _stream)
{
    int tile_num       = Prod(tile_dim_);
    float* dst         = _dst->dev_ptr_;
    const char* is_dof = poisson_vector_[0].is_dof_->dev_ptr_;
    const float* coef  = _coef->dev_ptr_;
    float* x           = _x->dev_ptr_;
    float* y           = _y->dev_ptr_;
    AxpyKernel<<<tile_num, 128, 0, _stream>>>(dst, tile_dim_, is_dof, coef, x, y);
}

__global__ void YmAxKernel(float* _dst, int3 _tile_dim, const char* _is_dof, const float* _coef, float* _x)
{
    int tile_idx = blockIdx.x;
    int t_id     = threadIdx.x;
    float coef   = -_coef[0];
    for (int i = 0; i < 4; i++) {
        int voxel_idx = i * 128 + t_id;
        int idx       = tile_idx * 512 + voxel_idx;
        if (_is_dof[idx]) {
            _dst[idx] = coef * _x[idx] + _dst[idx];
        }
    }
}

void AMGPCG::YmAxAsync(std::shared_ptr<DHMemory<float>> _dst,
                       const std::shared_ptr<DHMemory<float>> _coef, std::shared_ptr<DHMemory<float>> _x, cudaStream_t _stream)
{
    int tile_num       = Prod(tile_dim_);
    float* dst         = _dst->dev_ptr_;
    const char* is_dof = poisson_vector_[0].is_dof_->dev_ptr_;
    const float* coef  = _coef->dev_ptr_;
    float* x           = _x->dev_ptr_;
    YmAxKernel<<<tile_num, 128, 0, _stream>>>(dst, tile_dim_, is_dof, coef, x);
}

void AMGPCG::DotSumAsync(std::shared_ptr<DHMemory<float>> _dst, cudaStream_t _stream)
{
    int tile_num              = Prod(tile_dim_);
    void* d_temp_storage      = (void*)(dot_tmp_->dev_ptr_);
    size_t temp_storage_bytes = dot_tmp_->size_;
    const float* dot_buffer   = dot_buffer_->dev_ptr_;
    float* dst                = _dst->dev_ptr_;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dot_buffer, dst, tile_num, _stream);
}

void AMGPCG::SolveAsync(int _iter_num, cudaStream_t _stream)
{
    // float rel_tol = 1e-12;
    // float abs_tol = 1e-14;
    float eps = 1e-20;
    x_->ClearDevAsync(_stream);

    // DotAsync(rTr_, poisson_vector_[0].b_, poisson_vector_[0].b_, _stream);
    // rTr_->DevToHostAsync(_stream);
    // cudaStreamSynchronize(_stream);
    // float initial_rTr = rTr_->host_ptr_[0];
    // std::cout << "init |residual|_2 = " << sqrt(initial_rTr) << std::endl;
    // float tol = max(abs_tol, initial_rTr * rel_tol);
    if (pure_neumann_)
        RecenterAsync(b_, _stream);
    VcycleDotAsync(_stream);
    DotSumAsync(old_zTr_, _stream);
    p_.swap(poisson_vector_[0].x_);
    // int iter = 0;
    // while (true)
    for (int iter = 0; iter < _iter_num; iter++) {

        poisson_vector_[0].LaplacianDotAsync(Ap_, dot_buffer_, p_, _stream);
        DotSumAsync(pAp_, _stream);

        DivideAsync(alpha_, old_zTr_, pAp_, eps, _stream);

        AxpyAsync(x_, alpha_, p_, x_, _stream);

        if (iter == _iter_num - 1)
            break;

        YmAxAsync(poisson_vector_[0].b_, alpha_, Ap_, _stream);
        // DotSumAsync(rTr_, _stream);

        // rTr_->DevToHostAsync(_stream);
        // cudaStreamSynchronize(_stream);
        // std::cout << "iter " << iter << ", |residual|_2 = " << sqrt(rTr_->host_ptr_[0]) << std::endl;
        // if (rTr_->host_ptr_[0] < tol)
        //	break;
        if (pure_neumann_)
            RecenterAsync(b_, _stream);
        VcycleDotAsync(_stream);
        DotSumAsync(new_zTr_, _stream);

        DivideAsync(beta_, new_zTr_, old_zTr_, eps, _stream);

        AxpyAsync(p_, beta_, p_, poisson_vector_[0].x_, _stream);

        old_zTr_.swap(new_zTr_);

        // iter++;
    }
}

__global__ void CalcBlockSumKernel(float* _dst, const float* _src, const char* _is_dof)
{
    int tile_idx = blockIdx.x;
    int t_id     = threadIdx.x;
    float val[4];
    for (int i = 0; i < 4; i++) {
        int voxel_idx = i * 128 + t_id;
        if (_is_dof[tile_idx * 512 + voxel_idx])
            val[i] = _src[tile_idx * 512 + voxel_idx];
        else
            val[i] = 0.0f;
    }
    using BlockReduce = cub::BlockReduce<float, 128>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float sum = BlockReduce(temp_storage).Sum(val);
    if (t_id == 0)
        _dst[tile_idx] = sum;
}

__global__ void DivIntKernel(float* _dst, const float* _src1, const int* _src2)
{
    _dst[0] = _src1[0] / _src2[0];
}

__global__ void MinusKernel(float* _dst, const char* _is_dof, const float* _val)
{
    float val    = _val[0];
    int tile_idx = blockIdx.x;
    int t_id     = threadIdx.x;
    for (int i = 0; i < 4; i++) {
        int voxel_idx = i * 128 + t_id;
        if (_is_dof[tile_idx * 512 + voxel_idx])
            _dst[tile_idx * 512 + voxel_idx] -= val;
    }
}

void AMGPCG::RecenterAsync(std::shared_ptr<DHMemory<float>> _dst, cudaStream_t _stream)
{
    int tile_num       = Prod(tile_dim_);
    float* src         = _dst->dev_ptr_;
    float* dst         = dot_buffer_->dev_ptr_;
    const char* is_dof = poisson_vector_[0].is_dof_->dev_ptr_;
    CalcBlockSumKernel<<<tile_num, 128, 0, _stream>>>(dst, src, is_dof);
    void* d_temp_storage      = (void*)(dot_tmp_->dev_ptr_);
    size_t temp_storage_bytes = dot_tmp_->size_;
    float* avg                = avg_->dev_ptr_;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dst, avg, tile_num, _stream);
    int* num_dof = num_dof_->dev_ptr_;
    DivIntKernel<<<1, 1, 0, _stream>>>(avg, avg, num_dof);
    MinusKernel<<<tile_num, 128, 0, _stream>>>(src, is_dof, avg);
}

__global__ void CountBlockDofKernel(int* _dst, const char* _is_dof)
{
    int tile_idx = blockIdx.x;
    int t_id     = threadIdx.x;
    int count[4];
    for (int i = 0; i < 4; i++) {
        int voxel_idx = i * 128 + t_id;
        if (_is_dof[tile_idx * 512 + voxel_idx])
            count[i] = 1;
        else
            count[i] = 0;
    }

    using BlockReduce = cub::BlockReduce<int, 128>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int sum = BlockReduce(temp_storage).Sum(count);
    if (t_id == 0)
        _dst[tile_idx] = sum;
}

void AMGPCG::CountDofAsync(cudaStream_t _stream)
{
    int tile_num       = Prod(tile_dim_);
    int* dst           = block_num_dof_->dev_ptr_;
    const char* is_dof = poisson_vector_[0].is_dof_->dev_ptr_;
    CountBlockDofKernel<<<tile_num, 128, 0, _stream>>>(dst, is_dof);

    void* d_temp_storage      = (void*)(block_num_dof_tmp_->dev_ptr_);
    size_t temp_storage_bytes = block_num_dof_tmp_->size_;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, dst, num_dof_->dev_ptr_, tile_num, _stream);
}
};