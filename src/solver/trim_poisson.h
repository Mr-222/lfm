#pragma once

#include "mem.h"
#include "setting.h"
#include "util.h"
#include <memory>

namespace lfm {
class TrimPoisson {
public:
    int3 tile_dim_;
    std::shared_ptr<DHMemory<float>> x_;
    std::shared_ptr<DHMemory<char>> is_dof_;
    std::shared_ptr<DHMemory<float>> a_diag_;
    std::shared_ptr<DHMemory<float>> a_x_;
    std::shared_ptr<DHMemory<float>> a_y_;
    std::shared_ptr<DHMemory<float>> a_z_;
    std::shared_ptr<DHMemory<float>> b_;
    std::shared_ptr<DHMemory<float>> buffer_;

    // coefficient trim
    std::shared_ptr<DHMemory<char>> tile_trimmed_; // 0: empty, 1: trivial coefficient, 2: nontrivial coefficient
    float default_a_diag_;
    float default_a_off_diag_;

    TrimPoisson() = default;
    TrimPoisson(int3 _tile_dim);
    void Alloc(int3 _tile_dim);
    void TrimAsync(float _default_a_diag, float _default_a_off_diag, cudaStream_t _stream);

    void LaplacianDotAsync(std::shared_ptr<DHMemory<float>> _output, std::shared_ptr<DHMemory<float>> _dot_buffer, const std::shared_ptr<DHMemory<float>> _input, cudaStream_t _stream) const;
    void GaussSeidelRestrictAsync(std::shared_ptr<DHMemory<float>> _coarse_b, cudaStream_t _stream);
    void ProlongGaussSeidelDotAsync(const std::shared_ptr<DHMemory<float>> _coarse_x, std::shared_ptr<DHMemory<float>> _dot_buffer, bool _do_dot, cudaStream_t _stream);
    void CoarsestGaussSeidelAsync(int _num_iter, cudaStream_t _stream);
};
}