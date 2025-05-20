#pragma once

#include "mem.h"
#include <memory>

namespace lfm {
class Mesh {
public:
    int vtx_num_;
    int tri_num_;

    std::shared_ptr<DHMemory<float3>> v_;
    std::shared_ptr<DHMemory<int3>> tri_;

    Mesh() = default;
    Mesh(int _vtx_num, int _tri_num);
    void Alloc(int _vtx_num, int _tri_num);
};
}