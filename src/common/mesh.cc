#include "mesh.h"

namespace lfm {
Mesh::Mesh(int _vtx_num, int _tri_num)
{
    Alloc(_vtx_num, _tri_num);
}

void Mesh::Alloc(int _vtx_num, int _tri_num)
{
    vtx_num_ = _vtx_num;
    tri_num_ = _tri_num;

    v_   = std::make_shared<DHMemory<float3>>(vtx_num_);
    tri_ = std::make_shared<DHMemory<int3>>(tri_num_);
}
}