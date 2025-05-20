#include "util.h"
#include <iostream>

namespace lfm {
template <typename T>
void DevToDevCpyAsync(T* _dst, const T* _src, int _size, cudaStream_t _stream)
{
    cudaMemcpyAsync((void*)_dst, (const void*)_src, _size * sizeof(T), cudaMemcpyDeviceToDevice, _stream);
}

template void DevToDevCpyAsync<int>(int*, const int*, int, cudaStream_t);
template void DevToDevCpyAsync<float>(float*, const float*, int, cudaStream_t);
};