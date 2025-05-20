#include "mem.h"
#include "setting.h"
#include "util.h"
#include <cstring>

namespace lfm {
template <typename T>
DHMemory<T>::DHMemory(int _size)
{
    size_ = _size;
    cudaMalloc((void**)(&dev_ptr_), size_ * sizeof(T));
    host_ptr_ = new T[_size];
}

template <typename T>
DHMemory<T>::~DHMemory()
{
    cudaFree((void*)dev_ptr_);
    delete[] host_ptr_;
}

template <typename T>
void DHMemory<T>::DevToHostAsync(cudaStream_t _stream)
{
    cudaMemcpyAsync((void*)host_ptr_, (const void*)dev_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost, _stream);
}

template <typename T>
void DHMemory<T>::HostToDevAsync(cudaStream_t _stream)
{
    cudaMemcpyAsync((void*)dev_ptr_, (const void*)host_ptr_, size_ * sizeof(T), cudaMemcpyHostToDevice, _stream);
}

template <typename T>
void DHMemory<T>::ClearDevAsync(cudaStream_t _stream)
{
    cudaMemsetAsync((void*)dev_ptr_, 0, size_ * sizeof(T), _stream);
}

template <typename T>
void DHMemory<T>::ClearHost()
{
    memset((void*)host_ptr_, 0, size_ * sizeof(T));
}

template <typename T>
T& DHMemory<T>::operator[](int _i)
{
    return host_ptr_[_i];
}

template class DHMemory<char>;
template class DHMemory<int>;
template class DHMemory<float>;
template class DHMemory<int3>;
template class DHMemory<float3>;
template class DHMemory<float4>;
};