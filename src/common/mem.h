#pragma once

#include <cuda_runtime.h>

namespace lfm {
template <typename T>
class DHMemory {
public:
    int size_;
    T* dev_ptr_;
    T* host_ptr_;

    DHMemory()                                   = delete;
    DHMemory(DHMemory const& _rhs)               = delete;
    DHMemory<T>& operator=(DHMemory const& _rhs) = delete;
    DHMemory(int _size);
    ~DHMemory();
    void DevToHostAsync(cudaStream_t _stream);
    void HostToDevAsync(cudaStream_t _stream);
    void ClearDevAsync(cudaStream_t _stream);
    void ClearHost();
    T& operator[](int _i);
};
};