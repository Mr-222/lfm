#pragma once

#include <cuda_runtime.h>
#include <string>

namespace lfm {
class GPUTimer {
public:
    cudaEvent_t start, stop;

    GPUTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        Start();
    }
    ~GPUTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start();
    void Stop();
    void Elapsed(const std::string& _message);
};

}
