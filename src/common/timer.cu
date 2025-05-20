#include "timer.h"

namespace lfm {
void GPUTimer::Start(void)
{
    cudaEventRecord(start);
}

void GPUTimer::Stop()
{
    cudaEventRecord(stop);
}

void GPUTimer::Elapsed(const std::string& _message)
{
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("%s: %f ms\n", _message.c_str(), time);
}
}