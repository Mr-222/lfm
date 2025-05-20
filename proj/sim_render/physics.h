#pragma once

#include "function/physics/cuda_engine.h"
#include "nfm.h"
#include <glm/glm.hpp>

class PhysicsEngineUser : public CudaEngine {
    lfm::NFM nfm;
    virtual void initExternalMem() override;

public:
    virtual void init(Configuration& config, GlobalContext* g_ctx) override;
    virtual void step() override;
    virtual void cleanup() override;
};
