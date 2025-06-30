#pragma once

#include "core/config/config.h"
#include "lfm.h"

namespace lfm {
void InitLFMAsync(LFM& _lfm, const LFMConfiguration& _config, cudaStream_t _stream);
}