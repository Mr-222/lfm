#pragma once

#include "core/config/config.h"
#include "nfm.h"

namespace lfm {
void InitNFMAsync(NFM& _nfm, const NFMConfiguration& _config, cudaStream_t _stream);
}