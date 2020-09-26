#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "moana/cuda/bsdf.hpp"

namespace moana { namespace Materials {

extern std::vector<float3> baseColors;
extern std::vector<BSDFType> bsdfTypes;

} }
