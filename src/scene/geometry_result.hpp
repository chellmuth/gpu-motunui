#pragma once

#include <vector>

#include <optix.h>

namespace moana {

struct GeometryResult {
    OptixTraversableHandle handle;

    std::vector<CUdeviceptr> outputBuffers;
};

}
