#pragma once

#include <optix.h>
#include <cuda.h>

namespace moana {

struct Instances {
    int count;
    float *transforms;
};

struct GASInfo {
    OptixTraversableHandle handle;
    CUdeviceptr gasOutputBuffer;
    size_t outputBufferSizeInBytes;
};

}
