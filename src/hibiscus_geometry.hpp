#pragma once

#include <vector>

#include <optix.h>

namespace moana {

struct GeometryResult {
    OptixTraversableHandle handle;

    std::vector<CUdeviceptr> outputBuffers;
};

class HibiscusGeometry {
public:
    GeometryResult buildAcceleration(OptixDeviceContext context);
};

}
