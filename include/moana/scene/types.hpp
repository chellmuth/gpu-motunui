#pragma once

#include <vector>

#include <optix.h>
#include <cuda.h>

#include "moana/scene/as_arena.hpp"

namespace moana {

struct GeometryResult {
    OptixTraversableHandle handle;
    Snapshot snapshot;
};

struct Instances {
    int count;
    float *transforms;
};

}
