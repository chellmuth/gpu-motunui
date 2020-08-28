#pragma once

#include <vector>

#include <optix.h>

#include "moana/scene/as_arena.hpp"
#include "moana/scene/types.hpp"

namespace moana { namespace IAS {

void createOptixInstanceRecords(
    OptixDeviceContext context,
    std::vector<OptixInstance> &records,
    const Instances &instances,
    const OptixTraversableHandle &traversableHandle,
    int sbtOffset = 0 // not needed when traversableHandle is an IAS
);

OptixTraversableHandle iasFromInstanceRecords(
    OptixDeviceContext context,
    ASArena &arena,
    const std::vector<OptixInstance> &records
);

} }
