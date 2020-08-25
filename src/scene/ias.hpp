#pragma once

#include <vector>

#include <optix.h>

#include "scene/types.hpp"

namespace moana { namespace IAS {

void createOptixInstanceRecords(
    OptixDeviceContext context,
    std::vector<OptixInstance> &records,
    const Instances &instances,
    const OptixTraversableHandle &traversableHandle
);

OptixTraversableHandle iasFromInstanceRecords(
    OptixDeviceContext context,
    const std::vector<OptixInstance> &records
);

} }
