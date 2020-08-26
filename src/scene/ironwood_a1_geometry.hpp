#pragma once

#include <optix.h>

#include "scene/geometry_result.hpp"

namespace moana {

class IronwoodA1Geometry {
public:
    GeometryResult buildAcceleration(OptixDeviceContext context);
};

}
