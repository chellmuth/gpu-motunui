#pragma once

#include <optix.h>

#include "scene/geometry_result.hpp"

namespace moana {

class MountainAGeometry {
public:
    GeometryResult buildAcceleration(OptixDeviceContext context);
};

}
