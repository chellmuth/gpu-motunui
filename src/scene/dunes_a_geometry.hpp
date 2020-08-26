#pragma once

#include <optix.h>

#include "moana/scene/as_arena.hpp"
#include "moana/scene/types.hpp"

namespace moana {

class DunesAGeometry {
public:
    GeometryResult buildAcceleration(OptixDeviceContext context, ASArena &arena);
};

}
