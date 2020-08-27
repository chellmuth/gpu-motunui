#pragma once

#include <optix.h>

#include "moana/scene/as_arena.hpp"
#include "moana/scene/types.hpp"

namespace moana {

class Element {
public:
    virtual GeometryResult buildAcceleration(
        OptixDeviceContext context,
        ASArena &arena
    ) = 0;
};

}
