#pragma once

#include <vector>

#include <optix.h>

#include "moana/scene/as_arena.hpp"
#include "moana/scene/types.hpp"

namespace moana { namespace Container {

std::vector<GeometryResult> createGeometryResults(
    OptixDeviceContext context,
    ASArena &arena
);

} }
