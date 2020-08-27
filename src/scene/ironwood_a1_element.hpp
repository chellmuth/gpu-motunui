#pragma once

#include <optix.h>

#include "moana/scene/as_arena.hpp"
#include "moana/scene/types.hpp"
#include "scene/element.hpp"

namespace moana {

class IronwoodA1Element : public Element {
public:
    GeometryResult buildAcceleration(OptixDeviceContext context, ASArena &arena) override;
};

}
