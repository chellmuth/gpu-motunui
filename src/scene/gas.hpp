#pragma once

#include <optix.h>

#include "moana/parsers/obj_parser.hpp"
#include "moana/scene/as_arena.hpp"

namespace moana { namespace GAS {

OptixTraversableHandle gasInfoFromObjResult(
    OptixDeviceContext context,
    ASArena &arena,
    const ObjResult &model
);

} }
