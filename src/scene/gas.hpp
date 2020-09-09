#pragma once

#include <optix.h>

#include "moana/parsers/obj_parser.hpp"
#include "moana/scene/as_arena.hpp"

namespace moana { namespace GAS {

OptixTraversableHandle gasInfoFromMeshRecords(
    OptixDeviceContext context,
    ASArena &arena,
    const std::vector<MeshRecord> &records,
    int primitiveIndexOffset = 0
);

} }
