#pragma once

#include <optix.h>

#include "moana/parsers/obj_parser.hpp"
#include "scene/types.hpp"

namespace moana { namespace GAS {

GASInfo gasInfoFromObjResult(OptixDeviceContext context, const ObjResult &model);

} }
