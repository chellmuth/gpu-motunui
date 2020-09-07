#pragma once

#include "moana/core/frame.hpp"
#include "moana/core/vec3.hpp"

namespace moana {

struct BSDFSampleRecord {
    float3 point;
    Vec3 wiLocal;
    Vec3 normal;
    Frame frame;
};

}
