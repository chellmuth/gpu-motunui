#pragma once

#include "moana/core/frame.hpp"
#include "moana/core/vec3.hpp"

namespace moana {

struct BSDFSampleRecord {
    bool isValid = false;
    float3 point;
    Vec3 wiLocal;
    Vec3 normal;
    Frame frame;
    float weight;
    bool isDelta;
};

}
