#pragma once

#include <string>

#include <cuda_runtime.h>

#include "moana/core/vec3.hpp"

namespace moana {

class PtexTexture {
public:
    PtexTexture(const std::string &texturePath);
    Vec3 lookup(float2 uv, int faceIndex) const;

private:
    std::string m_texturePath;
};

}
