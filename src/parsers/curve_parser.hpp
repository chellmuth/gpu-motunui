#pragma once

#include <string>

#include <optix.h>

#include "moana/scene/as_arena.hpp"

namespace moana {

class Curve {
public:
    Curve(const std::string &filename);

    OptixTraversableHandle gasFromCurve(
        OptixDeviceContext context,
        ASArena &arena
    );

private:
    std::string m_filename;
};

}
