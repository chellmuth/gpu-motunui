#pragma once

#include <string>
#include <vector>

#include <optix.h>

#include "moana/scene/as_arena.hpp"
#include "moana/scene/types.hpp"

namespace moana {

class Element {
public:
    virtual GeometryResult buildAcceleration(
        OptixDeviceContext context,
        ASArena &arena
    );

protected:
    std::string m_baseObj;
    std::vector<std::string> m_objPaths;
    std::vector<std::string> m_binPaths;

    std::string m_elementInstancesBinPath;
};

}
