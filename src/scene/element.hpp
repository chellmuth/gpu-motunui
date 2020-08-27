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
    std::string m_elementName;

    std::vector<std::string> m_baseObjs;
    std::vector<std::string> m_elementInstancesBinPaths;
    std::vector<std::string> m_objArchivePaths;

    std::vector<std::vector<std::string> > m_primitiveInstancesBinPaths;
    std::vector<std::vector<int> > m_primitiveInstancesHandleIndices;
    std::vector<std::vector<std::string> > m_curveBinPathsByElementInstance;

    std::string m_elementInstancesBinPath;
};

}
