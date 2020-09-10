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
        ASArena &arena,
        int elementSBTOffset
    );

protected:
    std::string m_elementName;

    // Element geometries. See isPalmRig for example where every element instance
    // has unique geometry
    std::vector<std::string> m_baseObjs;

    // Transform bin paths for each element instance with unique geometry
    std::vector<std::string> m_elementInstancesBinPaths;

    // List of all .obj files used in archives
    std::vector<std::string> m_objArchivePaths;

    // Transforms for each archive (compiled from their json digest)
    std::vector<std::vector<std::string> > m_primitiveInstancesBinPaths;

    // Indices that are used to lookup handles (built earlier in the pipeline)
    // to primitive GAS's
    std::vector<std::vector<int> > m_primitiveInstancesHandleIndices;

    // Path to curve binaries for each element instance
    std::vector<std::vector<std::string> > m_curveBinPathsByElementInstance;

    // List ordering how the element's materials map to the SBT
    std::vector<std::string> m_mtlLookup;

    // Curve materials aren't defined in the obj like archives, store them here
    std::vector<std::vector<int> > m_curveMtlIndicesByElementInstance;

    int m_materialOffset = 0;
};

}
