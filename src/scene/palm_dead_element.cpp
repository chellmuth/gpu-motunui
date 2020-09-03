#include "scene/palm_dead_element.hpp"

namespace moana {

PalmDeadElement::PalmDeadElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isPalmDead";

    m_mtlLookup = {
        "roots",
        "trunk",
    };

    m_materialOffset = 95;

    m_baseObjs = {
        moanaRoot + "/island/obj/isPalmDead/isPalmDead.obj",
    };

    m_objArchivePaths = {

    };

    m_archivePrimitiveIndexOffsets = {
        
    };

    m_baseObjPrimitiveIndexOffsets = {
        0
    };

    m_elementInstancesBinPaths = {
        "../scene/isPalmDead.bin",
    };

    m_primitiveInstancesBinPaths = {
        {},
    };

    m_primitiveInstancesHandleIndices = {
        {},
    };

    m_curveBinPathsByElementInstance = {
        {},
    };

    m_curveMtlIndicesByElementInstance = {
        {},
    };

    }

}
