#include "scene/lava_rocks_element.hpp"

namespace moana {

LavaRocksElement::LavaRocksElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isLavaRocks";

    m_sbtOffset = 75;

    m_mtlLookup = {
        "default",
        "volcanicRock",
    };

    m_baseObjs = {
        moanaRoot + "/island/obj/isLavaRocks/isLavaRocks.obj",
        moanaRoot + "/island/obj/isLavaRocks/isLavaRocks1.obj",
    };

    m_objArchivePaths = {

    };

    m_archivePrimitiveIndexOffsets = {
        
    };

    m_baseObjPrimitiveIndexOffsets = {
        15294, 0
    };

    m_elementInstancesBinPaths = {
        "../scene/isLavaRocks.bin",
        "../scene/isLavaRocks1.bin",
    };

    m_primitiveInstancesBinPaths = {
        {},
        {},
    };

    m_primitiveInstancesHandleIndices = {
        {},
        {},
    };

    m_curveBinPathsByElementInstance = {
        {},
        {},
    };

    m_curveMtlIndicesByElementInstance = {
        {},
        {},
    };

    }

}
