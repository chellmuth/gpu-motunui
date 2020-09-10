#include "scene/lava_rocks_element.hpp"

namespace moana {

LavaRocksElement::LavaRocksElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isLavaRocks";

    m_mtlLookup = {
        "default",
        "volcanicRock",
    };

    m_materialOffset = 75;

    m_baseObjs = {
        moanaRoot + "/island/obj/isLavaRocks/isLavaRocks.obj",
        moanaRoot + "/island/obj/isLavaRocks/isLavaRocks1.obj",
    };

    m_objArchivePaths = {

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
