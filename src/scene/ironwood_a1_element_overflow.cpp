#include "scene/ironwood_a1_element_overflow.hpp"

namespace moana {

IronwoodA1ElementOverflow::IronwoodA1ElementOverflow()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isIronwoodA1";

    m_mtlLookup = {
        "archive_bark",
        "archive_pineNeedles",
        "archive_seedPod",
        "bark",
        "barkSimple",
        "leaves",
    };

    m_materialOffset = 60;

    m_baseObjs = {
        "../scene/isIronwoodA1-2.obj",
    };

    m_objArchivePaths = {
    };

    m_archivePrimitiveIndexOffsets = {
    };

    m_baseObjPrimitiveIndexOffsets = {
    };

    m_elementInstancesBinPaths = {
        "../scene/isIronwoodA1.bin",
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
