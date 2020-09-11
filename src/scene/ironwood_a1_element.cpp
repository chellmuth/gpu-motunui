#include "scene/ironwood_a1_element.hpp"

namespace moana {

IronwoodA1Element::IronwoodA1Element()
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
        "../scene/isIronwoodA1-1.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isIronwoodA1/archives/archiveseedpodb_mod.obj",
    };

    m_elementInstancesBinPaths = {
        "../scene/isIronwoodA1.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isIronwoodA1_xgBonsai--archiveseedpodb_mod.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {0},
    };

    m_curveBinPathsByElementInstance = {
        {},
    };

    m_curveMtlIndicesByElementInstance = {
        {},
    };


}


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
