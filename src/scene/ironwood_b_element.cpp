#include "scene/ironwood_b_element.hpp"

namespace moana {

IronwoodBElement::IronwoodBElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isIronwoodB";

    m_sbtOffset = 66;

    m_mtlLookup = {
        "archive_bark",
        "archive_pineNeedles",
        "archive_seedPod",
        "bark",
        "barkSimple",
        "leaves",
    };

    m_baseObjs = {
        moanaRoot + "/island/obj/isIronwoodB/isIronwoodB.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isIronwoodB/archives/archiveseedpodb_mod.obj",
    };

    m_elementInstancesBinPaths = {
        "../scene/isIronwoodB.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isIronwoodB_xgBonsai--archiveseedpodb_mod.bin"},
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

}
