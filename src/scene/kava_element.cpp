#include "scene/kava_element.hpp"

namespace moana {

KavaElement::KavaElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isKava";

    m_mtlLookup = {
        "bark",
        "barkSimple",
        "leaves",
    };

    m_materialOffset = 72;

    m_baseObjs = {
        moanaRoot + "/island/obj/isKava/isKava.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isKava/archives/archive_kava0001_mod.obj",
    };

    m_archivePrimitiveIndexOffsets = {
        0
    };

    m_baseObjPrimitiveIndexOffsets = {
        0
    };

    m_elementInstancesBinPaths = {
        "../scene/isKava.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isKava_xgBonsai--archive_kava0001_mod.bin"},
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
