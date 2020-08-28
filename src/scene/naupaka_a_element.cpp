#include "scene/naupaka_a_element.hpp"

namespace moana {

NaupakaAElement::NaupakaAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isNaupakaA";

    m_sbtOffset = 92;

    m_mtlLookup = {
        "hidden",
        "leaves",
        "stem",
    };

    m_baseObjs = {
        moanaRoot + "/island/obj/isNaupakaA/isNaupakaA.obj",
        moanaRoot + "/island/obj/isNaupakaA/isNaupakaA1.obj",
        moanaRoot + "/island/obj/isNaupakaA/isNaupakaA3.obj",
        moanaRoot + "/island/obj/isNaupakaA/isNaupakaA2.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isNaupakaA/archives/xgBonsai_isNaupakaBon_bon_hero_ALL.obj",
    };

    m_elementInstancesBinPaths = {
        "../scene/isNaupakaA.bin",
        "../scene/isNaupakaA1.bin",
        "../scene/isNaupakaA3.bin",
        "../scene/isNaupakaA2.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isNaupakaA_xgBonsai--xgBonsai_isNaupakaBon_bon_hero_ALL.bin"},
        {"../scene/isNaupakaA1_xgBonsai--xgBonsai_isNaupakaBon_bon_hero_ALL.bin"},
        {"../scene/isNaupakaA3_xgBonsai--xgBonsai_isNaupakaBon_bon_hero_ALL.bin"},
        {"../scene/isNaupakaA2_xgBonsai--xgBonsai_isNaupakaBon_bon_hero_ALL.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {0},
        {0},
        {0},
        {0},
    };

    m_curveBinPathsByElementInstance = {
        {},
        {},
        {},
        {},
    };

    m_curveMtlIndicesByElementInstance = {
        {},
        {},
        {},
        {},
    };

    }

}
