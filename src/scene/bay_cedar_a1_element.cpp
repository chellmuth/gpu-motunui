#include "scene/bay_cedar_a1_element.hpp"

namespace moana {

BayCedarA1Element::BayCedarA1Element()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isBayCedarA1";

    m_baseObjs = {
        moanaRoot + "/island/obj/isBayCedarA1/isBayCedarA1.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isBayCedarA1/archives/archivebaycedar0001_mod.obj",
    };

    m_elementInstancesBinPaths = {
        "../scene/isBayCedarA1.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isBayCedarA1_xgBonsai--archivebaycedar0001_mod.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {0},
    };

    }

}
