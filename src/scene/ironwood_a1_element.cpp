#include "scene/ironwood_a1_element.hpp"

namespace moana {

IronwoodA1Element::IronwoodA1Element()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isIronwoodA1";

    m_baseObjs = {
        moanaRoot + "/island/obj/isIronwoodA1/isIronwoodA1.obj",
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

    }

}
