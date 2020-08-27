#include "scene/ironwood_a1_element.hpp"

namespace moana {

IronwoodA1Element::IronwoodA1Element()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isIronwoodA1/isIronwoodA1.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isIronwoodA1/archives/archiveseedpodb_mod.obj",
    };

    m_binPaths = {
        "../scene/isIronwoodA1-archiveseedpodb_mod.bin",
    };

    m_hasElementInstances = true;
    m_elementInstancesBinPath = "../scene/isIronwoodA1-root.bin";
}

}
