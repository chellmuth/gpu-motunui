#include "scene/ironwood_b_element.hpp"

namespace moana {

IronwoodBElement::IronwoodBElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isIronwoodB/isIronwoodB.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isIronwoodB/archives/archiveseedpodb_mod.obj",
    };

    m_binPaths = {
        "../scene/isIronwoodB-archiveseedpodb_mod.bin",
    };

    m_hasElementInstances = false;
    m_elementInstancesBinPath = "";
}

}
