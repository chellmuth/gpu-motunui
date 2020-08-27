#include "scene/bay_cedar_a1_element.hpp"

namespace moana {

BayCedarA1Element::BayCedarA1Element()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isBayCedarA1/isBayCedarA1.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isBayCedarA1/archives/archivebaycedar0001_mod.obj",
    };

    m_binPaths = {
        "../scene/isBayCedarA1-archivebaycedar0001_mod.bin",
    };

    m_elementInstancesBinPath = "../scene/isBayCedarA1-root.bin";
}

}
