#include "scene/kava_element.hpp"

namespace moana {

KavaElement::KavaElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isKava/isKava.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isKava/archives/archive_kava0001_mod.obj",
    };

    m_binPaths = {
        "../scene/isKava-archive_kava0001_mod.bin",
    };

    m_elementInstancesBinPath = "../scene/isKava-root.bin";
}

}
