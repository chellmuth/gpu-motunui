#include "scene/naupaka_a_element.hpp"

namespace moana {

NaupakaAElement::NaupakaAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isNaupakaA/isNaupakaA.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isNaupakaA/archives/xgBonsai_isNaupakaBon_bon_hero_ALL.obj",
    };

    m_binPaths = {
        "../scene/isNaupakaA-xgBonsai_isNaupakaBon_bon_hero_ALL.bin",
    };

    m_elementInstancesBinPath = "../scene/isNaupakaA-root.bin";
}

}
