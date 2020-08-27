#include "scene/palm_dead_element.hpp"

namespace moana {

PalmDeadElement::PalmDeadElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isPalmDead/isPalmDead.obj";

    m_objPaths = {

    };

    m_binPaths = {

    };

    m_elementInstancesBinPath = "../scene/isPalmDead-root.bin";
}

}
