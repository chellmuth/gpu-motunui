#include "scene/palm_dead_element.hpp"

namespace moana {

PalmDeadElement::PalmDeadElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isPalmDead";

    m_baseObjs = {
        moanaRoot + "/island/obj/isPalmDead/isPalmDead.obj",
    };

    m_objArchivePaths = {

    };

    m_elementInstancesBinPaths = {
        "../scene/isPalmDead.bin",
    };

    m_primitiveInstancesBinPaths = {
        {},
    };

    m_primitiveInstancesHandleIndices = {
        {},
    };

    }

}
