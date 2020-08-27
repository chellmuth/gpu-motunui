#include "scene/pandanus_a_element.hpp"

namespace moana {

PandanusAElement::PandanusAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isPandanusA";

    m_baseObjs = {
        moanaRoot + "/island/obj/isPandanusA/isPandanusA.obj",
    };

    m_objArchivePaths = {

    };

    m_elementInstancesBinPaths = {
        "../scene/isPandanusA.bin",
    };

    m_primitiveInstancesBinPaths = {
        {},
    };

    m_primitiveInstancesHandleIndices = {
        {},
    };

    }

}
