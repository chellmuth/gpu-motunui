#include "scene/pandanus_a_element.hpp"

namespace moana {

PandanusAElement::PandanusAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isPandanusA/isPandanusA.obj";

    m_objPaths = {

    };

    m_binPaths = {

    };

    m_hasElementInstances = true;
    m_elementInstancesBinPath = "../scene/isPandanusA-root.bin";
}

}
