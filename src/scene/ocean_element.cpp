#include "scene/ocean_element.hpp"

namespace moana {

OceanElement::OceanElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/osOcean/osOcean.obj";

    m_objPaths = {

    };

    m_binPaths = {

    };

    m_elementInstancesBinPath = "../scene/osOcean-root.bin";
}

}
