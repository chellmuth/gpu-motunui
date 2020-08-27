#include "scene/ocean_element.hpp"

namespace moana {

OceanElement::OceanElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "osOcean";

    m_baseObjs = {
        moanaRoot + "/island/obj/osOcean/osOcean.obj",
    };

    m_objArchivePaths = {

    };

    m_elementInstancesBinPaths = {
        "../scene/osOcean.bin",
    };

    m_primitiveInstancesBinPaths = {
        {},
    };

    m_primitiveInstancesHandleIndices = {
        {},
    };

    m_curveBinPathsByElementInstance = {
        {},
    };

    }

}
