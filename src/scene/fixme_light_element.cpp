#include "scene/fixme_light_element.hpp"

namespace moana {

FixmeLightElement::FixmeLightElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "FixmeLight";

    m_mtlLookup = {
        "fixmelight",
    };

    m_materialOffset = 104;

    m_baseObjs = {
        "../scene/fixme-light.obj",
    };

    m_objArchivePaths = {

    };

    m_elementInstancesBinPaths = {
        "../scene/fixme-transform.bin",
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

    m_curveMtlIndicesByElementInstance = {
        {},
    };


}




}
