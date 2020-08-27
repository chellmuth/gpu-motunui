#include "scene/hibiscus_element.hpp"

namespace moana {

HibiscusElement::HibiscusElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isHibiscus";

    m_baseObjs = {
        moanaRoot + "/island/obj/isHibiscus/isHibiscus.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusFlower0001_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0001_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0002_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0003_mod.obj",
    };

    m_elementInstancesBinPaths = {
        "../scene/isHibiscus.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isHibiscus_xgBonsai--archiveHibiscusLeaf0001_mod.bin", "../scene/isHibiscus_xgBonsai--archiveHibiscusFlower0001_mod.bin", "../scene/isHibiscus_xgBonsai--archiveHibiscusLeaf0003_mod.bin", "../scene/isHibiscus_xgBonsai--archiveHibiscusLeaf0002_mod.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {1, 0, 3, 2},
    };

    }

}
