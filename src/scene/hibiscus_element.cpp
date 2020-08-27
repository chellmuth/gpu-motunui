#include "scene/hibiscus_element.hpp"

namespace moana {

HibiscusElement::HibiscusElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isHibiscus/isHibiscus.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0001_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusFlower0001_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0003_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0002_mod.obj"
    };

    m_binPaths = {
        "../scene/isHibiscus-archiveHibiscusLeaf0001_mod.bin",
        "../scene/isHibiscus-archiveHibiscusFlower0001_mod.bin",
        "../scene/isHibiscus-archiveHibiscusLeaf0002_mod.bin",
        "../scene/isHibiscus-archiveHibiscusLeaf0003_mod.bin",
    };

    m_hasElementInstances = true;
    m_elementInstancesBinPath = "../scene/isHibiscus-root.bin";
}

}
