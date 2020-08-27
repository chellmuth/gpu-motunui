#include "scene/hibiscus_young_element.hpp"

namespace moana {

HibiscusYoungElement::HibiscusYoungElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isHibiscusYoung/isHibiscusYoung.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0003_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0002_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0001_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusFlower0001_mod.obj",
    };

    m_binPaths = {
        "../scene/isHibiscusYoung-archiveHibiscusLeaf0003_mod.bin",
        "../scene/isHibiscusYoung-archiveHibiscusLeaf0002_mod.bin",
        "../scene/isHibiscusYoung-archiveHibiscusLeaf0001_mod.bin",
        "../scene/isHibiscusYoung-archiveHibiscusFlower0001_mod.bin",
    };

    m_elementInstancesBinPath = "../scene/isHibiscusYoung-root.bin";
}

}
