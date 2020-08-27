#include "scene/gardenia_a_element.hpp"

namespace moana {

GardeniaAElement::GardeniaAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isGardeniaA/isGardeniaA.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0008_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0006_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0007_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0002_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0003_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0005_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardeniaflw0001_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0001_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0004_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0009_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardeniaflw0002_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardeniaflw0003_mod.obj",
    };

    m_binPaths = {
        "../scene/isGardeniaA-archivegardenia0008_mod.bin",
        "../scene/isGardeniaA-archivegardenia0006_mod.bin",
        "../scene/isGardeniaA-archivegardenia0007_mod.bin",
        "../scene/isGardeniaA-archivegardenia0002_mod.bin",
        "../scene/isGardeniaA-archivegardenia0003_mod.bin",
        "../scene/isGardeniaA-archivegardenia0005_mod.bin",
        "../scene/isGardeniaA-archivegardeniaflw0001_mod.bin",
        "../scene/isGardeniaA-archivegardenia0001_mod.bin",
        "../scene/isGardeniaA-archivegardenia0004_mod.bin",
        "../scene/isGardeniaA-archivegardenia0009_mod.bin",
        "../scene/isGardeniaA-archivegardeniaflw0002_mod.bin",
        "../scene/isGardeniaA-archivegardeniaflw0003_mod.bin",
    };

    m_hasElementInstances = true;
    m_elementInstancesBinPath = "../scene/isGardeniaA-root.bin";
}

}
