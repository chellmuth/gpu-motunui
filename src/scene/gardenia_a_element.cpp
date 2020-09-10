#include "scene/gardenia_a_element.hpp"

namespace moana {

GardeniaAElement::GardeniaAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isGardeniaA";

    m_mtlLookup = {
        "bark",
        "barkSimple",
        "instances",
        "leaves",
    };

    m_materialOffset = 48;

    m_baseObjs = {
        moanaRoot + "/island/obj/isGardeniaA/isGardeniaA.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0001_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0002_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0003_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0004_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0005_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0006_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0007_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0008_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardenia0009_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardeniaflw0001_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardeniaflw0002_mod.obj",
        moanaRoot + "/island/obj/isGardeniaA/archives/archivegardeniaflw0003_mod.obj",
    };

    m_elementInstancesBinPaths = {
        "../scene/isGardeniaA.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isGardeniaA_xgBonsai--archivegardenia0008_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardenia0006_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardenia0007_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardenia0002_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardenia0003_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardenia0005_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardeniaflw0001_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardenia0001_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardenia0004_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardenia0009_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardeniaflw0002_mod.bin", "../scene/isGardeniaA_xgBonsai--archivegardeniaflw0003_mod.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {7, 5, 6, 1, 2, 4, 9, 0, 3, 8, 10, 11},
    };

    m_curveBinPathsByElementInstance = {
        {},
    };

    m_curveMtlIndicesByElementInstance = {
        {},
    };

}




}
