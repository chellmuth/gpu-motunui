#include "scene/beach_element.hpp"

namespace moana {

BeachElement::BeachElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isBeach";

    m_sbtOffset = 4;

    m_mtlLookup = {
        "archiveCoralPebbles",
        "archiveFibers",
        "archiveFlowerHibiscus",
        "archiveGroundCoverThin",
        "archiveLeaflet",
        "archivePebbles",
        "archiveSeaweed",
        "archiveShells",
        "archiveShellsSmall",
        "archiveVolcanicRock",
        "sand",
        "xgBonBabyGardeniaBranch",
        "xgBonBabyGardeniaLeaves",
        "xgGrass",
    };

    m_baseObjs = {
        moanaRoot + "/island/obj/isBeach/isBeach.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isBeach/archives/xgFibers_archivepineneedle0001_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgFibers_archivepineneedle0002_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgFibers_archivepineneedle0003_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgFibers_archiveseedpodb_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveCoral0001_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveCoral0002_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveCoral0003_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveCoral0004_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveCoral0005_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveCoral0006_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveCoral0007_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveCoral0008_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveCoral0009_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveRock0001_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveRock0002_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveRock0003_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveRock0004_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveRock0005_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveRock0006_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveRock0007_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveShell0001_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveShell0002_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveShell0003_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveShell0004_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveShell0005_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveShell0006_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveShell0007_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgGroundCover_archiveShell0008_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgHibiscus_archiveHibiscusFlower0002_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgHibiscus_archiveHibiscusFlower0003_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgHibiscus_archiveHibiscusFlower0004_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgHibiscus_archiveHibiscusFlower0005_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgHibiscus_archiveHibiscusFlower0006_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgHibiscus_archiveHibiscusFlower0007_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgHibiscus_archiveHibiscusFlower0008_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgHibiscus_archiveHibiscusFlower0009_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPalmDebris_archiveLeaflet0123_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPalmDebris_archiveLeaflet0124_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPalmDebris_archiveLeaflet0125_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPalmDebris_archiveLeaflet0126_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPalmDebris_archiveLeaflet0127_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveCoral0001_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveCoral0002_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveCoral0003_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveCoral0004_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveCoral0005_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveCoral0006_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveCoral0007_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveCoral0008_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveCoral0009_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveRock0001_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveRock0002_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveRock0003_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveRock0004_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveRock0005_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveRock0006_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgPebbles_archiveRock0007_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgSeaweed_archiveSeaweed0001_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgSeaweed_archiveSeaweed0002_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgSeaweed_archiveSeaweed0003_mod.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgSeaweed_archiveSeaweed0063_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgSeaweed_archiveSeaweed0064_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgSeaweed_archiveSeaweed0065_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShellsSmall_archiveShell0001_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShellsSmall_archiveShell0002_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShellsSmall_archiveShell0003_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShellsSmall_archiveShell0004_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShellsSmall_archiveShell0005_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShellsSmall_archiveShell0006_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShellsSmall_archiveShell0007_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShellsSmall_archiveShell0008_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShells_archiveShell0001_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShells_archiveShell0002_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShells_archiveShell0003_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShells_archiveShell0004_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShells_archiveShell0005_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShells_archiveShell0006_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShells_archiveShell0007_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgShells_archiveShell0008_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgStones_archiveRock0001_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgStones_archiveRock0002_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgStones_archiveRock0003_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgStones_archiveRock0004_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgStones_archiveRock0005_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgStones_archiveRock0006_geo.obj",
        moanaRoot + "/island/obj/isBeach/archives/xgStones_archiveRock0007_geo.obj",
    };

    m_archivePrimitiveIndexOffsets = {
        0, 0, 0, 0, 0, 360, 440, 520, 960, 1320, 1416, 1600, 1712, 0, 160, 232, 336, 372, 452, 548, 600, 808, 1016, 1224, 1432, 1600, 1744, 1848, 0, 3888, 0, 7776, 0, 11664, 15552, 0, 0, 112, 208, 312, 416, 0, 360, 440, 520, 960, 1320, 1416, 1600, 1712, 0, 160, 232, 336, 372, 452, 548, 0, 30944, 50064, 59328, 60296, 61264, 0, 208, 416, 624, 832, 1000, 1144, 1248, 0, 208, 416, 624, 832, 1000, 1144, 1248, 0, 72, 144, 248, 284, 364, 460
    };

    m_baseObjPrimitiveIndexOffsets = {
        0
    };

    m_elementInstancesBinPaths = {
        "../scene/isBeach.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isBeach_xgFibers--xgFibers_archiveseedpodb_mod.bin", "../scene/isBeach_xgFibers--xgFibers_archivepineneedle0003_mod.bin", "../scene/isBeach_xgFibers--xgFibers_archivepineneedle0002_mod.bin", "../scene/isBeach_xgFibers--xgFibers_archivepineneedle0001_mod.bin", "../scene/isBeach_xgHibiscus--xgHibiscus_archiveHibiscusFlower0006_mod.bin", "../scene/isBeach_xgHibiscus--xgHibiscus_archiveHibiscusFlower0007_mod.bin", "../scene/isBeach_xgHibiscus--xgHibiscus_archiveHibiscusFlower0002_mod.bin", "../scene/isBeach_xgHibiscus--xgHibiscus_archiveHibiscusFlower0003_mod.bin", "../scene/isBeach_xgHibiscus--xgHibiscus_archiveHibiscusFlower0004_mod.bin", "../scene/isBeach_xgHibiscus--xgHibiscus_archiveHibiscusFlower0005_mod.bin", "../scene/isBeach_xgHibiscus--xgHibiscus_archiveHibiscusFlower0008_mod.bin", "../scene/isBeach_xgHibiscus--xgHibiscus_archiveHibiscusFlower0009_mod.bin", "../scene/isBeach_xgStones--xgStones_archiveRock0001_geo.bin", "../scene/isBeach_xgStones--xgStones_archiveRock0004_geo.bin", "../scene/isBeach_xgStones--xgStones_archiveRock0006_geo.bin", "../scene/isBeach_xgStones--xgStones_archiveRock0005_geo.bin", "../scene/isBeach_xgStones--xgStones_archiveRock0007_geo.bin", "../scene/isBeach_xgStones--xgStones_archiveRock0003_geo.bin", "../scene/isBeach_xgStones--xgStones_archiveRock0002_geo.bin", "../scene/isBeach_xgShells--xgShells_archiveShell0003_geo.bin", "../scene/isBeach_xgShells--xgShells_archiveShell0001_geo.bin", "../scene/isBeach_xgShells--xgShells_archiveShell0007_geo.bin", "../scene/isBeach_xgShells--xgShells_archiveShell0006_geo.bin", "../scene/isBeach_xgShells--xgShells_archiveShell0004_geo.bin", "../scene/isBeach_xgShells--xgShells_archiveShell0002_geo.bin", "../scene/isBeach_xgShells--xgShells_archiveShell0005_geo.bin", "../scene/isBeach_xgShells--xgShells_archiveShell0008_geo.bin", "../scene/isBeach_xgSeaweed--xgSeaweed_archiveSeaweed0063_geo.bin", "../scene/isBeach_xgSeaweed--xgSeaweed_archiveSeaweed0064_geo.bin", "../scene/isBeach_xgSeaweed--xgSeaweed_archiveSeaweed0001_mod.bin", "../scene/isBeach_xgSeaweed--xgSeaweed_archiveSeaweed0065_geo.bin", "../scene/isBeach_xgSeaweed--xgSeaweed_archiveSeaweed0003_mod.bin", "../scene/isBeach_xgSeaweed--xgSeaweed_archiveSeaweed0002_mod.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveCoral0002_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveCoral0003_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveCoral0004_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveCoral0001_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveRock0002_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveRock0004_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveCoral0009_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveRock0005_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveRock0007_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveCoral0006_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveCoral0007_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveCoral0008_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveCoral0005_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveRock0001_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveRock0003_geo.bin", "../scene/isBeach_xgPebbles--xgPebbles_archiveRock0006_geo.bin", "../scene/isBeach_xgShellsSmall--xgShellsSmall_archiveShell0004_geo.bin", "../scene/isBeach_xgShellsSmall--xgShellsSmall_archiveShell0002_geo.bin", "../scene/isBeach_xgShellsSmall--xgShellsSmall_archiveShell0008_geo.bin", "../scene/isBeach_xgShellsSmall--xgShellsSmall_archiveShell0001_geo.bin", "../scene/isBeach_xgShellsSmall--xgShellsSmall_archiveShell0006_geo.bin", "../scene/isBeach_xgShellsSmall--xgShellsSmall_archiveShell0003_geo.bin", "../scene/isBeach_xgShellsSmall--xgShellsSmall_archiveShell0005_geo.bin", "../scene/isBeach_xgShellsSmall--xgShellsSmall_archiveShell0007_geo.bin", "../scene/isBeach_xgPalmDebris--xgPalmDebris_archiveLeaflet0127_geo.bin", "../scene/isBeach_xgPalmDebris--xgPalmDebris_archiveLeaflet0125_geo.bin", "../scene/isBeach_xgPalmDebris--xgPalmDebris_archiveLeaflet0126_geo.bin", "../scene/isBeach_xgPalmDebris--xgPalmDebris_archiveLeaflet0123_geo.bin", "../scene/isBeach_xgPalmDebris--xgPalmDebris_archiveLeaflet0124_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveRock0002_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveCoral0004_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveShell0004_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveShell0006_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveCoral0002_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveCoral0001_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveCoral0007_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveShell0001_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveShell0003_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveCoral0006_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveCoral0008_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveCoral0009_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveShell0007_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveCoral0005_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveShell0008_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveRock0007_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveCoral0003_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveRock0001_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveShell0005_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveShell0002_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveRock0004_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveRock0006_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveRock0005_geo.bin", "../scene/isBeach_xgGroundCover--xgGroundCover_archiveRock0003_geo.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {3, 2, 1, 0, 32, 33, 28, 29, 30, 31, 34, 35, 79, 82, 84, 83, 85, 81, 80, 73, 71, 77, 76, 74, 72, 75, 78, 60, 61, 57, 62, 59, 58, 42, 43, 44, 41, 51, 53, 49, 54, 56, 46, 47, 48, 45, 50, 52, 55, 66, 64, 70, 63, 68, 65, 67, 69, 40, 38, 39, 36, 37, 14, 7, 23, 25, 5, 4, 10, 20, 22, 9, 11, 12, 26, 8, 27, 19, 6, 13, 24, 21, 16, 18, 17, 15},
    };

    m_curveBinPathsByElementInstance = {
        {"../scene/curves__isBeach_xgGrass.bin"},
    };

    m_curveMtlIndicesByElementInstance = {
        {13},
    };

    }

}
