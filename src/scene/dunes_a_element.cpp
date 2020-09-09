#include "scene/dunes_a_element.hpp"

namespace moana {

DunesAElement::DunesAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isDunesA";

    m_mtlLookup = {
        "archiveHibiscusFlower",
        "archiveLeaflet",
        "archiveMuskFern",
        "archivePalm",
        "base",
        "shoots",
        "soil",
        "underDunes",
        "xgDebris",
        "xgRoots",
        "xgShootRoots",
    };

    m_materialOffset = 26;

    m_baseObjs = {
        moanaRoot + "/island/obj/isDunesA/isDunesA.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archivepineneedle0001_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archivepineneedle0002_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archivepineneedle0003_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archiveseedpoda_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archiveseedpodb_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0002_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0003_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0004_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0005_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0006_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0007_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0008_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0009_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0001_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0002_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0003_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0004_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0005_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0006_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0007_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0008_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0009_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0010_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0011_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0012_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0013_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0014_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0123_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0124_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0125_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0126_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0127_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archivePalmdead0004_mod.obj",
    };

    m_archivePrimitiveIndexOffsets = {
        0, 0, 0, 0, 0, 0, 3888, 7776, 11664, 15552, 23328, 27216, 31104, 0, 3364, 6728, 10092, 13456, 16820, 20184, 23548, 26912, 30276, 33640, 37004, 40368, 43732, 0, 112, 208, 312, 416, 0
    };

    m_baseObjPrimitiveIndexOffsets = {
        0
    };

    m_elementInstancesBinPaths = {
        "../scene/isDunesA.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isDunesA_xgPalmDebris--xgPalmDebris_archivePalmdead0004_mod.bin", "../scene/isDunesA_xgPalmDebris--xgPalmDebris_archiveLeaflet0124_geo.bin", "../scene/isDunesA_xgPalmDebris--xgPalmDebris_archiveLeaflet0126_geo.bin", "../scene/isDunesA_xgPalmDebris--xgPalmDebris_archiveLeaflet0123_geo.bin", "../scene/isDunesA_xgPalmDebris--xgPalmDebris_archiveLeaflet0127_geo.bin", "../scene/isDunesA_xgPalmDebris--xgPalmDebris_archiveLeaflet0125_geo.bin", "../scene/isDunesA_xgDebris--xgDebris_archiveseedpodb_mod.bin", "../scene/isDunesA_xgDebris--xgDebris_archiveseedpoda_mod.bin", "../scene/isDunesA_xgDebris--xgDebris_archivepineneedle0003_mod.bin", "../scene/isDunesA_xgDebris--xgDebris_archivepineneedle0002_mod.bin", "../scene/isDunesA_xgDebris--xgDebris_archivepineneedle0001_mod.bin", "../scene/isDunesA_xgHibiscusFlower--xgHibiscusFlower_archiveHibiscusFlower0009_mod.bin", "../scene/isDunesA_xgHibiscusFlower--xgHibiscusFlower_archiveHibiscusFlower0005_mod.bin", "../scene/isDunesA_xgHibiscusFlower--xgHibiscusFlower_archiveHibiscusFlower0008_mod.bin", "../scene/isDunesA_xgHibiscusFlower--xgHibiscusFlower_archiveHibiscusFlower0004_mod.bin", "../scene/isDunesA_xgHibiscusFlower--xgHibiscusFlower_archiveHibiscusFlower0007_mod.bin", "../scene/isDunesA_xgHibiscusFlower--xgHibiscusFlower_archiveHibiscusFlower0006_mod.bin", "../scene/isDunesA_xgHibiscusFlower--xgHibiscusFlower_archiveHibiscusFlower0003_mod.bin", "../scene/isDunesA_xgHibiscusFlower--xgHibiscusFlower_archiveHibiscusFlower0002_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0011_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0002_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0012_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0004_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0005_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0010_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0006_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0008_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0009_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0007_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0003_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0013_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0001_mod.bin", "../scene/isDunesA_xgMuskFern--xgMuskFern_fern0014_mod.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {32, 28, 30, 27, 31, 29, 4, 3, 2, 1, 0, 12, 8, 11, 7, 10, 9, 6, 5, 23, 14, 24, 16, 17, 22, 18, 20, 21, 19, 15, 25, 13, 26},
    };

    m_curveBinPathsByElementInstance = {
        {"../scene/curves__isDunesA_xgRoots.bin", "../scene/curves__isDunesA_xgShootRoots.bin"},
    };

    m_curveMtlIndicesByElementInstance = {
        {9, 10},
    };

    }

}
