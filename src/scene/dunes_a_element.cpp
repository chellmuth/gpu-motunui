#include "scene/dunes_a_element.hpp"

namespace moana {

DunesAElement::DunesAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isDunesA/isDunesA.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archivePalmdead0004_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0124_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0126_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0123_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0127_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0125_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archiveseedpodb_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archiveseedpoda_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archivepineneedle0003_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archivepineneedle0002_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archivepineneedle0001_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0009_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0005_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0008_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0004_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0007_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0006_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0003_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0002_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0011_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0002_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0012_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0004_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0005_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0010_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0006_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0008_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0009_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0007_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0003_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0013_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0001_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0014_mod.obj",
    };

    m_binPaths = {
        "../scene/isDunesA-xgPalmDebris_archivePalmdead0004_mod.bin",
        "../scene/isDunesA-xgPalmDebris_archiveLeaflet0124_geo.bin",
        "../scene/isDunesA-xgPalmDebris_archiveLeaflet0126_geo.bin",
        "../scene/isDunesA-xgPalmDebris_archiveLeaflet0123_geo.bin",
        "../scene/isDunesA-xgPalmDebris_archiveLeaflet0127_geo.bin",
        "../scene/isDunesA-xgPalmDebris_archiveLeaflet0125_geo.bin",
        "../scene/isDunesA-xgDebris_archiveseedpodb_mod.bin",
        "../scene/isDunesA-xgDebris_archiveseedpoda_mod.bin",
        "../scene/isDunesA-xgDebris_archivepineneedle0003_mod.bin",
        "../scene/isDunesA-xgDebris_archivepineneedle0002_mod.bin",
        "../scene/isDunesA-xgDebris_archivepineneedle0001_mod.bin",
        "../scene/isDunesA-xgHibiscusFlower_archiveHibiscusFlower0009_mod.bin",
        "../scene/isDunesA-xgHibiscusFlower_archiveHibiscusFlower0005_mod.bin",
        "../scene/isDunesA-xgHibiscusFlower_archiveHibiscusFlower0008_mod.bin",
        "../scene/isDunesA-xgHibiscusFlower_archiveHibiscusFlower0004_mod.bin",
        "../scene/isDunesA-xgHibiscusFlower_archiveHibiscusFlower0007_mod.bin",
        "../scene/isDunesA-xgHibiscusFlower_archiveHibiscusFlower0006_mod.bin",
        "../scene/isDunesA-xgHibiscusFlower_archiveHibiscusFlower0003_mod.bin",
        "../scene/isDunesA-xgHibiscusFlower_archiveHibiscusFlower0002_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0011_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0002_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0012_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0004_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0005_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0010_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0006_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0008_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0009_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0007_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0003_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0013_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0001_mod.bin",
        "../scene/isDunesA-xgMuskFern_fern0014_mod.bin",
    };

    m_hasElementInstances = true;
    m_elementInstancesBinPath = "../scene/isDunesA-root.bin";
}

}
