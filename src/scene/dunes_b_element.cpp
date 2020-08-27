#include "scene/dunes_b_element.hpp"

namespace moana {

DunesBElement::DunesBElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isDunesB/isDunesB.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isIronwoodA1/isIronwoodA1_variantA_lo.obj",
        moanaRoot + "/island/obj/isIronwoodA1/isIronwoodA1_variantB_lo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgPandanus_isPandanusAlo_base.obj",
        moanaRoot + "/island/obj/isIronwoodA1/isIronwoodA1_variantA_lo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0001_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0012_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0003_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0002_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0014_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0007_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0009_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0004_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0006_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0010_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0005_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0013_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0011_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0008_geo.obj",
    };

    m_binPaths = {
        "../scene/isDunesB-isIronwoodA1_variantA_lo.bin",
        "../scene/isDunesB-isIronwoodA1_variantB_lo.bin",
        "../scene/isDunesB-xgPandanus_isPandanusAlo_base.bin",
        "../scene/isDunesB-isIronwoodA1_variantA_lo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0001_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0012_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0003_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0002_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0014_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0007_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0009_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0004_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0006_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0010_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0005_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0013_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0011_geo.bin",
        "../scene/isDunesB-xgRoots_archiveroot0008_geo.bin",
    };

    m_elementInstancesBinPath = "../scene/isDunesB-root.bin";
}

}
