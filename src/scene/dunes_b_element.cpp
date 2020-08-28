#include "scene/dunes_b_element.hpp"

namespace moana {

DunesBElement::DunesBElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isDunesB";

    m_sbtOffset = 37;

    m_mtlLookup = {
        "isBayCedar_bark",
        "isBayCedar_barkSimple",
        "isBayCedar_leaves",
        "isIronwoodA_archive_bark",
        "isIronwoodA_barkSimple",
        "isIronwoodA_leaves",
        "isPandanus_leavesLower",
        "isPandanus_leavesSimple",
        "isPandanus_trunk",
        "soilSimple",
        "xgRoots",
    };

    m_baseObjs = {
        moanaRoot + "/island/obj/isDunesB/isDunesB.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isDunesB/archives/xgPandanus_isPandanusAlo_base.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0001_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0002_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0003_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0004_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0005_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0006_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0007_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0008_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0009_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0010_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0011_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0012_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0013_geo.obj",
        moanaRoot + "/island/obj/isDunesB/archives/xgRoots_archiveroot0014_geo.obj",
        moanaRoot + "/island/obj/isIronwoodA1/isIronwoodA1_variantA_lo.obj",
        moanaRoot + "/island/obj/isIronwoodA1/isIronwoodA1_variantB_lo.obj",
    };

    m_elementInstancesBinPaths = {
        "../scene/isDunesB.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isDunesB_xgTreeSkyLine--isIronwoodA1_variantA_lo.bin", "../scene/isDunesB_xgTreeSkyLine--isIronwoodA1_variantB_lo.bin", "../scene/isDunesB_xgPandanus--xgPandanus_isPandanusAlo_base.bin", "../scene/isDunesB_xgTreeSpecific--isIronwoodA1_variantA_lo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0001_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0012_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0003_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0002_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0014_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0007_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0009_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0004_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0006_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0010_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0005_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0013_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0011_geo.bin", "../scene/isDunesB_xgRoots--xgRoots_archiveroot0008_geo.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {15, 16, 0, 15, 1, 12, 3, 2, 14, 7, 9, 4, 6, 10, 5, 13, 11, 8},
    };

    m_curveBinPathsByElementInstance = {
        {},
    };

    m_curveMtlIndicesByElementInstance = {
        {},
    };

    }

}
