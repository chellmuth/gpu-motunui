#include "scene/coastline_element.hpp"

namespace moana {

CoastlineElement::CoastlineElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isCoastline";

    m_mtlLookup = {
        "archiveFibers",
        "archiveLeaflet",
        "sandSimple",
    };

    m_materialOffset = 18;

    m_baseObjs = {
        moanaRoot + "/island/obj/isCoastline/isCoastline.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archivepineneedle0001_mod.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archivepineneedle0002_mod.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archivepineneedle0003_mod.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archiveseedpodb_mod.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0123_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0124_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0125_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0126_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0127_geo.obj",
    };

    m_archivePrimitiveIndexOffsets = {
        0, 0, 0, 0, 0, 112, 208, 312, 416
    };

    m_baseObjPrimitiveIndexOffsets = {
        0
    };

    m_elementInstancesBinPaths = {
        "../scene/isCoastline.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isCoastline_xgPalmDebris--xgPalmDebris_archiveLeaflet0125_geo.bin", "../scene/isCoastline_xgPalmDebris--xgPalmDebris_archiveLeaflet0127_geo.bin", "../scene/isCoastline_xgPalmDebris--xgPalmDebris_archiveLeaflet0124_geo.bin", "../scene/isCoastline_xgPalmDebris--xgPalmDebris_archiveLeaflet0123_geo.bin", "../scene/isCoastline_xgPalmDebris--xgPalmDebris_archiveLeaflet0126_geo.bin", "../scene/isCoastline_xgFibers--xgFibers_archiveseedpodb_mod.bin", "../scene/isCoastline_xgFibers--xgFibers_archivepineneedle0002_mod.bin", "../scene/isCoastline_xgFibers--xgFibers_archivepineneedle0003_mod.bin", "../scene/isCoastline_xgFibers--xgFibers_archivepineneedle0001_mod.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {6, 8, 5, 4, 7, 3, 1, 2, 0},
    };

    m_curveBinPathsByElementInstance = {
        {"../scene/curves__isCoastline_xgGrass.bin"},
    };

    m_curveMtlIndicesByElementInstance = {
        {0},
    };

    }




}
