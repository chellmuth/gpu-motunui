#include "scene/coastline_element.hpp"

namespace moana {

CoastlineElement::CoastlineElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isCoastline/isCoastline.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0125_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0127_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0124_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0123_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0126_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archiveseedpodb_mod.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archivepineneedle0002_mod.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archivepineneedle0003_mod.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archivepineneedle0001_mod.obj",
    };

    m_binPaths = {
        "../scene/isCoastline-xgPalmDebris_archiveLeaflet0125_geo.bin",
        "../scene/isCoastline-xgPalmDebris_archiveLeaflet0127_geo.bin",
        "../scene/isCoastline-xgPalmDebris_archiveLeaflet0124_geo.bin",
        "../scene/isCoastline-xgPalmDebris_archiveLeaflet0123_geo.bin",
        "../scene/isCoastline-xgPalmDebris_archiveLeaflet0126_geo.bin",
        "../scene/isCoastline-xgFibers_archiveseedpodb_mod.bin",
        "../scene/isCoastline-xgFibers_archivepineneedle0002_mod.bin",
        "../scene/isCoastline-xgFibers_archivepineneedle0003_mod.bin",
        "../scene/isCoastline-xgFibers_archivepineneedle0001_mod.bin",
    };

    m_elementInstancesBinPath = "../scene/isCoastline-root.bin";
}

}
