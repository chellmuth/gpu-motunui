#include "scene/coral_element.hpp"

namespace moana {

CoralElement::CoralElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isCoral/isCoral.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0008_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0009_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0007_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0005_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0003_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0001_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0006_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0004_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0002_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgFlutes_flutes.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0002_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0007_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0005_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0003_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0009_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0008_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0006_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0004_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0001_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0008_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0006_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0002_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0010_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0009_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0005_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0007_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0004_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0001_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0003_geo.obj",
    };

    m_binPaths = {
        "../scene/isCoral-xgCabbage_archivecoral_cabbage0008_geo.bin",
        "../scene/isCoral-xgCabbage_archivecoral_cabbage0009_geo.bin",
        "../scene/isCoral-xgCabbage_archivecoral_cabbage0007_geo.bin",
        "../scene/isCoral-xgCabbage_archivecoral_cabbage0005_geo.bin",
        "../scene/isCoral-xgCabbage_archivecoral_cabbage0003_geo.bin",
        "../scene/isCoral-xgCabbage_archivecoral_cabbage0001_geo.bin",
        "../scene/isCoral-xgCabbage_archivecoral_cabbage0006_geo.bin",
        "../scene/isCoral-xgCabbage_archivecoral_cabbage0004_geo.bin",
        "../scene/isCoral-xgCabbage_archivecoral_cabbage0002_geo.bin",
        "../scene/isCoral-xgFlutes_flutes.bin",
        "../scene/isCoral-xgAntlers_archivecoral_antler0002_geo.bin",
        "../scene/isCoral-xgAntlers_archivecoral_antler0007_geo.bin",
        "../scene/isCoral-xgAntlers_archivecoral_antler0005_geo.bin",
        "../scene/isCoral-xgAntlers_archivecoral_antler0003_geo.bin",
        "../scene/isCoral-xgAntlers_archivecoral_antler0009_geo.bin",
        "../scene/isCoral-xgAntlers_archivecoral_antler0008_geo.bin",
        "../scene/isCoral-xgAntlers_archivecoral_antler0006_geo.bin",
        "../scene/isCoral-xgAntlers_archivecoral_antler0004_geo.bin",
        "../scene/isCoral-xgAntlers_archivecoral_antler0001_geo.bin",
        "../scene/isCoral-xgStaghorn_archivecoral_staghorn0008_geo.bin",
        "../scene/isCoral-xgStaghorn_archivecoral_staghorn0006_geo.bin",
        "../scene/isCoral-xgStaghorn_archivecoral_staghorn0002_geo.bin",
        "../scene/isCoral-xgStaghorn_archivecoral_staghorn0010_geo.bin",
        "../scene/isCoral-xgStaghorn_archivecoral_staghorn0009_geo.bin",
        "../scene/isCoral-xgStaghorn_archivecoral_staghorn0005_geo.bin",
        "../scene/isCoral-xgStaghorn_archivecoral_staghorn0007_geo.bin",
        "../scene/isCoral-xgStaghorn_archivecoral_staghorn0004_geo.bin",
        "../scene/isCoral-xgStaghorn_archivecoral_staghorn0001_geo.bin",
        "../scene/isCoral-xgStaghorn_archivecoral_staghorn0003_geo.bin",
    };

    m_elementInstancesBinPath = "../scene/isCoral-root.bin";
}

}
