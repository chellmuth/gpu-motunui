#include "scene/coral_element.hpp"

namespace moana {

CoralElement::CoralElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isCoral";

    m_mtlLookup = {
        "coral",
        "xgAntler",
        "xgCabbage",
        "xgFlutes",
        "xgStaghorn",
    };

    m_materialOffset = 21;

    m_baseObjs = {
        moanaRoot + "/island/obj/isCoral/isCoral.obj",
        moanaRoot + "/island/obj/isCoral/isCoral5.obj",
        moanaRoot + "/island/obj/isCoral/isCoral4.obj",
        moanaRoot + "/island/obj/isCoral/isCoral1.obj",
        moanaRoot + "/island/obj/isCoral/isCoral3.obj",
        moanaRoot + "/island/obj/isCoral/isCoral2.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0001_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0002_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0003_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0004_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0005_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0006_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0007_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0008_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgAntlers_archivecoral_antler0009_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0001_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0002_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0003_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0004_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0005_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0006_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0007_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0008_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgCabbage_archivecoral_cabbage0009_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgFlutes_flutes.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0001_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0002_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0003_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0004_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0005_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0006_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0007_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0008_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0009_geo.obj",
        moanaRoot + "/island/obj/isCoral/archives/xgStaghorn_archivecoral_staghorn0010_geo.obj",
    };

    m_elementInstancesBinPaths = {
        "../scene/isCoral.bin",
        "../scene/isCoral5.bin",
        "../scene/isCoral4.bin",
        "../scene/isCoral1.bin",
        "../scene/isCoral3.bin",
        "../scene/isCoral2.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isCoral_xgCabbage--xgCabbage_archivecoral_cabbage0008_geo.bin", "../scene/isCoral_xgCabbage--xgCabbage_archivecoral_cabbage0009_geo.bin", "../scene/isCoral_xgCabbage--xgCabbage_archivecoral_cabbage0007_geo.bin", "../scene/isCoral_xgCabbage--xgCabbage_archivecoral_cabbage0005_geo.bin", "../scene/isCoral_xgCabbage--xgCabbage_archivecoral_cabbage0003_geo.bin", "../scene/isCoral_xgCabbage--xgCabbage_archivecoral_cabbage0001_geo.bin", "../scene/isCoral_xgCabbage--xgCabbage_archivecoral_cabbage0006_geo.bin", "../scene/isCoral_xgCabbage--xgCabbage_archivecoral_cabbage0004_geo.bin", "../scene/isCoral_xgCabbage--xgCabbage_archivecoral_cabbage0002_geo.bin", "../scene/isCoral_xgFlutes--xgFlutes_flutes.bin", "../scene/isCoral_xgAntlers--xgAntlers_archivecoral_antler0002_geo.bin", "../scene/isCoral_xgAntlers--xgAntlers_archivecoral_antler0007_geo.bin", "../scene/isCoral_xgAntlers--xgAntlers_archivecoral_antler0005_geo.bin", "../scene/isCoral_xgAntlers--xgAntlers_archivecoral_antler0003_geo.bin", "../scene/isCoral_xgAntlers--xgAntlers_archivecoral_antler0009_geo.bin", "../scene/isCoral_xgAntlers--xgAntlers_archivecoral_antler0008_geo.bin", "../scene/isCoral_xgAntlers--xgAntlers_archivecoral_antler0006_geo.bin", "../scene/isCoral_xgAntlers--xgAntlers_archivecoral_antler0004_geo.bin", "../scene/isCoral_xgAntlers--xgAntlers_archivecoral_antler0001_geo.bin", "../scene/isCoral_xgStaghorn--xgStaghorn_archivecoral_staghorn0008_geo.bin", "../scene/isCoral_xgStaghorn--xgStaghorn_archivecoral_staghorn0006_geo.bin", "../scene/isCoral_xgStaghorn--xgStaghorn_archivecoral_staghorn0002_geo.bin", "../scene/isCoral_xgStaghorn--xgStaghorn_archivecoral_staghorn0010_geo.bin", "../scene/isCoral_xgStaghorn--xgStaghorn_archivecoral_staghorn0009_geo.bin", "../scene/isCoral_xgStaghorn--xgStaghorn_archivecoral_staghorn0005_geo.bin", "../scene/isCoral_xgStaghorn--xgStaghorn_archivecoral_staghorn0007_geo.bin", "../scene/isCoral_xgStaghorn--xgStaghorn_archivecoral_staghorn0004_geo.bin", "../scene/isCoral_xgStaghorn--xgStaghorn_archivecoral_staghorn0001_geo.bin", "../scene/isCoral_xgStaghorn--xgStaghorn_archivecoral_staghorn0003_geo.bin"},
        {"../scene/isCoral5_xgCabbage--xgCabbage_archivecoral_cabbage0008_geo.bin", "../scene/isCoral5_xgCabbage--xgCabbage_archivecoral_cabbage0009_geo.bin", "../scene/isCoral5_xgCabbage--xgCabbage_archivecoral_cabbage0007_geo.bin", "../scene/isCoral5_xgCabbage--xgCabbage_archivecoral_cabbage0005_geo.bin", "../scene/isCoral5_xgCabbage--xgCabbage_archivecoral_cabbage0003_geo.bin", "../scene/isCoral5_xgCabbage--xgCabbage_archivecoral_cabbage0001_geo.bin", "../scene/isCoral5_xgCabbage--xgCabbage_archivecoral_cabbage0006_geo.bin", "../scene/isCoral5_xgCabbage--xgCabbage_archivecoral_cabbage0004_geo.bin", "../scene/isCoral5_xgCabbage--xgCabbage_archivecoral_cabbage0002_geo.bin", "../scene/isCoral5_xgFlutes--xgFlutes_flutes.bin", "../scene/isCoral5_xgAntlers--xgAntlers_archivecoral_antler0002_geo.bin", "../scene/isCoral5_xgAntlers--xgAntlers_archivecoral_antler0007_geo.bin", "../scene/isCoral5_xgAntlers--xgAntlers_archivecoral_antler0005_geo.bin", "../scene/isCoral5_xgAntlers--xgAntlers_archivecoral_antler0003_geo.bin", "../scene/isCoral5_xgAntlers--xgAntlers_archivecoral_antler0009_geo.bin", "../scene/isCoral5_xgAntlers--xgAntlers_archivecoral_antler0008_geo.bin", "../scene/isCoral5_xgAntlers--xgAntlers_archivecoral_antler0006_geo.bin", "../scene/isCoral5_xgAntlers--xgAntlers_archivecoral_antler0004_geo.bin", "../scene/isCoral5_xgAntlers--xgAntlers_archivecoral_antler0001_geo.bin", "../scene/isCoral5_xgStaghorn--xgStaghorn_archivecoral_staghorn0008_geo.bin", "../scene/isCoral5_xgStaghorn--xgStaghorn_archivecoral_staghorn0006_geo.bin", "../scene/isCoral5_xgStaghorn--xgStaghorn_archivecoral_staghorn0002_geo.bin", "../scene/isCoral5_xgStaghorn--xgStaghorn_archivecoral_staghorn0010_geo.bin", "../scene/isCoral5_xgStaghorn--xgStaghorn_archivecoral_staghorn0009_geo.bin", "../scene/isCoral5_xgStaghorn--xgStaghorn_archivecoral_staghorn0005_geo.bin", "../scene/isCoral5_xgStaghorn--xgStaghorn_archivecoral_staghorn0007_geo.bin", "../scene/isCoral5_xgStaghorn--xgStaghorn_archivecoral_staghorn0004_geo.bin", "../scene/isCoral5_xgStaghorn--xgStaghorn_archivecoral_staghorn0001_geo.bin", "../scene/isCoral5_xgStaghorn--xgStaghorn_archivecoral_staghorn0003_geo.bin"},
        {"../scene/isCoral4_xgCabbage--xgCabbage_archivecoral_cabbage0008_geo.bin", "../scene/isCoral4_xgCabbage--xgCabbage_archivecoral_cabbage0009_geo.bin", "../scene/isCoral4_xgCabbage--xgCabbage_archivecoral_cabbage0007_geo.bin", "../scene/isCoral4_xgCabbage--xgCabbage_archivecoral_cabbage0005_geo.bin", "../scene/isCoral4_xgCabbage--xgCabbage_archivecoral_cabbage0003_geo.bin", "../scene/isCoral4_xgCabbage--xgCabbage_archivecoral_cabbage0001_geo.bin", "../scene/isCoral4_xgCabbage--xgCabbage_archivecoral_cabbage0006_geo.bin", "../scene/isCoral4_xgCabbage--xgCabbage_archivecoral_cabbage0004_geo.bin", "../scene/isCoral4_xgCabbage--xgCabbage_archivecoral_cabbage0002_geo.bin", "../scene/isCoral4_xgFlutes--xgFlutes_flutes.bin", "../scene/isCoral4_xgAntlers--xgAntlers_archivecoral_antler0002_geo.bin", "../scene/isCoral4_xgAntlers--xgAntlers_archivecoral_antler0007_geo.bin", "../scene/isCoral4_xgAntlers--xgAntlers_archivecoral_antler0005_geo.bin", "../scene/isCoral4_xgAntlers--xgAntlers_archivecoral_antler0003_geo.bin", "../scene/isCoral4_xgAntlers--xgAntlers_archivecoral_antler0009_geo.bin", "../scene/isCoral4_xgAntlers--xgAntlers_archivecoral_antler0008_geo.bin", "../scene/isCoral4_xgAntlers--xgAntlers_archivecoral_antler0006_geo.bin", "../scene/isCoral4_xgAntlers--xgAntlers_archivecoral_antler0004_geo.bin", "../scene/isCoral4_xgAntlers--xgAntlers_archivecoral_antler0001_geo.bin", "../scene/isCoral4_xgStaghorn--xgStaghorn_archivecoral_staghorn0008_geo.bin", "../scene/isCoral4_xgStaghorn--xgStaghorn_archivecoral_staghorn0006_geo.bin", "../scene/isCoral4_xgStaghorn--xgStaghorn_archivecoral_staghorn0002_geo.bin", "../scene/isCoral4_xgStaghorn--xgStaghorn_archivecoral_staghorn0010_geo.bin", "../scene/isCoral4_xgStaghorn--xgStaghorn_archivecoral_staghorn0009_geo.bin", "../scene/isCoral4_xgStaghorn--xgStaghorn_archivecoral_staghorn0005_geo.bin", "../scene/isCoral4_xgStaghorn--xgStaghorn_archivecoral_staghorn0007_geo.bin", "../scene/isCoral4_xgStaghorn--xgStaghorn_archivecoral_staghorn0004_geo.bin", "../scene/isCoral4_xgStaghorn--xgStaghorn_archivecoral_staghorn0001_geo.bin", "../scene/isCoral4_xgStaghorn--xgStaghorn_archivecoral_staghorn0003_geo.bin"},
        {"../scene/isCoral1_xgCabbage--xgCabbage_archivecoral_cabbage0008_geo.bin", "../scene/isCoral1_xgCabbage--xgCabbage_archivecoral_cabbage0009_geo.bin", "../scene/isCoral1_xgCabbage--xgCabbage_archivecoral_cabbage0007_geo.bin", "../scene/isCoral1_xgCabbage--xgCabbage_archivecoral_cabbage0005_geo.bin", "../scene/isCoral1_xgCabbage--xgCabbage_archivecoral_cabbage0003_geo.bin", "../scene/isCoral1_xgCabbage--xgCabbage_archivecoral_cabbage0001_geo.bin", "../scene/isCoral1_xgCabbage--xgCabbage_archivecoral_cabbage0006_geo.bin", "../scene/isCoral1_xgCabbage--xgCabbage_archivecoral_cabbage0004_geo.bin", "../scene/isCoral1_xgCabbage--xgCabbage_archivecoral_cabbage0002_geo.bin", "../scene/isCoral1_xgFlutes--xgFlutes_flutes.bin", "../scene/isCoral1_xgAntlers--xgAntlers_archivecoral_antler0002_geo.bin", "../scene/isCoral1_xgAntlers--xgAntlers_archivecoral_antler0007_geo.bin", "../scene/isCoral1_xgAntlers--xgAntlers_archivecoral_antler0005_geo.bin", "../scene/isCoral1_xgAntlers--xgAntlers_archivecoral_antler0003_geo.bin", "../scene/isCoral1_xgAntlers--xgAntlers_archivecoral_antler0009_geo.bin", "../scene/isCoral1_xgAntlers--xgAntlers_archivecoral_antler0008_geo.bin", "../scene/isCoral1_xgAntlers--xgAntlers_archivecoral_antler0006_geo.bin", "../scene/isCoral1_xgAntlers--xgAntlers_archivecoral_antler0004_geo.bin", "../scene/isCoral1_xgAntlers--xgAntlers_archivecoral_antler0001_geo.bin", "../scene/isCoral1_xgStaghorn--xgStaghorn_archivecoral_staghorn0008_geo.bin", "../scene/isCoral1_xgStaghorn--xgStaghorn_archivecoral_staghorn0006_geo.bin", "../scene/isCoral1_xgStaghorn--xgStaghorn_archivecoral_staghorn0002_geo.bin", "../scene/isCoral1_xgStaghorn--xgStaghorn_archivecoral_staghorn0010_geo.bin", "../scene/isCoral1_xgStaghorn--xgStaghorn_archivecoral_staghorn0009_geo.bin", "../scene/isCoral1_xgStaghorn--xgStaghorn_archivecoral_staghorn0005_geo.bin", "../scene/isCoral1_xgStaghorn--xgStaghorn_archivecoral_staghorn0007_geo.bin", "../scene/isCoral1_xgStaghorn--xgStaghorn_archivecoral_staghorn0004_geo.bin", "../scene/isCoral1_xgStaghorn--xgStaghorn_archivecoral_staghorn0001_geo.bin", "../scene/isCoral1_xgStaghorn--xgStaghorn_archivecoral_staghorn0003_geo.bin"},
        {"../scene/isCoral3_xgCabbage--xgCabbage_archivecoral_cabbage0008_geo.bin", "../scene/isCoral3_xgCabbage--xgCabbage_archivecoral_cabbage0009_geo.bin", "../scene/isCoral3_xgCabbage--xgCabbage_archivecoral_cabbage0007_geo.bin", "../scene/isCoral3_xgCabbage--xgCabbage_archivecoral_cabbage0005_geo.bin", "../scene/isCoral3_xgCabbage--xgCabbage_archivecoral_cabbage0003_geo.bin", "../scene/isCoral3_xgCabbage--xgCabbage_archivecoral_cabbage0001_geo.bin", "../scene/isCoral3_xgCabbage--xgCabbage_archivecoral_cabbage0006_geo.bin", "../scene/isCoral3_xgCabbage--xgCabbage_archivecoral_cabbage0004_geo.bin", "../scene/isCoral3_xgCabbage--xgCabbage_archivecoral_cabbage0002_geo.bin", "../scene/isCoral3_xgFlutes--xgFlutes_flutes.bin", "../scene/isCoral3_xgAntlers--xgAntlers_archivecoral_antler0002_geo.bin", "../scene/isCoral3_xgAntlers--xgAntlers_archivecoral_antler0007_geo.bin", "../scene/isCoral3_xgAntlers--xgAntlers_archivecoral_antler0005_geo.bin", "../scene/isCoral3_xgAntlers--xgAntlers_archivecoral_antler0003_geo.bin", "../scene/isCoral3_xgAntlers--xgAntlers_archivecoral_antler0009_geo.bin", "../scene/isCoral3_xgAntlers--xgAntlers_archivecoral_antler0008_geo.bin", "../scene/isCoral3_xgAntlers--xgAntlers_archivecoral_antler0006_geo.bin", "../scene/isCoral3_xgAntlers--xgAntlers_archivecoral_antler0004_geo.bin", "../scene/isCoral3_xgAntlers--xgAntlers_archivecoral_antler0001_geo.bin", "../scene/isCoral3_xgStaghorn--xgStaghorn_archivecoral_staghorn0008_geo.bin", "../scene/isCoral3_xgStaghorn--xgStaghorn_archivecoral_staghorn0006_geo.bin", "../scene/isCoral3_xgStaghorn--xgStaghorn_archivecoral_staghorn0002_geo.bin", "../scene/isCoral3_xgStaghorn--xgStaghorn_archivecoral_staghorn0010_geo.bin", "../scene/isCoral3_xgStaghorn--xgStaghorn_archivecoral_staghorn0009_geo.bin", "../scene/isCoral3_xgStaghorn--xgStaghorn_archivecoral_staghorn0005_geo.bin", "../scene/isCoral3_xgStaghorn--xgStaghorn_archivecoral_staghorn0007_geo.bin", "../scene/isCoral3_xgStaghorn--xgStaghorn_archivecoral_staghorn0004_geo.bin", "../scene/isCoral3_xgStaghorn--xgStaghorn_archivecoral_staghorn0001_geo.bin", "../scene/isCoral3_xgStaghorn--xgStaghorn_archivecoral_staghorn0003_geo.bin"},
        {"../scene/isCoral2_xgCabbage--xgCabbage_archivecoral_cabbage0008_geo.bin", "../scene/isCoral2_xgCabbage--xgCabbage_archivecoral_cabbage0009_geo.bin", "../scene/isCoral2_xgCabbage--xgCabbage_archivecoral_cabbage0007_geo.bin", "../scene/isCoral2_xgCabbage--xgCabbage_archivecoral_cabbage0005_geo.bin", "../scene/isCoral2_xgCabbage--xgCabbage_archivecoral_cabbage0003_geo.bin", "../scene/isCoral2_xgCabbage--xgCabbage_archivecoral_cabbage0001_geo.bin", "../scene/isCoral2_xgCabbage--xgCabbage_archivecoral_cabbage0006_geo.bin", "../scene/isCoral2_xgCabbage--xgCabbage_archivecoral_cabbage0004_geo.bin", "../scene/isCoral2_xgCabbage--xgCabbage_archivecoral_cabbage0002_geo.bin", "../scene/isCoral2_xgFlutes--xgFlutes_flutes.bin", "../scene/isCoral2_xgAntlers--xgAntlers_archivecoral_antler0002_geo.bin", "../scene/isCoral2_xgAntlers--xgAntlers_archivecoral_antler0007_geo.bin", "../scene/isCoral2_xgAntlers--xgAntlers_archivecoral_antler0005_geo.bin", "../scene/isCoral2_xgAntlers--xgAntlers_archivecoral_antler0003_geo.bin", "../scene/isCoral2_xgAntlers--xgAntlers_archivecoral_antler0009_geo.bin", "../scene/isCoral2_xgAntlers--xgAntlers_archivecoral_antler0008_geo.bin", "../scene/isCoral2_xgAntlers--xgAntlers_archivecoral_antler0006_geo.bin", "../scene/isCoral2_xgAntlers--xgAntlers_archivecoral_antler0004_geo.bin", "../scene/isCoral2_xgAntlers--xgAntlers_archivecoral_antler0001_geo.bin", "../scene/isCoral2_xgStaghorn--xgStaghorn_archivecoral_staghorn0008_geo.bin", "../scene/isCoral2_xgStaghorn--xgStaghorn_archivecoral_staghorn0006_geo.bin", "../scene/isCoral2_xgStaghorn--xgStaghorn_archivecoral_staghorn0002_geo.bin", "../scene/isCoral2_xgStaghorn--xgStaghorn_archivecoral_staghorn0010_geo.bin", "../scene/isCoral2_xgStaghorn--xgStaghorn_archivecoral_staghorn0009_geo.bin", "../scene/isCoral2_xgStaghorn--xgStaghorn_archivecoral_staghorn0005_geo.bin", "../scene/isCoral2_xgStaghorn--xgStaghorn_archivecoral_staghorn0007_geo.bin", "../scene/isCoral2_xgStaghorn--xgStaghorn_archivecoral_staghorn0004_geo.bin", "../scene/isCoral2_xgStaghorn--xgStaghorn_archivecoral_staghorn0001_geo.bin", "../scene/isCoral2_xgStaghorn--xgStaghorn_archivecoral_staghorn0003_geo.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {16, 17, 15, 13, 11, 9, 14, 12, 10, 18, 1, 6, 4, 2, 8, 7, 5, 3, 0, 26, 24, 20, 28, 27, 23, 25, 22, 19, 21},
        {16, 17, 15, 13, 11, 9, 14, 12, 10, 18, 1, 6, 4, 2, 8, 7, 5, 3, 0, 26, 24, 20, 28, 27, 23, 25, 22, 19, 21},
        {16, 17, 15, 13, 11, 9, 14, 12, 10, 18, 1, 6, 4, 2, 8, 7, 5, 3, 0, 26, 24, 20, 28, 27, 23, 25, 22, 19, 21},
        {16, 17, 15, 13, 11, 9, 14, 12, 10, 18, 1, 6, 4, 2, 8, 7, 5, 3, 0, 26, 24, 20, 28, 27, 23, 25, 22, 19, 21},
        {16, 17, 15, 13, 11, 9, 14, 12, 10, 18, 1, 6, 4, 2, 8, 7, 5, 3, 0, 26, 24, 20, 28, 27, 23, 25, 22, 19, 21},
        {16, 17, 15, 13, 11, 9, 14, 12, 10, 18, 1, 6, 4, 2, 8, 7, 5, 3, 0, 26, 24, 20, 28, 27, 23, 25, 22, 19, 21},
    };

    m_curveBinPathsByElementInstance = {
        {},
        {},
        {},
        {},
        {},
        {},
    };

    m_curveMtlIndicesByElementInstance = {
        {},
        {},
        {},
        {},
        {},
        {},
    };


}




}
