#include "scene/mountain_a_element.hpp"

namespace moana {

MountainAElement::MountainAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isMountainA";

    m_mtlLookup = {
        "branches",
        "breadFruit",
        "fronds",
        "leavesOnHillside",
        "mountainLo",
        "trunk",
    };

    m_materialOffset = 77;

    m_baseObjs = {
        moanaRoot + "/island/obj/isMountainA/isMountainA.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isMountainA/archives/xgBreadFruit_archiveBreadFruitBaked.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig1.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig12.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig13.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig14.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig15.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig16.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig17.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig2.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig3.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig4.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig5.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig6.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig7.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig8.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgFoliageC_treeMadronaBaked_canopyOnly_lo.obj",
    };

    m_archivePrimitiveIndexOffsets = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    m_baseObjPrimitiveIndexOffsets = {
        0
    };

    m_elementInstancesBinPaths = {
        "../scene/isMountainA.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isMountainA_xgFoliageC--xgFoliageC_treeMadronaBaked_canopyOnly_lo.bin", "../scene/isMountainA_xgBreadFruit--xgBreadFruit_archiveBreadFruitBaked.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig5.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig6.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig13.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig14.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig3.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig1.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig16.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig15.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig4.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig12.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig8.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig2.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig7.bin", "../scene/isMountainA_xgCocoPalms--xgCocoPalms_isPalmRig17.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {16, 0, 12, 13, 4, 5, 10, 2, 7, 6, 11, 1, 3, 15, 9, 14, 8},
    };

    m_curveBinPathsByElementInstance = {
        {},
    };

    m_curveMtlIndicesByElementInstance = {
        {},
    };

    }




}
