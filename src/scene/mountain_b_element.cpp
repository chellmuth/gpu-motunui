#include "scene/mountain_b_element.hpp"

namespace moana {

MountainBElement::MountainBElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isMountainB";

    m_mtlLookup = {
        "branches",
        "breadFruit",
        "ferns",
        "fronds",
        "leavesOnHillside",
        "loGrowth",
        "mountainLo",
        "trunk",
        "trunkDetailed",
    };

    m_materialOffset = 83;

    m_baseObjs = {
        moanaRoot + "/island/obj/isMountainB/isMountainB.obj",
    };

    m_objArchivePaths = {
        moanaRoot + "/island/obj/isMountainB/archives/xgBreadFruit_archiveBreadFruitBaked.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig1.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig12.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig13.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig14.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig15.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig16.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig17.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig2.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig3.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig6.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig8.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0001_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0002_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0003_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0004_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0005_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0006_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0007_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0008_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0009_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0010_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0011_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0012_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0013_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0014_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFoliageA_treeMadronaBaked_canopyOnly_lo.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFoliageAd_treeMadronaBaked_canopyOnly_lo.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFoliageB_treeMadronaBaked_canopyOnly_lo.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFoliageC_treeMadronaBaked_canopyOnly_lo.obj",
    };

    m_elementInstancesBinPaths = {
        "../scene/isMountainB.bin",
    };

    m_primitiveInstancesBinPaths = {
        {"../scene/isMountainB_xgFoliageB--xgFoliageB_treeMadronaBaked_canopyOnly_lo.bin", "../scene/isMountainB_xgFoliageC--xgFoliageC_treeMadronaBaked_canopyOnly_lo.bin", "../scene/isMountainB_xgFoliageA--xgFoliageA_treeMadronaBaked_canopyOnly_lo.bin", "../scene/isMountainB_xgFoliageAd--xgFoliageAd_treeMadronaBaked_canopyOnly_lo.bin", "../scene/isMountainB_xgBreadFruit--xgBreadFruit_archiveBreadFruitBaked.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig3.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig14.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig17.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig8.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig12.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig2.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig1.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig16.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig15.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig6.bin", "../scene/isMountainB_xgCocoPalms--xgCocoPalms_isPalmRig13.bin", "../scene/isMountainB_xgFern--xgFern_fern0006_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0013_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0005_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0007_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0009_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0004_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0001_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0011_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0012_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0014_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0008_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0010_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0002_mod.bin", "../scene/isMountainB_xgFern--xgFern_fern0003_mod.bin"},
    };

    m_primitiveInstancesHandleIndices = {
        {29, 30, 27, 28, 0, 10, 5, 8, 12, 3, 1, 9, 2, 7, 6, 11, 4, 18, 25, 17, 19, 21, 16, 13, 23, 24, 26, 20, 22, 14, 15},
    };

    m_curveBinPathsByElementInstance = {
        {},
    };

    m_curveMtlIndicesByElementInstance = {
        {},
    };


}




}
