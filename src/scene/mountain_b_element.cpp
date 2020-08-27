#include "scene/mountain_b_element.hpp"

namespace moana {

MountainBElement::MountainBElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isMountainB/isMountainB.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isMountainB/archives/xgFoliageB_treeMadronaBaked_canopyOnly_lo.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFoliageC_treeMadronaBaked_canopyOnly_lo.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFoliageA_treeMadronaBaked_canopyOnly_lo.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFoliageAd_treeMadronaBaked_canopyOnly_lo.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgBreadFruit_archiveBreadFruitBaked.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig3.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig14.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig17.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig8.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig12.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig2.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig1.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig16.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig15.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig6.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgCocoPalms_isPalmRig13.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0006_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0013_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0005_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0007_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0009_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0004_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0001_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0011_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0012_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0014_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0008_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0010_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0002_mod.obj",
        moanaRoot + "/island/obj/isMountainB/archives/xgFern_fern0003_mod.obj",
    };

    m_binPaths = {
        "../scene/isMountainB-xgFoliageB_treeMadronaBaked_canopyOnly_lo.bin",
        "../scene/isMountainB-xgFoliageC_treeMadronaBaked_canopyOnly_lo.bin",
        "../scene/isMountainB-xgFoliageA_treeMadronaBaked_canopyOnly_lo.bin",
        "../scene/isMountainB-xgFoliageAd_treeMadronaBaked_canopyOnly_lo.bin",
        "../scene/isMountainB-xgBreadFruit_archiveBreadFruitBaked.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig3.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig14.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig17.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig8.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig12.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig2.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig1.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig16.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig15.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig6.bin",
        "../scene/isMountainB-xgCocoPalms_isPalmRig13.bin",
        "../scene/isMountainB-xgFern_fern0006_mod.bin",
        "../scene/isMountainB-xgFern_fern0013_mod.bin",
        "../scene/isMountainB-xgFern_fern0005_mod.bin",
        "../scene/isMountainB-xgFern_fern0007_mod.bin",
        "../scene/isMountainB-xgFern_fern0009_mod.bin",
        "../scene/isMountainB-xgFern_fern0004_mod.bin",
        "../scene/isMountainB-xgFern_fern0001_mod.bin",
        "../scene/isMountainB-xgFern_fern0011_mod.bin",
        "../scene/isMountainB-xgFern_fern0012_mod.bin",
        "../scene/isMountainB-xgFern_fern0014_mod.bin",
        "../scene/isMountainB-xgFern_fern0008_mod.bin",
        "../scene/isMountainB-xgFern_fern0010_mod.bin",
        "../scene/isMountainB-xgFern_fern0002_mod.bin",
        "../scene/isMountainB-xgFern_fern0003_mod.bin",
    };

    m_hasElementInstances = true;
    m_elementInstancesBinPath = "../scene/isMountainB-root.bin";
}

}
