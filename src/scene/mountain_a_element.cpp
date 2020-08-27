#include "scene/mountain_a_element.hpp"

namespace moana {

MountainAElement::MountainAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isMountainA/isMountainA.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isMountainA/archives/xgFoliageC_treeMadronaBaked_canopyOnly_lo.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgBreadFruit_archiveBreadFruitBaked.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig5.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig6.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig13.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig14.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig3.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig1.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig16.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig15.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig4.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig12.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig8.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig2.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig7.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig17.obj",
    };

    m_binPaths = {
        "../scene/isMountainA-xgFoliageC_treeMadronaBaked_canopyOnly_lo.bin",
        "../scene/isMountainA-xgBreadFruit_archiveBreadFruitBaked.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig5.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig6.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig13.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig14.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig3.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig1.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig16.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig15.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig4.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig12.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig8.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig2.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig7.bin",
        "../scene/isMountainA-xgCocoPalms_isPalmRig17.bin",
    };

    m_hasElementInstances = true;
    m_elementInstancesBinPath = "../scene/isMountainA-root.bin";
}

}
