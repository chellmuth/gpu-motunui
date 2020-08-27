#include "scene/mountain_a_element.hpp"

namespace moana {

MountainAElement::MountainAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isMountainA/isMountainA.obj";

    m_objPaths = {
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig8.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig4.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig1.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig12.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig3.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig7.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig15.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig16.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig14.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig17.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig6.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig5.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig13.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig2.obj",

        moanaRoot + "/island/obj/isMountainA/archives/xgBreadFruit_archiveBreadFruitBaked.obj"
    };


    m_binPaths = {
        "../scene/mountainA-xgCocoPalms_isPalmRig8.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig4.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig1.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig12.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig3.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig7.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig15.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig16.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig14.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig17.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig6.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig5.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig13.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig2.bin",

        "../scene/mountainA-xgBreadFruit_archiveBreadFruitBaked.bin",
    };

    m_hasElementInstances = false;
    m_elementInstancesBinPath = "";
}

}
