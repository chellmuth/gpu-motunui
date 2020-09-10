#include "scene/palm_rig_element.hpp"

namespace moana {

PalmRigElement::PalmRigElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isPalmRig";

    m_mtlLookup = {
        "branches",
        "fronds",
        "trunk",
    };

    m_materialOffset = 97;

    m_baseObjs = {
        moanaRoot + "/island/obj/isPalmRig/isPalmRig.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig18.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig19.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig16.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig17.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig14.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig15.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig12.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig13.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig10.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig11.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig30.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig31.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig32.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig33.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig8.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig9.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig4.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig5.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig6.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig7.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig2.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig3.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig27.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig26.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig25.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig24.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig23.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig22.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig21.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig20.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig29.obj",
        moanaRoot + "/island/obj/isPalmRig/isPalmRig28.obj",
    };

    m_objArchivePaths = {

    };

    m_archivePrimitiveIndexOffsets = {
        
    };

    m_baseObjPrimitiveIndexOffsets = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    m_elementInstancesBinPaths = {
        "../scene/isPalmRig.bin",
        "../scene/isPalmRig18.bin",
        "../scene/isPalmRig19.bin",
        "../scene/isPalmRig16.bin",
        "../scene/isPalmRig17.bin",
        "../scene/isPalmRig14.bin",
        "../scene/isPalmRig15.bin",
        "../scene/isPalmRig12.bin",
        "../scene/isPalmRig13.bin",
        "../scene/isPalmRig10.bin",
        "../scene/isPalmRig11.bin",
        "../scene/isPalmRig30.bin",
        "../scene/isPalmRig31.bin",
        "../scene/isPalmRig32.bin",
        "../scene/isPalmRig33.bin",
        "../scene/isPalmRig8.bin",
        "../scene/isPalmRig9.bin",
        "../scene/isPalmRig4.bin",
        "../scene/isPalmRig5.bin",
        "../scene/isPalmRig6.bin",
        "../scene/isPalmRig7.bin",
        "../scene/isPalmRig2.bin",
        "../scene/isPalmRig3.bin",
        "../scene/isPalmRig27.bin",
        "../scene/isPalmRig26.bin",
        "../scene/isPalmRig25.bin",
        "../scene/isPalmRig24.bin",
        "../scene/isPalmRig23.bin",
        "../scene/isPalmRig22.bin",
        "../scene/isPalmRig21.bin",
        "../scene/isPalmRig20.bin",
        "../scene/isPalmRig29.bin",
        "../scene/isPalmRig28.bin",
    };

    m_primitiveInstancesBinPaths = {
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    };

    m_primitiveInstancesHandleIndices = {
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
    };

    m_curveBinPathsByElementInstance = {
        {"../scene/curves__isPalmRig_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig18_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig19_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig16_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig17_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig14_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig15_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig12_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig13_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig10_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig11_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig30_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig31_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig32_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig33_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig8_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig9_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig4_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig5_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig6_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig7_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig2_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig3_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig27_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig26_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig25_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig24_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig23_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig22_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig21_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig20_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig29_xgFrondsA.bin"},
        {"../scene/curves__isPalmRig28_xgFrondsA.bin"},
    };

    m_curveMtlIndicesByElementInstance = {
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
    };

    }




}
