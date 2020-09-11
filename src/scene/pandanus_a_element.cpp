#include "scene/pandanus_a_element.hpp"

namespace moana {

PandanusAElement::PandanusAElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_elementName = "isPandanusA";

    m_mtlLookup = {
        "leaves",
        "leavesLower",
        "trunk",
    };

    m_materialOffset = 100;

    m_baseObjs = {
        moanaRoot + "/island/obj/isPandanusA/isPandanusA.obj",
    };

    m_objArchivePaths = {

    };

    m_elementInstancesBinPaths = {
        "../scene/isPandanusA.bin",
    };

    m_primitiveInstancesBinPaths = {
        {},
    };

    m_primitiveInstancesHandleIndices = {
        {},
    };

    m_curveBinPathsByElementInstance = {
        {"../scene/curves__isPandanusA_xgLeavesH.bin", "../scene/curves__isPandanusA_xgLeavesI.bin", "../scene/curves__isPandanusA_xgLeavesLower.bin", "../scene/curves__isPandanusA_xgLeavesA.bin", "../scene/curves__isPandanusA_xgLeavesB.bin", "../scene/curves__isPandanusA_xgLeavesC.bin", "../scene/curves__isPandanusA_xgLeavesD.bin", "../scene/curves__isPandanusA_xgLeavesE.bin", "../scene/curves__isPandanusA_xgLeavesF.bin", "../scene/curves__isPandanusA_xgLeavesG.bin"},
    };

    m_curveMtlIndicesByElementInstance = {
        {0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    };


}




}
