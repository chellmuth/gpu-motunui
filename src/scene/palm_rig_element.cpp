#include "scene/palm_rig_element.hpp"

namespace moana {

PalmRigElement::PalmRigElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isPalmRig/isPalmRig.obj";

    m_objPaths = {

    };

    m_binPaths = {

    };

    m_hasElementInstances = true;
    m_elementInstancesBinPath = "../scene/isPalmRig-root.bin";
}

}
