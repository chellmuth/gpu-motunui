#include "scene/lava_rocks_element.hpp"

namespace moana {

LavaRocksElement::LavaRocksElement()
{
    const std::string moanaRoot = MOANA_ROOT;

    m_baseObj = moanaRoot + "/island/obj/isLavaRocks/isLavaRocks.obj";

    m_objPaths = {

    };

    m_binPaths = {

    };

    m_elementInstancesBinPath = "../scene/isLavaRocks-root.bin";
}

}
