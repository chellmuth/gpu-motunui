#include "util/color_map.hpp"

namespace moana {

float3 ColorMap::get(int index)
{
    if (m_colorMap.count(index) == 0) {
        const float r = m_rng.next();
        const float g = m_rng.next();
        const float b = m_rng.next();

        const float3 color = float3{r, g, b};
        m_colorMap[index] = color;
    }

    return m_colorMap[index];
}

}
