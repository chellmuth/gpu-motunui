#include "moana/core/camera.hpp"

#include <cmath>
#include <iostream> // fixme

namespace moana {

Camera::Camera(
    const Vec3 &origin,
    const Vec3 &target,
    const Vec3 &up,
    float verticalFOV,
    const Resolution &resolution,
    bool flipHandedness
) : m_origin(origin),
    m_target(target),
    m_up(up),
    m_verticalFOV(verticalFOV),
    m_resolution(resolution),
    m_flipHandedness(flipHandedness)
{
    m_cameraToWorld = lookAt(m_origin, m_target, m_up, m_flipHandedness);
}

}
