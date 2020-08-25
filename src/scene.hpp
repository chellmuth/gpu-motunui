#pragma once

#include "moana/core/camera.hpp"

namespace moana {

enum class Cam {
    BeachCam,
    BirdseyeCam,
    DunesACam,
    GrassCam,
    PalmsCam,
    RootsCam,
    ShotCam
};

class Scene {
public:
    Scene(Cam cam);

    Camera getCamera(int width, int height) const;

private:
    Cam m_cam;
};

}
