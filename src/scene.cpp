#include "moana/scene.hpp"

#include "moana/core/vec3.hpp"

namespace moana {

Scene::Scene(Cam cam)
    : m_cam(cam)
{}

Camera Scene::getCamera(int width, int height) const
{
    struct CameraParams {
        Vec3 origin;
        Vec3 target;
        Vec3 up;
    };

    CameraParams params;
    switch (m_cam) {
    case Cam::BeachCam: {
        params.origin = Vec3(
          -510.5239066714681f,
          87.30874393630907f,
          181.77019700660784f
        );
        params.target = Vec3(
          152.46530457260906f,
          30.939794764162578f,
          -72.72751680648216f
        );
        params.up = Vec3(
          0.0738708545527661f,
          0.996864591331903f,
          -0.02835636430013811f
        );
        break;
    }
    case Cam::BirdseyeCam: {
        params.origin = Vec3(
            3.5526606717518376f,
            850.6418895294337f,
            747.5497754610369f
        );
        params.target = Vec3(
            237.07531671546286f,
            52.9718477246937f,
            -263.9479752910547f
        );
        params.up = Vec3(
            0.1370609562125062f,
            0.7929456689992689f,
            -0.5936762251407878f
        );
        break;
    }
    case Cam::DunesACam: {
        params.origin = Vec3(
            -71.3357952505853f,
            78.734578313642f,
            108.92994817257102f
        );
        params.target = Vec3(
            -271.1048187567149f,
            80.80085405252899f,
            -297.0543150237934f
        );
        params.up = Vec3(
            0.002016176422133449f,
            0.9999895730713552f,
            0.004097411524808375f
        );
        break;
    }
    case Cam::GrassCam: {
        params.origin = Vec3(
            -5.171248679251219f,
            20.334400261222573f,
            -89.97306056602213f
        );
        params.target = Vec3(
            18.549566601169055f,
            8.00826275343514f,
            -107.84797699232291f
        );
        params.up = Vec3(
            0.3061184080739167f,
            0.9236231643732346f,
            -0.23067676621511765f
        );
        break;
    }
    case Cam::PalmsCam: {
        params.origin = Vec3(
            -124.02546471925854f,
            405.62214562283157f,
            369.1730463283022f
        );
        params.target = Vec3(
            472.3129244023174f,
            571.462848388009f,
            16.499506125608377f
        );
        params.up = Vec3(
            -0.2003759454709524f,
            0.9725259665269878f,
            0.11850200381162401f
        );
        break;
    }
    case Cam::RootsCam: {
        params.origin = Vec3(
            -53.247679762217224f,
            63.459326699391625f,
            -57.57774331317834f
        );
        params.target = Vec3(
            -44.0959264818672f,
            56.9923848671219f,
            -68.5620187630634f
        );
        params.up = Vec3(
            0.2638047892040311f,
            0.9111276124080613f,
            -0.31662834222571157f
        );
        break;
    }
    case Cam::ShotCam: {
        params.origin = Vec3(
            -1139.01589265f,
            23.28673313185658f,
            1479.7947229f
        );
        params.target = Vec3(
            244.81433650665076f,
            238.8071478842799f,
            560.3801168449178f
        );
        params.up = Vec3(
            -0.10714942339176316f,
            0.9916909792130254f,
            0.07118990669600059f
        );
        break;
    }
    default: {
        throw std::runtime_error("Cam not supported");
    }
    }

    return Camera(
        params.origin,
        params.target,
        params.up,
        24.386729394448643f,
        { width, height },
        false
    );

}

}
