#include <iostream>
#include <string>

#include "moana/driver.hpp"
#include "moana/scene.hpp"
#include "moana/types.hpp"

int main(int argc, char *argv[])
{
    using namespace moana;

    std::cout << "Moana!" << std::endl;

    Driver driver;
    driver.init();

    RenderRequest request;
    request.spp = 1;
    request.bounces = 1;

    driver.launch(request, Cam::ShotCam, "shot.exr");
    // driver.launch(Cam::BeachCam, "beach.exr");
    // driver.launch(Cam::BirdseyeCam, "birdseye.exr");
    // driver.launch(Cam::DunesACam, "dunesA.exr");
    // driver.launch(Cam::GrassCam, "grass.exr");
    // driver.launch(Cam::PalmsCam, "palms.exr");
    // driver.launch(Cam::RootsCam, "roots.exr");

    return 0;
}
