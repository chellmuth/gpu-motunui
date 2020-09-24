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
    request.spp = 128;
    request.bounces = 2;
    request.width = 1024;
    request.height = 429;

    driver.launch(request, Cam::ShotCam, "shot.exr");
    driver.launch(request, Cam::BeachCam, "beach.exr");
    driver.launch(request, Cam::BirdseyeCam, "birdseye.exr");
    driver.launch(request, Cam::DunesACam, "dunesA.exr");
    driver.launch(request, Cam::GrassCam, "grass.exr");
    driver.launch(request, Cam::PalmsCam, "palms.exr");
    driver.launch(request, Cam::RootsCam, "roots.exr");

    return 0;
}
