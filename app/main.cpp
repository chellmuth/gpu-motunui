#include <iostream>
#include <string>

#include "moana/driver.hpp"
#include "moana/scene.hpp"

int main(int argc, char *argv[])
{
    using namespace moana;

    std::cout << "Moana!" << std::endl;

    Driver driver;
    driver.init();

    driver.launch(Cam::ShotCam, "shot.exr");
    driver.launch(Cam::BeachCam, "beach.exr");
    driver.launch(Cam::BirdseyeCam, "birdseye.exr");
    driver.launch(Cam::DunesACam, "dunesA.exr");
    driver.launch(Cam::GrassCam, "grass.exr");
    driver.launch(Cam::PalmsCam, "palms.exr");
    driver.launch(Cam::RootsCam, "roots.exr");

    exit(0);
}
