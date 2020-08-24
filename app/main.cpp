#include <iostream>

#include "moana/driver.hpp"

int main(int argc, char *argv[])
{
    std::cout << "Moana!" << std::endl;

    moana::Driver driver;
    driver.init();
    driver.launch();

    exit(0);
}
