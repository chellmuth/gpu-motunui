#include <iostream>
#include <string>

#include "moana/driver.hpp"
#include "moana/parsers/obj_parser.hpp"

int main(int argc, char *argv[])
{
    std::cout << "Moana!" << std::endl;

    // moana::Driver driver;
    // driver.init();
    // driver.launch();

    moana::ObjParser parser("/home/cjh/workpad/moana/island/obj/isHibiscus/isHibiscus.obj");
    auto result = parser.parse();

    exit(0);
}
