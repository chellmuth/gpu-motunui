#include <iostream>
#include <string>

#include "moana/driver.hpp"
#include "moana/parsers/obj_parser.hpp"

int main(int argc, char *argv[])
{
    std::cout << "Moana!" << std::endl;

    moana::ObjParser parser("/home/cjh/workpad/moana/island/obj/isHibiscus/isHibiscus.obj");
    auto result = parser.parse();

    moana::Driver driver;
    driver.init(result);
    driver.launch();

    exit(0);
}
