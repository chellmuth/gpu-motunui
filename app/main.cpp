#include <iostream>
#include <string>

#include "moana/driver.hpp"
#include "moana/parsers/obj_parser.hpp"

int main(int argc, char *argv[])
{
    std::cout << "Moana!" << std::endl;

    std::string root(MOANA_ROOT);
    std::string path(root + "/island/obj/isHibiscus/isHibiscus.obj");
    moana::ObjParser parser(path);
    auto result = parser.parse();

    moana::Driver driver;
    driver.init(result);
    driver.launch();

    exit(0);
}
