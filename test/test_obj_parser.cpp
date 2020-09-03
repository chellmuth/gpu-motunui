#include <iostream> // fixme
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <catch2/catch.hpp>

#include <moana/parsers/obj_parser.hpp>

using namespace moana;

static std::string obj1 = R"(
g default
v 0 1 2
v 3 4 5
v 6 7 8
v 9 10 11
vn 12 13 14
vn 15 16 17
vn 18 19 20
vn 20 21 22
s 1
g mesh_name
usemtl custom_mtl
f 1//4 2//3 3//2 4//1
)";

TEST_CASE("parse a simple obj", "[obj]") {
    const std::vector<std::string> mtlLookup = { "custom_mtl" };

    ObjParser parser(
        std::make_unique<std::istringstream>(obj1),
        mtlLookup
    );

    auto records = parser.parseMeshes();
    REQUIRE(records.size() == 1);

    auto record = records[0];
    std::vector<float> expectedVertices = {
        0.f, 1.f, 2.f,
        3.f, 4.f, 5.f,
        6.f, 7.f, 8.f,
        9.f, 10.f, 11.f
    };
    REQUIRE(record.vertices == expectedVertices);

    std::vector<int> expectedVertexIndices = {
        0, 1, 2,
        0, 2, 3,
    };
    REQUIRE(record.vertexIndices == expectedVertexIndices);

    std::vector<float> expectedNormals = {
        12.f, 13.f, 14.f,
        15.f, 16.f, 17.f,
        18.f, 19.f, 20.f,
        20.f, 21.f, 22.f,
    };
    REQUIRE(record.normals == expectedNormals);

    std::vector<int> expectedNormalIndices = {
        3, 2, 1,
        3, 1, 0,
    };
    REQUIRE(record.normalIndices == expectedNormalIndices);

    REQUIRE(record.indexTripletCount == 2);
    REQUIRE(record.materialIndex == 0);
    REQUIRE(!record.hidden);
}

static std::string obj2 = R"(
g default
v 0 1 2
v 3 4 5
v 6 7 8
v 9 10 11
vn 12 13 14
vn 15 16 17
vn 18 19 20
vn 20 21 22
s 1
g mesh_name1
usemtl custom_mtl
f 1//4 2//3 3//2 4//1
g default
v 23 24 25
v 26 27 28
v 29 30 31
v 32 33 34
vn 35 36 37
vn 38 39 40
vn 41 42 43
vn 44 45 46
s 1
g mesh_name2
usemtl custom_mtl
f 5//8 6//7 7//6 8//5
)";

TEST_CASE("parse an obj with two meshes", "[obj]") {
    const std::vector<std::string> mtlLookup = { "custom_mtl" };

    ObjParser parser(
        std::make_unique<std::istringstream>(obj2),
        mtlLookup
    );

    auto records = parser.parseMeshes();
    REQUIRE(records.size() == 2);

    {
        auto record = records[0];
        std::vector<float> expectedVertices = {
            0.f, 1.f, 2.f,
            3.f, 4.f, 5.f,
            6.f, 7.f, 8.f,
            9.f, 10.f, 11.f
        };
        REQUIRE(record.vertices == expectedVertices);

        std::vector<int> expectedVertexIndices = {
            0, 1, 2,
            0, 2, 3,
        };
        REQUIRE(record.vertexIndices == expectedVertexIndices);

        std::vector<float> expectedNormals = {
            12.f, 13.f, 14.f,
            15.f, 16.f, 17.f,
            18.f, 19.f, 20.f,
            20.f, 21.f, 22.f,
        };
        REQUIRE(record.normals == expectedNormals);

        std::vector<int> expectedNormalIndices = {
            3, 2, 1,
            3, 1, 0,
        };
        REQUIRE(record.normalIndices == expectedNormalIndices);

        REQUIRE(record.indexTripletCount == 2);
        REQUIRE(record.materialIndex == 0);
        REQUIRE(!record.hidden);
    }
    {
        auto record = records[1];
        std::vector<float> expectedVertices = {
            23.f, 24.f, 25.f,
            26.f, 27.f, 28.f,
            29.f, 30.f, 31.f,
            32.f, 33.f, 34.f,
        };
        REQUIRE(record.vertices == expectedVertices);

        std::vector<int> expectedVertexIndices = {
            0, 1, 2,
            0, 2, 3,
        };
        REQUIRE(record.vertexIndices == expectedVertexIndices);

        std::vector<float> expectedNormals = {
            35.f, 36.f, 37.f,
            38.f, 39.f, 40.f,
            41.f, 42.f, 43.f,
            44.f, 45.f, 46.f,
        };
        REQUIRE(record.normals == expectedNormals);

        std::vector<int> expectedNormalIndices = {
            3, 2, 1,
            3, 1, 0,
        };
        REQUIRE(record.normalIndices == expectedNormalIndices);

        REQUIRE(record.indexTripletCount == 2);
        REQUIRE(record.materialIndex == 0);
        REQUIRE(!record.hidden);
    }
}

static std::string obj3 = R"(
g default
v 0 1 2
v 3 4 5
v 6 7 8
v 9 10 11
vn 12 13 14
vn 15 16 17
vn 18 19 20
vn 20 21 22
s 1
g mesh_name
usemtl hidden
f 1//4 2//3 3//2 4//1
)";

TEST_CASE("mark a hidden mesh", "[obj]") {
    const std::vector<std::string> mtlLookup = {};

    ObjParser parser(
        std::make_unique<std::istringstream>(obj3),
        mtlLookup
    );

    auto records = parser.parseMeshes();
    REQUIRE(records.size() == 1);
    REQUIRE(records[0].hidden);
}

static std::string obj4 = R"(
g default
v 0 1 2
v 3 4 5
v 6 7 8
v 9 10 11
vn 12 13 14
vn 15 16 17
vn 18 19 20
vn 20 21 22
s 1
g mesh_name
f 1//4 2//3 3//2 4//1
)";

TEST_CASE("mark shadow meshes hidden (no usemtl)", "[obj]") {
    const std::vector<std::string> mtlLookup = {};

    ObjParser parser(
        std::make_unique<std::istringstream>(obj4),
        mtlLookup
    );

    auto records = parser.parseMeshes();
    REQUIRE(records.size() == 1);
    REQUIRE(records[0].hidden);
}
