#include "moana/parsers/obj_parser.hpp"

#include <fstream>
#include <optional>

#include "moana/parsers/string_util.hpp"

namespace moana {

ObjParser::ObjParser(const std::string &objFilename)
    : m_objFilename(objFilename)
{}

ObjResult ObjParser::parse()
{
    std::ifstream objFile(m_objFilename);

    std::string line;
    while(std::getline(objFile, line)) {
        std::string_view lineView(line);
        parseLine(lineView);
    }

    ObjResult result;
    result.vertices = m_vertices;
    result.vertexCount = m_vertices.size() / 3.f;

    result.indices = m_indices;
    result.indexTripletCount = m_indices.size() / 3.f;

    return result;
}

void ObjParser::parseLine(std::string_view &line)
{
    if (line.empty()) { return; }

    std::string::size_type spaceIndex = line.find_first_of(" \t");
    if (spaceIndex == std::string::npos) { return; }

    std::string_view command = line.substr(0, spaceIndex);
    if (command[0] == '#') { return; }

    std::optional<std::string_view> rest = StringUtil::lTrim(line.substr(spaceIndex + 1));
    if (!rest) { return; }

    if (command == "v") {
        processVertex(rest.value());
    // } else if (command == "vn") {
    //     processNormal(rest);
    } else if (command == "f") {
        processFace(rest.value());
    }
}

void ObjParser::processVertex(std::string_view &vertexArgs)
{
    std::string::size_type index = 0;
    std::string_view rest = vertexArgs;

    float x = std::stof(rest.data(), &index);

    rest = rest.substr(index);
    float y = std::stof(rest.data(), &index);

    rest = rest.substr(index);
    float z = std::stof(rest.data(), &index);

    m_vertices.insert(
        m_vertices.end(),
        { x, y, z }
    );
}

void ObjParser::processNormal(std::string &normalArgs)
{
    std::string::size_type index = 0;
    std::string rest = normalArgs;

    float x = std::stof(rest, &index);

    rest = rest.substr(index);
    float y = std::stof(rest, &index);

    rest = rest.substr(index);
    float z = std::stof(rest, &index);

    // m_normals.push_back(Point(x, y, z));
}

void ObjParser::processFace(std::string_view &faceArgs)
{
    if (processDoubleFaceVertexAndNormal(faceArgs)) { return; }
    throw std::runtime_error("Unsupported face pattern: " + std::string(faceArgs));
}


bool ObjParser::processDoubleFaceVertexAndNormal(std::string_view &faceArgs)
{
    int vertexIndices[4];
    int normalIndices[4];

    std::size_t pos;
    for (int i = 0; i < 4; i++) {
        vertexIndices[i] = std::stoi(faceArgs.data(), &pos);
        faceArgs = faceArgs.substr(pos);

        const std::string::size_type firstSlash = faceArgs.find_first_of("/");
        faceArgs = faceArgs.substr(firstSlash + 1);

        const std::string::size_type secondSlash = faceArgs.find_first_of("/");
        faceArgs = faceArgs.substr(secondSlash + 1);

        normalIndices[i] = std::stoi(faceArgs.data(), &pos);
        faceArgs = faceArgs.substr(pos);
    }

    processTriangle(
        vertexIndices[0], vertexIndices[1], vertexIndices[2],
        normalIndices[0], normalIndices[1], normalIndices[2]
    );
    processTriangle(
        vertexIndices[0], vertexIndices[2], vertexIndices[3],
        normalIndices[0], normalIndices[2], normalIndices[3]
    );

    return true;
}

void ObjParser::processTriangle(
    int vertexIndex0, int vertexIndex1, int vertexIndex2,
    int normalIndex0, int normalIndex1, int normalIndex2
) {
    correctIndices(m_vertices, &vertexIndex0, &vertexIndex1, &vertexIndex2);
    // correctIndices(m_normals, &normalIndex0, &normalIndex1, &normalIndex2);

    m_indices.insert(
        m_indices.end(),
        { vertexIndex0, vertexIndex1, vertexIndex2 }
    );
}

template <class T>
void ObjParser::correctIndex(const std::vector<T> &indices, int *index)
{
    if (*index < 0) {
        *index += indices.size();
    } else {
        *index -= 1;
    }
}

template <class T>
void ObjParser::correctIndices(
    const std::vector<T> &indices,
    int *index0,
    int *index1,
    int *index2
) {
    correctIndex(indices, index0);
    correctIndex(indices, index1);
    correctIndex(indices, index2);
}

}
