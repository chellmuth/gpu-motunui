#include "moana/parsers/obj_parser.hpp"

#include <fstream>
#include <regex>

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
        parseLine(line);
    }

    ObjResult result;
    result.vertices = m_vertices;
    result.vertexCount = m_vertices.size() / 3.f;

    result.indices = m_indices;
    result.indexTripletCount = m_indices.size() / 3.f;

    return result;
}

void ObjParser::parseLine(std::string &line)
{
    if (line.empty()) { return; }

    std::string::size_type spaceIndex = line.find_first_of(" \t");
    if (spaceIndex == std::string::npos) { return; }

    std::string command = line.substr(0, spaceIndex);
    if (command[0] == '#') { return; }

    std::string rest = StringUtil::lTrim(line.substr(spaceIndex + 1));

    if (command == "v") {
        processVertex(rest);
    } else if (command == "vn") {
        processNormal(rest);
    } else if (command == "f") {
        processFace(rest);
    }
}

void ObjParser::processVertex(std::string &vertexArgs)
{
    std::string::size_type index = 0;
    std::string rest = vertexArgs;

    float x = std::stof(rest, &index);

    rest = rest.substr(index);
    float y = std::stof(rest, &index);

    rest = rest.substr(index);
    float z = std::stof(rest, &index);

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

void ObjParser::processFace(std::string &faceArgs)
{
    if (processDoubleFaceVertexAndNormal(faceArgs)) { return; }
    throw std::runtime_error("Unsupported face pattern: " + faceArgs);
}

bool ObjParser::processDoubleFaceVertexAndNormal(const std::string &faceArgs)
{
    static std::regex expression("(-?\\d+)//(-?\\d+) (-?\\d+)//(-?\\d+) (-?\\d+)//(-?\\d+) (-?\\d+)//(-?\\d+)\\s*");
    std::smatch match;
    std::regex_match(faceArgs, match, expression);

    if (match.empty()) {
        return false;
    }

    int vertexIndex0 = std::stoi(match[1]);
    int vertexIndex1 = std::stoi(match[3]);
    int vertexIndex2 = std::stoi(match[5]);
    int vertexIndex3 = std::stoi(match[7]);

    int normalIndex0 = std::stoi(match[2]);
    int normalIndex1 = std::stoi(match[4]);
    int normalIndex2 = std::stoi(match[6]);
    int normalIndex3 = std::stoi(match[8]);

    processTriangle(
        vertexIndex0, vertexIndex1, vertexIndex2,
        normalIndex0, normalIndex1, normalIndex2
    );
    processTriangle(
        vertexIndex0, vertexIndex2, vertexIndex3,
        normalIndex0, normalIndex2, normalIndex3
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
