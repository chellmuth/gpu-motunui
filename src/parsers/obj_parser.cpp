#include "moana/parsers/obj_parser.hpp"

#include <iostream>
#include <fstream>
#include <regex>
#include <optional>

#include "moana/parsers/string_util.hpp"

namespace moana {

ObjParser::ObjParser(
    const std::string &objFilename,
    const std::vector<std::string> &mtlLookup
)
    : m_objFilename(objFilename),
      m_mtlLookup(mtlLookup),
      m_faceFormat(ObjFaceFormat::Unknown)
{}

ObjResult ObjParser::parse()
{
    std::ifstream objFile(m_objFilename);

    for (const auto &mtlName : m_mtlLookup) {
        m_nestedIndices.push_back({});
    }
    if (m_nestedIndices.empty()) {
        m_nestedIndices.push_back({}); // fixme: catch-all material
    }

    std::string line;
    while(std::getline(objFile, line)) {
        std::string_view lineView(line);
        parseLine(lineView);
    }

    ObjResult result;
    result.vertices = m_vertices;
    result.vertexCount = m_vertices.size() / 3;

    result.buildInputResults = {};

    int totalIndexTripletCount = 0;
    for (const auto &indices : m_nestedIndices) {
        BuildInputResult buildInputResult;
        buildInputResult.indices = indices;
        buildInputResult.indexTripletCount = indices.size() / 3;
        totalIndexTripletCount += buildInputResult.indexTripletCount;

        result.buildInputResults.push_back(buildInputResult);
    }

    std::cout << "  Geometry:" << std::endl
              << "    Vertex count: " << result.vertexCount << std::endl
              << "    Build Inputs count: " << result.buildInputResults.size() << std::endl
              << "    Index triplet count: " << totalIndexTripletCount << std::endl;

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
    } else if (command == "usemtl") {
        const std::string key = rest.value().data();
        const auto iter = std::find(m_mtlLookup.begin(), m_mtlLookup.end(), key);
        if (iter == m_mtlLookup.end()) {
            m_currentMtlIndex = 0;
        } else {
            int index = std::distance(m_mtlLookup.begin(), iter);
            m_currentMtlIndex = index;
        }

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

static ObjFaceFormat identityFaceFormat(std::string faceArgs)
{
    {
        static std::regex expression("^(-?\\d+)//(-?\\d+) (-?\\d+)//(-?\\d+) (-?\\d+)//(-?\\d+)\\s*");
        std::smatch match;
        std::regex_match(faceArgs, match, expression);

        if (!match.empty()) {
            return ObjFaceFormat::SingleFaceVertexAndNormal;
        }
    }
    {
        static std::regex expression("^(-?\\d+)//(-?\\d+) (-?\\d+)//(-?\\d+) (-?\\d+)//(-?\\d+) (-?\\d+)//(-?\\d+)\\s*");
        std::smatch match;
        std::regex_match(faceArgs, match, expression);

        if (!match.empty()) {
            return ObjFaceFormat::DoubleFaceVertexAndNormal;
        }
    }

    throw std::runtime_error("Unsupported face format: " + faceArgs);
}

void ObjParser::processFace(std::string_view &faceArgs)
{
    if (m_faceFormat == ObjFaceFormat::Unknown) {
        m_faceFormat = identityFaceFormat(std::string(faceArgs));
    }

    switch(m_faceFormat) {
    case ObjFaceFormat::SingleFaceVertexAndNormal: {
        processSingleFaceVertexAndNormal(faceArgs);
        return;
    }
    case ObjFaceFormat::DoubleFaceVertexAndNormal: {
        processDoubleFaceVertexAndNormal(faceArgs);
        return;
    }
    }

    throw std::runtime_error("Unsupported face pattern: " + std::string(faceArgs));
}

void ObjParser::processSingleFaceVertexAndNormal(std::string_view &faceArgs)
{
    int vertexIndices[3];
    int normalIndices[3];

    std::size_t pos;
    for (int i = 0; i < 3; i++) {
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
}

void ObjParser::processDoubleFaceVertexAndNormal(std::string_view &faceArgs)
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
}

void ObjParser::processTriangle(
    int vertexIndex0, int vertexIndex1, int vertexIndex2,
    int normalIndex0, int normalIndex1, int normalIndex2
) {
    correctIndices(m_vertices, &vertexIndex0, &vertexIndex1, &vertexIndex2);
    // correctIndices(m_normals, &normalIndex0, &normalIndex1, &normalIndex2);

    std::vector<int> &indices = m_nestedIndices[m_currentMtlIndex];
    indices.insert(
        indices.end(),
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
