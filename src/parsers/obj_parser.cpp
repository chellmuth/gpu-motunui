#include "moana/parsers/obj_parser.hpp"

#include <assert.h>
#include <iostream>
#include <optional>
#include <regex>

#include "moana/parsers/string_util.hpp"

namespace moana {

ObjParser::ObjParser(
    std::unique_ptr<std::istream> objFilePtr,
    const std::vector<std::string> &mtlLookup
)
    : m_objFilePtr(std::move(objFilePtr)),
      m_mtlLookup(mtlLookup),
      m_faceFormat(ObjFaceFormat::Unknown)
{}

ObjParser::ObjParser(
    const std::string &objFilename,
    const std::vector<std::string> &mtlLookup
) : ObjParser(
        std::unique_ptr<std::istream>(new std::ifstream(objFilename)),
        mtlLookup
    )
{}

ParseState ObjParser::parseHeader(
    std::string_view command,
    std::string_view rest,
    MeshRecord &record
) {
    if (command == "g") {
        assert(rest == "default");

        // don't return the new state until we
        // hit the first unprocessable line
        return ParseState::Header;
    } else if (command == "v") {
        return ParseState::Vertices;
    }

    return ParseState::Header;
}

ParseState ObjParser::parseVertices(
    std::string_view command,
    std::string_view rest,
    MeshRecord &record
) {
    if (command == "v") {
        processVertex(rest, record);
        return ParseState::Vertices;
    } else if (command == "vn") {
        return ParseState::Normals;
    }

    return ParseState::Error;
}

ParseState ObjParser::parseNormals(
    std::string_view command,
    std::string_view rest,
    MeshRecord &record
) {
    if (command == "vn") {
        processNormal(rest, record);
        return ParseState::Normals;
    } else {
        return ParseState::MeshName;
    }
}

ParseState ObjParser::parseMeshName(
    std::string_view command,
    std::string_view rest,
    MeshRecord &record
) {
    if (command == "g") {
        record.name = rest.data();

        return ParseState::Material;
    } else {
        return ParseState::MeshName;
    }
}

ParseState ObjParser::parseMaterial(
    std::string_view command,
    std::string_view rest,
    MeshRecord &record
) {
    if (command == "usemtl") {
        std::string mtlName = rest.data();
        record.hidden = (mtlName == "hidden");

        // Search for the index of the material key
        // If unfound, use the special 0 material
        const auto iter = std::find(m_mtlLookup.begin(), m_mtlLookup.end(), mtlName);
        if (iter == m_mtlLookup.end()) {
            record.materialIndex = 0;
        } else {
            const int index = std::distance(m_mtlLookup.begin(), iter);
            record.materialIndex = index;
        }

        return ParseState::Material;
    } else if (command == "f") {
        return ParseState::Faces;
    }
    return ParseState::Material;
}

ParseState ObjParser::parseFaces(
    std::string_view command,
    std::string_view rest,
    MeshRecord &record
) {
    if (command == "f") {
        processFace(rest, record);
        return ParseState::Faces;
    } else {
        return ParseState::Complete;
    }
}

std::vector<MeshRecord> ObjParser::parse()
{
    std::vector<MeshRecord> records;
    while (m_objFilePtr->good()) {
        MeshRecord record = parseMesh();

        assert(record.vertices.size() % 3 == 0);
        assert(record.normals.size() % 3 == 0);

        m_offsets.verticesOffset += record.vertices.size() / 3;
        m_offsets.normalsOffset += record.normals.size() / 3;

        if (record.hidden) { continue; }

        records.push_back(record);
    }
    return records;
}

MeshRecord ObjParser::parseMesh()
{
    MeshRecord record;
    ParseState state = ParseState::Header;

    std::string lineData;
    while (std::getline(*m_objFilePtr, lineData)) {
        std::string_view line(lineData);
        if (line.empty()) { continue; }

        std::string::size_type spaceIndex = line.find_first_of(" \t");
        if (spaceIndex == std::string::npos) { continue; }

        const std::string_view command = line.substr(0, spaceIndex);
        if (command[0] == '#') { continue; }

        std::optional<std::string_view> rest = StringUtil::lTrim(line.substr(spaceIndex + 1));
        // fixme: account for blank usemtl lines

        if (!rest) { continue; }

        if (state == ParseState::Header) {
            state = parseHeader(command, rest.value(), record);
        }
        if (state == ParseState::Vertices) {
            state = parseVertices(command, rest.value(), record);
        }
        if (state == ParseState::Normals) {
            state = parseNormals(command, rest.value(), record);
        }
        if (state == ParseState::MeshName) {
            state = parseMeshName(command, rest.value(), record);
        }
        if (state == ParseState::Material) {
            state = parseMaterial(command, rest.value(), record);
        }
        if (state == ParseState::Faces) {
            state = parseFaces(command, rest.value(), record);
        }
        if (state == ParseState::Error) {
            throw std::runtime_error("Invalid obj");
        }

        if (state == ParseState::Complete) {
            break;
        }
    }

    assert(record.vertexIndices.size() == record.normalIndices.size());
    assert(record.vertexIndices.size() % 3 == 0);
    record.indexTripletCount = record.vertexIndices.size() / 3;

    std::cout << "  Mesh Geometry:" << std::endl
              << "    Vertex count: " << record.vertices.size() / 3 << std::endl
              << "    Vertex indices count: " << record.vertexIndices.size() / 3 << std::endl
              << "    Normal count: " << record.normals.size() / 3 << std::endl
              << "    Normal indices count: " << record.normalIndices.size() / 3 << std::endl
              << "    Material index: " << record.materialIndex << std::endl
              << "    Hidden: " << record.hidden << std::endl;

    return record;
}

void ObjParser::parseLine(std::string_view &line)
{
}

void ObjParser::processVertex(std::string_view &vertexArgs, MeshRecord &record)
{
    std::string::size_type index = 0;
    std::string_view rest = vertexArgs;

    float x = std::stof(rest.data(), &index);

    rest = rest.substr(index);
    float y = std::stof(rest.data(), &index);

    rest = rest.substr(index);
    float z = std::stof(rest.data(), &index);

    record.vertices.insert(
        record.vertices.end(),
        { x, y, z }
    );
}

void ObjParser::processNormal(std::string_view &normalArgs, MeshRecord &record)
{
    std::string::size_type index = 0;
    std::string_view rest = normalArgs;

    float x = std::stof(rest.data(), &index);

    rest = rest.substr(index);
    float y = std::stof(rest.data(), &index);

    rest = rest.substr(index);
    float z = std::stof(rest.data(), &index);

    record.normals.insert(
        record.normals.end(),
        { x, y, z }
    );
}

static ObjFaceFormat identifyFaceFormat(std::string faceArgs)
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

void ObjParser::processFace(std::string_view &faceArgs, MeshRecord &record)
{
    if (m_faceFormat == ObjFaceFormat::Unknown) {
        m_faceFormat = identifyFaceFormat(std::string(faceArgs));
    }

    switch(m_faceFormat) {
    case ObjFaceFormat::SingleFaceVertexAndNormal: {
        processSingleFaceVertexAndNormal(faceArgs, record);
        return;
    }
    case ObjFaceFormat::DoubleFaceVertexAndNormal: {
        processDoubleFaceVertexAndNormal(faceArgs, record);
        return;
    }
    }

    throw std::runtime_error("Unsupported face pattern: " + std::string(faceArgs));
}

void ObjParser::processSingleFaceVertexAndNormal(
    std::string_view &faceArgs,
    MeshRecord &record
) {
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
        normalIndices[0], normalIndices[1], normalIndices[2],
        record
    );
}

void ObjParser::processDoubleFaceVertexAndNormal(
    std::string_view &faceArgs,
    MeshRecord &record
) {
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
        normalIndices[0], normalIndices[1], normalIndices[2],
        record
    );
    processTriangle(
        vertexIndices[0], vertexIndices[2], vertexIndices[3],
        normalIndices[0], normalIndices[2], normalIndices[3],
        record
    );
}

void ObjParser::processTriangle(
    int vertexIndex0, int vertexIndex1, int vertexIndex2,
    int normalIndex0, int normalIndex1, int normalIndex2,
    MeshRecord &record
) {
    correctIndices(
        record.vertices,
        m_offsets.verticesOffset,
        &vertexIndex0,
        &vertexIndex1,
        &vertexIndex2
    );
    correctIndices(
        record.normals,
        m_offsets.normalsOffset,
        &normalIndex0,
        &normalIndex1,
        &normalIndex2
    );

    record.vertexIndices.insert(
        record.vertexIndices.end(),
        { vertexIndex0, vertexIndex1, vertexIndex2 }
    );

    record.normalIndices.insert(
        record.normalIndices.end(),
        { normalIndex0, normalIndex1, normalIndex2 }
    );
}

template <class T>
void ObjParser::correctIndex(const std::vector<T> &indices, int offset, int *index)
{
    if (*index < 0) {
        *index += indices.size();
    } else {
        *index -= 1;
    }

    *index -= offset;
}

template <class T>
void ObjParser::correctIndices(
    const std::vector<T> &indices,
    int offset,
    int *index0,
    int *index1,
    int *index2
) {
    correctIndex(indices, offset, index0);
    correctIndex(indices, offset, index1);
    correctIndex(indices, offset, index2);
}

}
