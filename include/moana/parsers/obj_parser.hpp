#pragma once

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace moana {

struct BuildInputResult {
    std::vector<int> indices = {};
    int indexTripletCount = 0;
};

struct MeshRecord {
    std::vector<float> vertices = {};
    std::vector<int> vertexIndices = {};

    std::vector<float> normals = {};
    std::vector<int> normalIndices = {};

    int indexTripletCount = 0;

    int materialIndex = 0;
    bool hidden = true;
};

struct ObjResult {
    std::vector<float> vertices;
    std::vector<BuildInputResult> buildInputResults;

    int vertexCount;
};

enum class ObjFaceFormat {
    DoubleFaceVertexAndNormal,
    SingleFaceVertexAndNormal,
    Unknown
};

enum class ParseState {
    Header = 0,
    Vertices,
    Normals,
    MeshName,
    Material,
    Faces,
    Complete,
    Error,
};

class ObjParser {
public:
    ObjParser(
        std::unique_ptr<std::istream> objFilePtr,
        const std::vector<std::string> &mtlLookup
    );

    ObjParser(
        const std::string &objFilename,
        const std::vector<std::string> &mtlLookup
    );

    ObjResult parse();
    std::vector<MeshRecord> parseMeshes(); // fixme
    MeshRecord parseMesh(); // fixme

private:
    struct GeometryOffsets {
        int verticesOffset = 0;
        int normalsOffset = 0;
    };

    void parseLine(std::string_view &line);

    void processVertex(std::string_view &vertexArgs, MeshRecord &record);
    void processNormal(std::string_view &normalArgs, MeshRecord &record);
    void processFace(std::string_view &faceArgs, MeshRecord &record);

    void processSingleFaceVertexAndNormal(std::string_view &faceArgs, MeshRecord &record);
    void processDoubleFaceVertexAndNormal(std::string_view &faceArgs, MeshRecord &record);

    void processTriangle(
        int vertexIndex0, int vertexIndex1, int vertexIndex2,
        int normalIndex0, int normalIndex1, int normalIndex2,
        MeshRecord &record
    );

    template <class T>
    void correctIndex(const std::vector<T> &indices, int offset, int *index);

    template <class T>
    void correctIndices(
        const std::vector<T> &indices,
        int offset,
        int *index0,
        int *index1,
        int *index2
    );

    ParseState parseHeader(
        std::string_view command,
        std::string_view rest,
        MeshRecord &record
    );

    ParseState parseVertices(
        std::string_view command,
        std::string_view rest,
        MeshRecord &record
    );

    ParseState parseNormals(
        std::string_view command,
        std::string_view rest,
        MeshRecord &record
    );

    ParseState parseMeshName(
        std::string_view command,
        std::string_view rest,
        MeshRecord &record
    );

    ParseState parseMaterial(
        std::string_view command,
        std::string_view rest,
        MeshRecord &record
    );

    ParseState parseFaces(
        std::string_view command,
        std::string_view rest,
        MeshRecord &record
    );

    std::unique_ptr<std::istream> m_objFilePtr;
    const std::vector<std::string> m_mtlLookup;

    ObjFaceFormat m_faceFormat;

    GeometryOffsets m_offsets;
};

}
