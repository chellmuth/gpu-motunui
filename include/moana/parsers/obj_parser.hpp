#pragma once

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace moana {

struct BuildInputResult {
    std::vector<int> indices;
    int indexTripletCount;
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

class ObjParser {
public:
    ObjParser(
        const std::string &objFilename,
        const std::vector<std::string> &mtlLookup
    );

    ObjResult parse();

private:
    void parseLine(std::string_view &line);

    void processVertex(std::string_view &vertexArgs);
    void processNormal(std::string &normalArgs);
    void processFace(std::string_view &faceArgs);

    void processSingleFaceVertexAndNormal(std::string_view &faceArgs);
    void processDoubleFaceVertexAndNormal(std::string_view &faceArgs);

    void processTriangle(
        int vertexIndex0, int vertexIndex1, int vertexIndex2,
        int normalIndex0, int normalIndex1, int normalIndex2
    );

    template <class T>
    void correctIndex(const std::vector<T> &indices, int *index);

    template <class T>
    void correctIndices(
        const std::vector<T> &indices,
        int *index0,
        int *index1,
        int *index2
    );

    std::string m_objFilename;

    ObjFaceFormat m_faceFormat;
    std::vector<float> m_vertices;
    std::vector<std::vector<int> > m_nestedIndices;

    int m_currentMtlIndex = -1;
    const std::vector<std::string> m_mtlLookup;

    bool m_skipFaces;
};

}
