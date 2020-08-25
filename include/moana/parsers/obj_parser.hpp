#pragma once

#include <map>
#include <string>
#include <vector>

namespace moana {

struct ObjResult {
    std::vector<float> vertices;
    std::vector<int> indices;

    int vertexCount;
    int indexTripletCount;
};

class ObjParser {
public:
    ObjParser(const std::string &objFilename);

    ObjResult parse();

private:
    void parseLine(std::string &line);

    void processVertex(std::string &vertexArgs);
    void processNormal(std::string &normalArgs);
    void processFace(std::string &faceArgs);

    bool processDoubleFaceVertexAndNormal(const std::string &faceArgs);

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

    std::vector<float> m_vertices;
    std::vector<int> m_indices;
};

}
