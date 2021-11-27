#ifndef FBX_PARSER_H
#define FBX_PARSER_H

#include <vector>
#include <array>
#include <fbxsdk.h>

#include "scene_parser.h"

class FbxParser : public SceneParser {
public:
    FbxParser(void);
    virtual Scene parse(const std::string &filename);
private:
    void processNode(const FbxNode *node);
    void processAttribute(const FbxNodeAttribute *attr);
    void processMesh(const FbxMesh *mesh);
    void processPolygon(const FbxMesh *mesh, int polygon_index);
    FbxAMatrix calculateGlobalTransform(FbxNode *node);
    void processLightSource(const FbxLight *light);
private:
    Scene m_scene;
}; // End of class



#endif // FBX_PARSER_H
