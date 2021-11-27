#ifndef PLY_PARSER_H
#define PLY_PARSER_H

#include <vector>
#include <array>

#include "scene_parser.h"

class PlyParser : public SceneParser {
public:
    PlyParser(void);
    virtual Scene parse(const std::string &filename);
private:
    void triangulatePolygon(const std::vector<size_t> &polygon_indices);
private:
    Scene m_scene;
    std::vector<std::array<double, 3>> m_points;
    size_t m_polygon_counter;
}; // End of class

#endif // PLY_PARSER_H
