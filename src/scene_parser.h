#ifndef SCENE_PARSER_H
#define SCENE_PARSER_H

#include <string>

#include "scene.h"

class SceneParser {
public:
    SceneParser(void);
    virtual Scene parse(const std::string &filename) = 0;
    virtual ~SceneParser(void) = default;
    size_t getPolygonCount(void) const;
    size_t getTrianglesCount(void) const;
protected:
    size_t m_polygon_counter;
    size_t m_triangles_counter;
}; // End of class


inline SceneParser::SceneParser(void)
    : m_polygon_counter(0)
    , m_triangles_counter(0)
{}

inline size_t SceneParser::getPolygonCount(void) const
{
    return m_polygon_counter;
}

inline size_t SceneParser::getTrianglesCount(void) const
{
    return m_triangles_counter;
}

#endif // SCENE_PARSER_H
