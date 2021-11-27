#include <glm/glm.hpp>
#include <happly.h>
#include <earcut.hpp>

#include "ply_parser.h"


PlyParser::PlyParser(void)
{}

Scene PlyParser::parse(const std::string &filename)
{
    happly::PLYData plydata(filename);

    m_points = plydata.getVertexPositions();
    std::vector<std::vector<size_t>> face_indices = plydata.getFaceIndices();

    m_polygon_counter = face_indices.size();
    m_triangles_counter = 0;

    for (const auto &polygon_indices : face_indices) {
        triangulatePolygon(polygon_indices);
    }  

    return m_scene;
}

void PlyParser::triangulatePolygon(const std::vector<size_t> &polygon_indices)
{
    Material red(Color(255, 0, 0), Color(255, 0, 0), Color(255, 0, 0), 10.f);
    Material blue(Color(0, 0, 255), Color(0, 0, 255), Color(0, 0, 255), 5.f);
    Material yellow(Color(255, 255, 0), Color(255, 255, 0), Color(255, 255, 0), 5.f);
    Material mirror(Color(192, 192, 192), Color(192, 192, 192), Color(192, 192, 192), 1.f, 0.8f); // no more then 1
    Material grey(Color(192, 192, 192), Color(192, 192, 192), Color(192, 192, 192));

    constexpr size_t x = 0;
    constexpr size_t y = 1;
    constexpr size_t z = 2;

    size_t index_point1 = polygon_indices[0];
    size_t index_point2 = polygon_indices[1];
    size_t index_point3 = polygon_indices[2];

    glm::vec3 a = glm::vec3(m_points[index_point1][x], m_points[index_point1][y], m_points[index_point1][z]);
    glm::vec3 b = glm::vec3(m_points[index_point2][x], m_points[index_point2][y], m_points[index_point2][z]);
    glm::vec3 c = glm::vec3(m_points[index_point3][x], m_points[index_point3][y], m_points[index_point3][z]);

    glm::vec3 norm = glm::normalize(glm::cross(a - b, a - c));
    glm::vec3 z_basis = norm;

    glm::vec3 x_basis = glm::normalize(glm::cross(norm, b - c));
    glm::vec3 y_basis = glm::normalize(glm::cross(z_basis, x_basis));

    glm::mat4 transform_matrix = {
        {x_basis[x], y_basis[x], z_basis[x], a[x]},
        {x_basis[y], y_basis[y], z_basis[y], a[y]},
        {x_basis[z], y_basis[z], z_basis[z], a[z]},
        {0.f, 0.f, 0.f, 1.f},
    };

    std::vector<std::array<double, 2>> positions2d;

    for (const auto &index : polygon_indices) {
        glm::vec4 point2d = transform_matrix * glm::vec4(m_points[index][x], m_points[index][y], m_points[index][z], 1.f);
        positions2d.push_back({point2d[x], point2d[y]});
    }

    std::vector<std::vector<std::array<double, 2>>> polygon;

    polygon.push_back(positions2d);

    std::vector<size_t> indices = mapbox::earcut<size_t>(polygon);

    for (size_t i = 0; i < indices.size(); i += 3) {

        index_point1 = polygon_indices[indices[i]];
        index_point2 = polygon_indices[indices[i + 1]];
        index_point3 = polygon_indices[indices[i + 2]];

        float x1 = m_points[index_point1][x];
        float y1 = m_points[index_point1][y];
        float z1 = m_points[index_point1][z];

        float x2 = m_points[index_point2][x];
        float y2 = m_points[index_point2][y];
        float z2 = m_points[index_point2][z];

        float x3 = m_points[index_point3][x];
        float y3 = m_points[index_point3][y];
        float z3 = m_points[index_point3][z];

        m_triangles_counter += 1;

        // printf("{%.2f, %.2f, %.2f} -> {%.2f, %.2f, %.2f}\n", m_points[index_point1][x], m_points[index_point1][y], m_points[index_point1][z], x1, y1, z1);
        // printf("{%.2f, %.2f, %.2f} -> {%.2f, %.2f, %.2f}\n", m_points[index_point2][x], m_points[index_point2][y], m_points[index_point2][z], x2, y2, z2);
        // printf("{%.2f, %.2f, %.2f} -> {%.2f, %.2f, %.2f}\n", m_points[index_point3][x], m_points[index_point3][y], m_points[index_point3][z], x3, y3, z3);
        // printf("\n");

        m_scene.addObject(std::make_shared<Triangle>(glm::vec3(x1, y1, z1), glm::vec3(x2, y2, z2), glm::vec3(x3, y3, z3), blue));
    }
}
