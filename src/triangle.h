#ifndef POLYGON_H
#define POLYGON_H

#include "graph_object.h"
#include "cuda_helper.h"

class Triangle : public GraphObject {
public:
    CUDA_CALLABLE_MEMBER Triangle(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, const Material &material);
    CUDA_CALLABLE_MEMBER Triangle(void) = default;
    CUDA_CALLABLE_MEMBER Triangle(const Triangle &other) = default;
    CUDA_CALLABLE_MEMBER virtual Intersect intersect(const Ray &ray) const override;
    enum : Type { TYPE_ID = 3 };
    CUDA_CALLABLE_MEMBER virtual Type getType(void) const override;
private:
    glm::vec3 m_a;
    glm::vec3 m_b;
    glm::vec3 m_c;
    Material m_material;
}; // End of class


inline Triangle::Triangle(const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, const Material &material)
    : m_a(a)
    , m_b(b)
    , m_c(c)
    , m_material(material)
{}

inline Triangle::Type Triangle::getType(void) const
{
    return TYPE_ID;
}

#endif // POLYGON_H
