#ifndef PLANE_H
#define PLANE_H

#include "graph_object.h"
#include "material.h"

class Plane : public GraphObject {
public:
    CUDA_CALLABLE_MEMBER Plane(const glm::vec3 &abc, float d, const Material &material);
    CUDA_CALLABLE_MEMBER Plane(void) = default;
    CUDA_CALLABLE_MEMBER Plane(const Plane &other) = default;
    CUDA_CALLABLE_MEMBER virtual Intersect intersect(const Ray &ray) const override;
    CUDA_CALLABLE_MEMBER glm::vec3 getAbc(void) const;
    CUDA_CALLABLE_MEMBER float getD(void) const;
    CUDA_CALLABLE_MEMBER Material getMaterial(void) const;
    enum : Type { TYPE_ID = 2 };
    CUDA_CALLABLE_MEMBER virtual Type getType(void) const override;
private:
    glm::vec3 m_abc;
    float m_d;
    Material m_material;
}; // End of class


inline Plane::Plane(const glm::vec3 &abc, float d, const Material &material)
    : m_abc(abc)
    , m_d(d)
    , m_material(material)
{}

inline glm::vec3 Plane::getAbc(void) const
{
    return m_abc;
}

inline float Plane::getD(void) const
{
    return m_d;
}

inline Material Plane::getMaterial(void) const
{
    return m_material;
}

inline Plane::Type Plane::getType(void) const
{
    return TYPE_ID;
}

#endif // PLANE_H
