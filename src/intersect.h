#ifndef INTERSECT_H
#define INTERSECT_H

#include <glm/glm.hpp>

#include "material.h"
#include "cuda_helper.h"

class Intersect {
public:
    CUDA_CALLABLE_MEMBER Intersect(void);
    CUDA_CALLABLE_MEMBER Intersect(float distance, const glm::vec3 &point, const glm::vec3 &normal, const Material &material, const glm::vec3 &dir);
    CUDA_CALLABLE_MEMBER Intersect(const Intersect &other) = default;
    CUDA_CALLABLE_MEMBER bool isHit(void) const;
    CUDA_CALLABLE_MEMBER float getDistance(void) const;
    CUDA_CALLABLE_MEMBER glm::vec3 getPoint(void) const;
    CUDA_CALLABLE_MEMBER glm::vec3 getNormal(void) const;
    CUDA_CALLABLE_MEMBER Material getMaterial(void) const;
    CUDA_CALLABLE_MEMBER glm::vec3 getDirection(void) const;
private:
    bool m_hit;
    float m_distance;
    glm::vec3 m_point;
    glm::vec3 m_normal;
    Material m_material;
    glm::vec3 m_dir;
}; // End of class


inline Intersect::Intersect(void)
    : m_hit(false)
    , m_distance(0.f)
    , m_material(Color(0, 0, 0), Color(0, 0, 0), Color(0, 0, 0))
{}

inline Intersect::Intersect(float distance, const glm::vec3 &point, const glm::vec3 &normal, const Material &material, const glm::vec3 &dir)
    : m_hit(true), m_distance(distance)
    , m_point(point), m_normal(normal)
    , m_material(material)
    , m_dir(dir)
{}

inline bool Intersect::isHit(void) const
{
    return m_hit;
}

inline float Intersect::getDistance(void) const
{
    return m_distance;
}

inline glm::vec3 Intersect::getPoint(void) const
{
    return m_point;
}

inline glm::vec3 Intersect::getNormal(void) const
{
    return m_normal;
}

inline Material Intersect::getMaterial(void) const
{
    return m_material;
}

inline glm::vec3 Intersect::getDirection(void) const
{
    return m_dir;
}

#endif // INTERSECT_H
