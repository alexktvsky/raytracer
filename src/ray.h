#ifndef RAY_H
#define RAY_H

#include <glm/glm.hpp>

#include "cuda_helper.h"

class Ray {
public:
    CUDA_CALLABLE_MEMBER Ray(const glm::vec3 &origin, const glm::vec3 &dir);
    CUDA_CALLABLE_MEMBER Ray(void) = default;
    CUDA_CALLABLE_MEMBER Ray(const Ray &other) = default;
    CUDA_CALLABLE_MEMBER glm::vec3 getOrigin(void) const;
    CUDA_CALLABLE_MEMBER glm::vec3 getDirection(void) const;
private:
    glm::vec3 m_origin;
    glm::vec3 m_dir;
}; // End of class


inline Ray::Ray(const glm::vec3 &origin, const glm::vec3 &dir)
    : m_origin(origin)
    // Such normalization is recommended, otherwise t will represent the
    // distance in terms of the length of the direction vector.
    , m_dir(glm::normalize(dir))
{}

inline glm::vec3 Ray::getOrigin(void) const
{
    return m_origin;
}

inline glm::vec3 Ray::getDirection(void) const
{
    return m_dir;
}

#endif // RAY_H
