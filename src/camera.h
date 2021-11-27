#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>

#include "ray.h"
#include "cuda_helper.h"

class Camera {
public:
    Camera(const glm::vec3 &position, float aspect_ratio);
    Camera(const Camera &other) = default;
    CUDA_CALLABLE_MEMBER Ray getRay(ptrdiff_t x, ptrdiff_t y) const;
private:
    glm::vec3 m_position;
    float m_aspect_ratio;
}; // End of class


inline Camera::Camera(const glm::vec3 &position, float aspect_ratio)
    : m_position(position)
    , m_aspect_ratio(aspect_ratio)
{}

#endif // CAMERA_H
