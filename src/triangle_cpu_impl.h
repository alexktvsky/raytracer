// This file contains common implementation for CPU.
//
// The idea is that particular .cpp and .cu files simply include this file
// without worry of copying actual implementation over.

#ifndef INCLUDE_IMPLEMENTATION_FILE
#error "triangle_cpu_impl.h contains implementation and must not be used directly."
#endif

#include <algorithm>


Intersect Triangle::intersect(const Ray &ray) const
{
    glm::vec3 e1 = m_b - m_a;
    glm::vec3 e2 = m_c - m_a;

    glm::vec3 normal = glm::cross(e1, e2);

    constexpr float epsilon = 1e-8f;

    float normal_dot_dir = glm::dot(normal, ray.getDirection());

#if defined(__CUDA_ARCH__)
    if (abs(normal_dot_dir) < epsilon) {
#else
    if (std::abs(normal_dot_dir) < epsilon) {
#endif
        // They are parallel so they don't intersect.
        return Intersect();
    }

    float d = glm::dot(normal, m_a); 
    float t = (glm::dot(normal, ray.getOrigin()) + d) / normal_dot_dir; 
    if (t < 0.f) {
        // The triangle is behind.
        return Intersect();
    }

    glm::vec3 p = ray.getOrigin() + t * ray.getDirection();

    // edge 0
    glm::vec3 edge0 = m_b - m_a;
    glm::vec3 vp0 = p - m_a;

    glm::vec3 c = glm::cross(edge0, vp0);

    if (glm::dot(normal, c) < 0.f) {
        // P is on the right side.
        return Intersect();
    }

    // edge 1
    glm::vec3 edge1 = m_c - m_b;
    glm::vec3 vp1 = p - m_b; 
    c = glm::cross(edge1, vp1);
    float u = glm::dot(normal, c);
    if (u < 0.f) {
        // P is on the right side.
        return Intersect();
    }

    // edge 2
    glm::vec3 edge2 = m_a - m_c;
    glm::vec3 vp2 = p - m_c; 
    c = glm::cross(edge2, vp2);
    float v = glm::dot(normal, c);
    if (v < 0.f) {
        // P is on the right side.
        return Intersect();
    }

    glm::vec3 point = glm::vec3(
        ray.getOrigin()[0] + ray.getDirection()[0] * t,
        ray.getOrigin()[1] + ray.getDirection()[1] * t,
        ray.getOrigin()[2] + ray.getDirection()[2] * t);

    return Intersect(t, point, normal, m_material, ray.getDirection());
}
