#ifndef SPHERE_H
#define SPHERE_H

#include "graph_object.h"
#include "material.h"

class Sphere : public GraphObject {
public:
    CUDA_CALLABLE_MEMBER Sphere(const glm::vec3 &center, float radius, const Material &material);
    CUDA_CALLABLE_MEMBER Sphere(void) = default;
    CUDA_CALLABLE_MEMBER Sphere(const Sphere &other) = default;
    CUDA_CALLABLE_MEMBER virtual Intersect intersect(const Ray &ray) const override;
    CUDA_CALLABLE_MEMBER glm::vec3 getCenter(void) const;
    CUDA_CALLABLE_MEMBER float getRadius(void) const;
    CUDA_CALLABLE_MEMBER Material getMaterial(void) const;
    enum : Type { TYPE_ID = 1 };
    CUDA_CALLABLE_MEMBER virtual Type getType(void) const override;
private:
    glm::vec3 m_center;
    float m_radius;
    Material m_material;
}; // End of class


inline Sphere::Sphere(const glm::vec3 &center, float radius, const Material &material)
    : m_center(center)
    , m_radius(radius)
    , m_material(material)
{}

#if defined(__CUDA_ARCH__)

inline Intersect Sphere::intersect(const Ray &ray) const
{
    // Find the ray to the center and its length squared
    glm::vec3 oc = m_center - ray.getOrigin();
    float l2oc = glm::dot(oc, oc);

    float radius2 = m_radius * m_radius;

    // Calculate the closest approach along the ray to the sphere's center
    float tca = glm::dot(oc, ray.getDirection());

    if (tca <= 0.f) {
        return Intersect();
    }

    // Calculate the half chord distance squared
    float t2hc = radius2 - l2oc + tca * tca;

    if (t2hc <= 0.f) {
        return Intersect();
    }

    float t;

    // Calculate the intersection distance
    if (l2oc < radius2) {
        // the ray origin is inside the sphere
        t = tca + __fsqrt_rn(t2hc);
    }
    else {
        // the ray origin is outside the sphere
        t = tca - __fsqrt_rn(t2hc);
    }

    // if (std::abs(t) < 0.00001f) {
    //     return Intersect();
    // }

    glm::vec3 point = glm::vec3(
        ray.getOrigin()[0] + ray.getDirection()[0] * t,
        ray.getOrigin()[1] + ray.getDirection()[1] * t,
        ray.getOrigin()[2] + ray.getDirection()[2] * t);

    glm::vec3 normal = glm::vec3(
        (point[0] - m_center[0]) / m_radius,
        (point[1] - m_center[1]) / m_radius,
        (point[2] - m_center[2]) / m_radius);

    return Intersect(t, point, normal, m_material, ray.getDirection());
}

#else


inline Intersect Sphere::intersect(const Ray &ray) const
{
    // Find the ray to the center and its length squared
    glm::vec3 oc = m_center - ray.getOrigin();
    float l2oc = glm::dot(oc, oc);

    float radius2 = m_radius * m_radius;

    // Calculate the closest approach along the ray to the sphere's center
    float tca = glm::dot(oc, ray.getDirection());

    if (tca <= 0.f) {
        return Intersect();
    }

    // Calculate the half chord distance squared
    float t2hc = radius2 - l2oc + tca * tca;

    if (t2hc <= 0.f) {
        return Intersect();
    }

    float t;

    // Calculate the intersection distance
    if (l2oc < radius2) {
        // the ray origin is inside the sphere
        t = tca + std::sqrt(t2hc);
    }
    else {
        // the ray origin is outside the sphere
        t = tca - std::sqrt(t2hc);
    }

    // if (std::abs(t) < 0.00001f) {
    //     return Intersect();
    // }

    glm::vec3 point = glm::vec3(
        ray.getOrigin()[0] + ray.getDirection()[0] * t,
        ray.getOrigin()[1] + ray.getDirection()[1] * t,
        ray.getOrigin()[2] + ray.getDirection()[2] * t);

    glm::vec3 normal = glm::vec3(
        (point[0] - m_center[0]) / m_radius,
        (point[1] - m_center[1]) / m_radius,
        (point[2] - m_center[2]) / m_radius);

    return Intersect(t, point, normal, m_material, ray.getDirection());
}

#endif


inline glm::vec3 Sphere::getCenter(void) const
{
    return m_center;
}

inline float Sphere::getRadius(void) const
{
    return m_radius;
}

inline Material Sphere::getMaterial(void) const
{
    return m_material;
}

inline Sphere::Type Sphere::getType(void) const
{
    return TYPE_ID;
}

#endif // SPHERE_H
