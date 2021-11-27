#include <algorithm>

#include "sphere.h"


// #if defined(__CUDA_ARCH__)

// Intersect Sphere::intersect(const Ray &ray) const
// {
//     // Find the ray to the center and its length squared
//     glm::vec3 oc = m_center - ray.getOrigin();
//     float l2oc = glm::dot(oc, oc);

//     float radius2 = m_radius * m_radius;

//     // Calculate the closest approach along the ray to the sphere's center
//     float tca = glm::dot(oc, ray.getDirection());

//     if (tca <= 0.f) {
//         return Intersect();
//     }

//     // Calculate the half chord distance squared
//     float t2hc = radius2 - l2oc + tca * tca;

//     if (t2hc <= 0.f) {
//         return Intersect();
//     }

//     float t;

//     // Calculate the intersection distance
//     if (l2oc < radius2) {
//         // the ray origin is inside the sphere
//         t = tca + __fsqrt_rn(t2hc);
//     }
//     else {
//         // the ray origin is outside the sphere
//         t = tca - __fsqrt_rn(t2hc);
//     }

//     // if (std::abs(t) < 0.00001f) {
//     //     return Intersect();
//     // }

//     glm::vec3 point = glm::vec3(
//         ray.getOrigin()[0] + ray.getDirection()[0] * t,
//         ray.getOrigin()[1] + ray.getDirection()[1] * t,
//         ray.getOrigin()[2] + ray.getDirection()[2] * t);

//     glm::vec3 normal = glm::vec3(
//         (point[0] - m_center[0]) / m_radius,
//         (point[1] - m_center[1]) / m_radius,
//         (point[2] - m_center[2]) / m_radius);

//     return Intersect(t, point, normal, m_material, ray.getDirection());
// }

// #else

// #define INCLUDE_IMPLEMENTATION_FILE
// #include "sphere_cpu_impl.h"

// #endif
