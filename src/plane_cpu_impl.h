// This file contains common implementation for CPU.
//
// The idea is that particular .cpp and .cu files simply include this file
// without worry of copying actual implementation over.

#ifndef INCLUDE_IMPLEMENTATION_FILE
#error "plane_cpu_impl.h contains implementation and must not be used directly."
#endif


Intersect Plane::intersect(const Ray &ray) const
{
    float vd = glm::dot(m_abc, ray.getDirection());
    float v0 = -(glm::dot(m_abc, ray.getOrigin()) + m_d);

    // Calculate the ratio of the dot products.
    float t = v0 / vd;

    // Line defmed by the ray intersects the plane behind the ray's
    // origin and so no actual intersection occurs.
    if (t < 0.f) {
        return Intersect();
    }

    glm::vec3 point = glm::vec3(
        ray.getOrigin()[0] + ray.getDirection()[0] * t,
        ray.getOrigin()[1] + ray.getDirection()[1] * t,
        ray.getOrigin()[2] + ray.getDirection()[2] * t);

    glm::vec3 normal;

    if (vd < 0.f) {
        normal = m_abc;
    }
    else {
        normal = -m_abc;
    }  

    return Intersect(t, point, normal, m_material, ray.getDirection());
}
