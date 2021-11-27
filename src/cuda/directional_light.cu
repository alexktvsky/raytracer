#include "directional_light.h"

#if defined(__CUDA_ARCH__)

Color DirectionalLight::calculateColor(const Intersect &intersect) const
{
    float diffuse_light_intensity = 0.f;
    glm::vec3 l = m_dir;
    float n_dot_l = glm::dot(intersect.getNormal(), l);
    if (n_dot_l > 0.f) {
        diffuse_light_intensity = m_intensity * n_dot_l / (glm::length(intersect.getNormal()) * glm::length(l));
    }

    float specular_factor = 0.f;
    glm::vec3 r = 2.f * intersect.getNormal() * n_dot_l - l;
    glm::vec3 d = -intersect.getDirection();
    float r_dot_v = glm::dot(r, d);
    if (r_dot_v > 0.f) {
        specular_factor += m_intensity * __powf(r_dot_v/(glm::length(r) * glm::length(d)), intersect.getMaterial().getShininess());
    }

    return intersect.getMaterial().getDiffuse() * diffuse_light_intensity + intersect.getMaterial().getSpecular() * specular_factor;
}

#else

#define INCLUDE_IMPLEMENTATION_FILE
#include "directional_light_cpu_impl.h"

#endif
