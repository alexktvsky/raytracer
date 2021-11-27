#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include "light_source.h"

class DirectionalLight : public LightSource {
public:
    CUDA_CALLABLE_MEMBER DirectionalLight(float intensity, const glm::vec3 &dir);
    CUDA_CALLABLE_MEMBER DirectionalLight(const DirectionalLight &other) = default;
    CUDA_CALLABLE_MEMBER glm::vec3 getDirection(void) const;
    CUDA_CALLABLE_MEMBER virtual Color calculateColor(const Intersect &intersect) const override;
    CUDA_CALLABLE_MEMBER DirectionalLight &operator=(const DirectionalLight &other) = default;
    enum : Type { TYPE_ID = 2 };
    CUDA_CALLABLE_MEMBER virtual Type getType(void) const override;
private:
    glm::vec3 m_dir;
}; // End of class


inline DirectionalLight::DirectionalLight(float intensity, const glm::vec3 &dir)
    : LightSource(intensity)
    , m_dir(dir)
{}

inline glm::vec3 DirectionalLight::getDirection(void) const
{
    return m_dir;
}

inline DirectionalLight::Type DirectionalLight::getType(void) const
{
    return TYPE_ID;
}

#endif // DIRECTIONAL_LIGHT_H
