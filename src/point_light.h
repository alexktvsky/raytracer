#ifndef POINT_LIGHT_H
#define POINT_LIGHT_H

#include "light_source.h"

class PointLight : public LightSource {
public:
    CUDA_CALLABLE_MEMBER PointLight(float intensity, const glm::vec3 &pos);
    CUDA_CALLABLE_MEMBER PointLight(void) = default;
    CUDA_CALLABLE_MEMBER PointLight(const PointLight &other) = default;
    CUDA_CALLABLE_MEMBER glm::vec3 getPosition(void) const;
    CUDA_CALLABLE_MEMBER virtual Color calculateColor(const Intersect &intersect) const override;
    enum : Type { TYPE_ID = 3 };
    CUDA_CALLABLE_MEMBER virtual Type getType(void) const override;
private:
    glm::vec3 m_pos;
}; // End of class


inline PointLight::PointLight(float intensity, const glm::vec3 &pos)
    : LightSource(intensity)
    , m_pos(pos)
{}

inline glm::vec3 PointLight::getPosition(void) const
{
    return m_pos;
}

inline PointLight::Type PointLight::getType(void) const
{
    return TYPE_ID;
}

#endif // POINT_LIGHT_H
