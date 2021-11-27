#ifndef LIGHT_SOURCE_H
#define LIGHT_SOURCE_H

#include <glm/glm.hpp>

#include "intersect.h"
#include "color.h"
#include "cuda_helper.h"

class LightSource {
public:
    CUDA_CALLABLE_MEMBER LightSource(float intensity);
    CUDA_CALLABLE_MEMBER LightSource(void) = default;
    CUDA_CALLABLE_MEMBER LightSource(const LightSource &other) = default;
    CUDA_CALLABLE_MEMBER float getIntensity(void) const;
    CUDA_CALLABLE_MEMBER void setIntensity(float intensity);
    CUDA_CALLABLE_MEMBER virtual Color calculateColor(const Intersect &intersect) const = 0;
    using Type = uint8_t;
    CUDA_CALLABLE_MEMBER virtual Type getType(void) const = 0;
    virtual ~LightSource(void) = default;
protected:
    float m_intensity;
}; // End of class


inline LightSource::LightSource(float intensity)
    : m_intensity(intensity)
{}

inline float LightSource::getIntensity(void) const
{
    return m_intensity;
}

inline void LightSource::setIntensity(float intensity)
{
    m_intensity = intensity;
}


#endif // LIGHT_SOURCE_H
