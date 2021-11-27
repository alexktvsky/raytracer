#ifndef AMBIENT_LIGHT_H
#define AMBIENT_LIGHT_H

#include "light_source.h"

class AmbientLight : public LightSource {
public:
    CUDA_CALLABLE_MEMBER AmbientLight(float intensity);
    CUDA_CALLABLE_MEMBER AmbientLight(const AmbientLight &other) = default;
    CUDA_CALLABLE_MEMBER AmbientLight &operator=(const AmbientLight &other) = default;
    CUDA_CALLABLE_MEMBER virtual Color calculateColor(const Intersect &intersect) const override;
    enum : Type { TYPE_ID = 1 };
    CUDA_CALLABLE_MEMBER virtual Type getType(void) const override;
}; // End of class


inline AmbientLight::AmbientLight(float intensity)
    : LightSource(intensity)
{}

inline AmbientLight::Type AmbientLight::getType(void) const
{
    return TYPE_ID;
}

#endif // AMBIENT_LIGHT_H
