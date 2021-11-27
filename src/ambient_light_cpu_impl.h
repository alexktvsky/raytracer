// This file contains common implementation for CPU.
//
// The idea is that particular .cpp and .cu files simply include this file
// without worry of copying actual implementation over.

#ifndef INCLUDE_IMPLEMENTATION_FILE
#error "ambient_light_cpu_impl.h contains implementation and must not be used directly."
#endif


Color AmbientLight::calculateColor(const Intersect &intersect) const
{
    return intersect.getMaterial().getAmbient() * m_intensity;
}
