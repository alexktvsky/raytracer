// This file contains common implementation for CPU.
//
// The idea is that particular .cpp and .cu files simply include this file
// without worry of copying actual implementation over.

#ifndef INCLUDE_IMPLEMENTATION_FILE
#error "camera_cpu_impl.h contains implementation and must not be used directly."
#endif


Ray Camera::getRay(ptrdiff_t x, ptrdiff_t y) const
{
    float d = 1.f;
    float v_x = static_cast<float>(x) / m_aspect_ratio;
    float v_y = static_cast<float>(y) / m_aspect_ratio;
    return Ray(m_position, glm::vec3(v_x, v_y, d));
}
