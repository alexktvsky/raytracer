#ifndef COLOR_H
#define COLOR_H

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <limits>
#include <climits> // UCHAR_MAX
#include <glm/glm.hpp>

#include "cuda_helper.h"

class Color {
public:
    CUDA_CALLABLE_MEMBER constexpr Color(uint8_t r = 0, uint8_t g = 0, uint8_t b = 0);
    CUDA_CALLABLE_MEMBER constexpr Color(const glm::vec3 &vec);
    CUDA_CALLABLE_MEMBER constexpr Color(const Color &other) = default;
    CUDA_CALLABLE_MEMBER constexpr uint8_t getR(void) const;
    CUDA_CALLABLE_MEMBER constexpr uint8_t getG(void) const;
    CUDA_CALLABLE_MEMBER constexpr uint8_t getB(void) const;
    CUDA_CALLABLE_MEMBER constexpr void setR(uint8_t r);
    CUDA_CALLABLE_MEMBER constexpr void setG(uint8_t g);
    CUDA_CALLABLE_MEMBER constexpr void setB(uint8_t b);
    CUDA_CALLABLE_MEMBER constexpr Color operator+(const Color &other) const;
    CUDA_CALLABLE_MEMBER constexpr Color operator*(float x) const;
    CUDA_CALLABLE_MEMBER constexpr Color operator*(const Color &other) const;
    CUDA_CALLABLE_MEMBER constexpr Color &operator=(const Color &other) = default;
private:
    CUDA_CALLABLE_MEMBER constexpr uint8_t convertValue(float val) const;
private:
    uint8_t m_r;
    uint8_t m_g;
    uint8_t m_b;
}; // End of class


constexpr Color::Color(uint8_t r, uint8_t g, uint8_t b)
    : m_r(r)
    , m_g(g)
    , m_b(b)
{}

constexpr Color::Color(const glm::vec3 &vec)
    : m_r(convertValue(vec[0]))
    , m_g(convertValue(vec[1]))
    , m_b(convertValue(vec[2]))
{}

constexpr uint8_t Color::getR(void) const
{
    return m_r;
}

constexpr uint8_t Color::getG(void) const
{
    return m_g;
}

constexpr uint8_t Color::getB(void) const
{
    return m_b;
}

constexpr void Color::setR(uint8_t r)
{
    m_r = r;
}

constexpr void Color::setG(uint8_t g)
{
    m_g = g;
}

constexpr void Color::setB(uint8_t b)
{
    m_b = b;
}

constexpr Color Color::operator*(float x) const
{
    return Color(convertValue(m_r * x), convertValue(m_g * x), convertValue(m_b * x));
}

constexpr Color Color::operator+(const Color &other) const
{
    return Color(convertValue(m_r + other.m_r), convertValue(m_g + other.m_g), convertValue(m_b + other.m_b));
}

constexpr Color Color::operator*(const Color &other) const
{
    return Color(glm::vec3(m_r, m_g, m_b) * glm::vec3(other.m_r, other.m_g, other.m_b));
}

constexpr uint8_t Color::convertValue(float val) const
{
#if defined(__CUDA_ARCH__)
    return max(0, min(static_cast<int>(UCHAR_MAX), static_cast<int>(val)));
#else
    return std::max(0, std::min(static_cast<int>(std::numeric_limits<uint8_t>::max()), static_cast<int>(val)));
#endif
}

#endif // COLOR_H
