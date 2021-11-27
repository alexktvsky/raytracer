#ifndef MATERIAL_H
#define MATERIAL_H

#include "color.h"
#include "cuda_helper.h"

class Material {
public:
    CUDA_CALLABLE_MEMBER constexpr Material(const Color &ambient, const Color &diffuse, const Color &specular, float shininess = 1.f, float reflective = 0.f);
    CUDA_CALLABLE_MEMBER constexpr Material(const Material &other) = default;
    CUDA_CALLABLE_MEMBER constexpr Color getAmbient(void) const;
    CUDA_CALLABLE_MEMBER constexpr Color getDiffuse(void) const;
    CUDA_CALLABLE_MEMBER constexpr Color getSpecular(void) const;
    CUDA_CALLABLE_MEMBER constexpr float getShininess(void) const;
    CUDA_CALLABLE_MEMBER constexpr float getReflective(void) const;
private:
    Color m_ambient;
    Color m_diffuse;
    Color m_specular;
    float m_shininess;
    float m_reflective;
}; // End of class


constexpr Material::Material(const Color &ambient, const Color &diffuse, const Color &specular, float shininess, float reflective)
    : m_ambient(ambient)
    , m_diffuse(diffuse)
    , m_specular(specular)
    , m_shininess(shininess)
    , m_reflective(reflective)
{}

constexpr Color Material::getAmbient(void) const
{
    return m_ambient;
}

constexpr Color Material::getDiffuse(void) const
{
    return m_diffuse;
}

constexpr Color Material::getSpecular(void) const
{
    return m_specular;
}

constexpr float Material::getShininess(void) const
{
    return m_shininess;
}

constexpr float Material::getReflective(void) const
{
    return m_reflective;
}

#endif // MATERIAL_H
