#include "raytracer.h"
#include "intersect.h"
#include "sphere.h"
#include "plane.h"
#include "triangle.h"
#include "light_source.h"
#include "ambient_light.h"
#include "directional_light.h"
#include "point_light.h"

#if defined(HAVE_CUDA)
#include "render_kernel.h"
#endif


Raytracer::Raytracer(size_t width, size_t height, const Scene &scene, const Camera &camera)
    : m_width(width)
    , m_height(height)
    , m_scene(scene)
    , m_camera(camera)
{}

Intersect Raytracer::closestIntersection(const Ray &ray)
{
    Intersect closest_intersect;

    for (const auto &object : m_scene.getObjects()) {

        Intersect intersect = object->intersect(ray);

        if (intersect.isHit()) {
            if (!closest_intersect.isHit()) {
                closest_intersect = intersect; // Save first hit
            }
            else if (closest_intersect.getDistance() > intersect.getDistance()) {
                closest_intersect = intersect;
            }
        }
    }

    return closest_intersect;
}

glm::vec3 Raytracer::reflectedDirection(const glm::vec3 &r, const glm::vec3 &n)
{
    return 2.f * n * glm::dot(n, r) - r;
}

Color Raytracer::processLight(const Intersect &intersect)
{
    Color result_color;

    for (const auto &light : m_scene.getLightSources()) {

        glm::vec3 shadow_orig, l;

        LightSource::Type light_type = light->getType();

        if (light_type == DirectionalLight::TYPE_ID) {
            l = std::reinterpret_pointer_cast<DirectionalLight>(light)->getDirection();
        }
        else if (light_type == PointLight::TYPE_ID) {
            l = std::reinterpret_pointer_cast<PointLight>(light)->getPosition() - intersect.getPoint();
        }
        else {
            result_color = result_color + light->calculateColor(intersect);
            continue;
        }

        // Prevent situation when a point discards the shadow to itself.
        // So we shift the point in the direction of the normal.
        if (glm::dot(l, intersect.getNormal()) < 0.f) {
            shadow_orig = intersect.getPoint() - intersect.getNormal() * 1e-3f;
        }
        else {
            shadow_orig = intersect.getPoint() + intersect.getNormal() * 1e-3f;
        }

        if (!closestIntersection(Ray(shadow_orig, l)).isHit()) {
            result_color = result_color + light->calculateColor(intersect);
        }
    }

    return result_color;
}

Color Raytracer::traceRay(const Ray &ray, size_t reflection_depth)
{
    Intersect closest_intersect = closestIntersection(ray);

    if (closest_intersect.isHit()) {

        Color result_color = processLight(closest_intersect);

        if (closest_intersect.getMaterial().getReflective() > 0.f && reflection_depth > 0) {
            Color reflected_color = traceRay(Ray(closest_intersect.getPoint(), reflectedDirection(-ray.getDirection(), closest_intersect.getNormal())), reflection_depth - 1);
            return result_color * (1.f - closest_intersect.getMaterial().getReflective()) + reflected_color * closest_intersect.getMaterial().getReflective();
        }

        return result_color;
    }

    return Color(0, 255, 255);
}

std::vector<Color> Raytracer::render(void)
{
#if defined (HAVE_CUDA)
    return renderWithCuda();
#else
    return renderWithCpu();
#endif
}

std::vector<Color> Raytracer::renderWithCpu(void)
{
    std::vector<Color> frame(m_width * m_height);

    auto setPixel = [&frame, width = m_width, height = m_height] (size_t x, size_t y, const Color &color) {
        size_t s_x = width / 2 + x;
        size_t s_y = height / 2 + y;
        frame[s_y * width + s_x] = color;
    };

    const size_t reflection_depth = 4;

    for (ptrdiff_t i = -static_cast<ptrdiff_t>(m_width / 2); i < static_cast<ptrdiff_t>(m_width / 2); ++i) {
        for (ptrdiff_t j = -static_cast<ptrdiff_t>(m_height / 2); j < static_cast<ptrdiff_t>(m_height / 2); ++j) {
            setPixel(i, j, traceRay(m_camera.getRay(i, j), reflection_depth));
        }
    }

    return frame;
}

#if defined (HAVE_CUDA)
std::vector<Color> Raytracer::renderWithCuda(void)
{
    return renderKernel(m_width, m_height, m_camera, m_scene);
}
#endif
