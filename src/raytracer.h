#ifndef RAYTRACER_H
#define RAYTRACER_H

#include <vector>

#include "ray.h"
#include "color.h"
#include "scene.h"
#include "camera.h"

class Raytracer {
public:
    Raytracer(size_t width, size_t height, const Scene &scene, const Camera &camera);
    std::vector<Color> render(void);
    std::vector<Color> renderWithCpu(void);
#if defined (HAVE_CUDA)
    std::vector<Color> renderWithCuda(void);
#endif
private:
    Color traceRay(const Ray &ray, size_t depth);
    Intersect closestIntersection(const Ray &ray);
    glm::vec3 reflectedDirection(const glm::vec3 &r, const glm::vec3 &n);
    Color processLight(const Intersect &intersect);
private:
    size_t m_width;
    size_t m_height;
    Scene m_scene;
    Camera m_camera;
}; // End of class

#endif // RAYTRACER_H
