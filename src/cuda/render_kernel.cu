#include <memory>
#include <algorithm>

#include "color.h"
#include "ray.h"
#include "intersect.h"
#include "camera.h"
#include "scene.h"
#include "sphere.h"
#include "plane.h"
#include "triangle.h"
#include "light_source.h"
#include "ambient_light.h"
#include "directional_light.h"
#include "point_light.h"
#include "cuda_helper.h"
#include "render_kernel.h"


__device__ Intersect closestIntersection(const Ray &ray, GraphObject **objects, size_t n_objects)
{
    Intersect closest_intersect;

    for (size_t i = 0; i < n_objects; ++i) {

        Intersect intersect = objects[i]->intersect(ray);

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

__device__ glm::vec3 reflectedDirection(const glm::vec3 &r, const glm::vec3 &n)
{
    return 2.f * n * glm::dot(n, r) - r;
}

__device__ Color processLight(const Intersect &intersect, GraphObject **objects, size_t n_objects, LightSource **lights, size_t n_lights)
{
    Color result_color;

    for (size_t i = 0; i < n_lights; ++i) {

        glm::vec3 shadow_orig, l;

        LightSource::Type light_type = lights[i]->getType();

        if (light_type == DirectionalLight::TYPE_ID) {
            l = reinterpret_cast<DirectionalLight*>(lights[i])->getDirection();
        }
        else if (light_type == PointLight::TYPE_ID) {
            l = reinterpret_cast<PointLight*>(lights[i])->getPosition() - intersect.getPoint();
        }
        else {
            result_color = result_color + lights[i]->calculateColor(intersect);
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

        if (!closestIntersection(Ray(shadow_orig, l), objects, n_objects).isHit()) {
            result_color = result_color + lights[i]->calculateColor(intersect);
        }
    }

    return result_color;
}

__device__ Color traceRay(Color *buf, const Ray &ray, GraphObject **objects, size_t n_objects, LightSource **lights, size_t n_lights, size_t reflection_depth)
{
    Intersect closest_intersect = closestIntersection(ray, objects, n_objects);

    if (closest_intersect.isHit()) {

        Color result_color = processLight(closest_intersect, objects, n_objects, lights, n_lights);

        if (closest_intersect.getMaterial().getReflective() > 0.f && reflection_depth > 0) {
            Color reflected_color = traceRay(buf, Ray(closest_intersect.getPoint(), reflectedDirection(-ray.getDirection(), closest_intersect.getNormal())), objects, n_objects, lights, n_lights, reflection_depth - 1);
            return result_color * (1.f - closest_intersect.getMaterial().getReflective()) + reflected_color * closest_intersect.getMaterial().getReflective();
        }

        return result_color;
    }

    return Color(0, 255, 255);
}

__global__ void render(Color *buf, Camera *camera, GraphObject **objects, size_t n_objects, LightSource **lights, size_t n_lights)
{
    size_t x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t offset = x + y * blockDim.x * gridDim.x;
    size_t width = blockDim.x * gridDim.x;
    size_t height = blockDim.y * gridDim.y;
    const size_t reflection_depth = 4;

    buf[offset] = traceRay(buf, camera->getRay(x - width / 2, y - height / 2), objects, n_objects, lights, n_lights, reflection_depth);
}

std::vector<Color> renderKernel(size_t width, size_t height, const Camera &camera, const Scene &scene)
{
    std::vector<Color> frame(width * height);

    std::unique_ptr<Color, decltype(&cudaFree)> dev_frame(allocDeviceMemory<Color>(frame.size()), &cudaFree);
    std::unique_ptr<Camera, decltype(&cudaFree)> dev_camera(copyObjectToDevice(camera), &cudaFree);

    GraphObject **objects = std::get<GraphObject**>(scene.getObjectsCuda());
    size_t n_objects = std::get<size_t>(scene.getObjectsCuda());
    LightSource **lights = std::get<LightSource**>(scene.getLightSourcesCuda());
    size_t n_lights = std::get<size_t>(scene.getLightSourcesCuda());

    const size_t blocks_dim = 16;
    dim3 threads(blocks_dim, blocks_dim);
    dim3 blocks((width + blocks_dim - 1) / blocks_dim, (height + blocks_dim - 1) / blocks_dim);

#if defined(DEBUG)
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsed_time = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
#endif

    render<<<blocks, threads>>>(dev_frame.get(), dev_camera.get(), objects, n_objects, lights, n_lights);

    checkCudaError(cudaDeviceSynchronize());

#if defined(DEBUG)
    checkCudaError(cudaEventRecord(stop, 0));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsed_time, start, stop));

    printf("GPU elapsed time: %.2fms\n", elapsed_time);

    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));
#endif

    checkCudaError(cudaMemcpy(frame.data(), dev_frame.get(), frame.size() * sizeof(Color), cudaMemcpyDeviceToHost));

    return frame;
}
