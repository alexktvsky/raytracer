#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <memory>
#include <utility>

#include "graph_object.h"
#include "sphere.h"
#include "plane.h"
#include "triangle.h"
#include "light_source.h"
#include "ambient_light.h"
#include "directional_light.h"
#include "point_light.h"
#include "cuda_helper.h"

class Scene {
public:
    Scene(void);
    void addObject(const std::shared_ptr<GraphObject> &object);
    void addLightSource(const std::shared_ptr<LightSource> &light);
    const std::vector<std::shared_ptr<GraphObject>> &getObjects(void) const;
    const std::vector<std::shared_ptr<LightSource>> &getLightSources(void) const;
#if defined(HAVE_CUDA)
    const std::pair<GraphObject **, size_t> getObjectsCuda(void) const;
    const std::pair<LightSource **, size_t> getLightSourcesCuda(void) const;
#endif
private:
    std::vector<std::shared_ptr<GraphObject>> m_objects;
    std::vector<std::shared_ptr<LightSource>> m_lights;
    float m_total_intensity;
#if defined(HAVE_CUDA)
    std::vector<std::shared_ptr<GraphObject>> m_objects_cuda;
    mutable std::shared_ptr<GraphObject*> m_objects_cuda_array;
    std::vector<std::shared_ptr<LightSource>> m_lights_cuda;
    mutable std::shared_ptr<LightSource*> m_lights_cuda_array;
    mutable bool m_objects_cuda_ready;
    mutable bool m_lights_cuda_ready;
#endif
}; // End of class


inline Scene::Scene(void)
    : m_total_intensity(0.f)
#if defined(HAVE_CUDA)
    , m_objects_cuda_ready(false)
    , m_lights_cuda_ready(false)
#endif
{}

inline const std::vector<std::shared_ptr<GraphObject>> &Scene::getObjects(void) const
{
    return m_objects;
}

inline const std::vector<std::shared_ptr<LightSource>> &Scene::getLightSources(void) const
{
    return m_lights;
}

#endif // SCENE_H
