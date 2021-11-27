#include "scene.h"

#if !defined(HAVE_CUDA)


void Scene::addObject(const std::shared_ptr<GraphObject> &object)
{
    m_objects.push_back(object);
}

void Scene::addLightSource(const std::shared_ptr<LightSource> &light)
{
    if (m_total_intensity + light->getIntensity() > 1.f) {
        float diff = 1.f - m_total_intensity;
        light->setIntensity(diff);
        
        m_total_intensity = 1.f;
    }
    else {
        m_total_intensity += light->getIntensity();
    }

    m_lights.push_back(light);
}

#endif
