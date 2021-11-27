#include "scene.h"
#include "exception.h"


void Scene::addObject(const std::shared_ptr<GraphObject> &object)
{
    GraphObject::Type object_type = object->getType();

    std::shared_ptr<GraphObject> dev_ptr;

    if (object_type == Sphere::TYPE_ID) {
        dev_ptr = std::shared_ptr<GraphObject>(createObjectOnDevice(*std::reinterpret_pointer_cast<Sphere>(object)), &cudaFree);
    }
    else if (object_type == Plane::TYPE_ID) {
        dev_ptr = std::shared_ptr<GraphObject>(createObjectOnDevice(*std::reinterpret_pointer_cast<Plane>(object)), &cudaFree);
    }
    else if (object_type == Triangle::TYPE_ID) {
        dev_ptr = std::shared_ptr<GraphObject>(createObjectOnDevice(*std::reinterpret_pointer_cast<Triangle>(object)), &cudaFree);
    }
    else {
        throw ExceptionFromHere("Unknown graphic object.");
    }

    m_objects_cuda_ready = false;

    m_objects.push_back(object);
    m_objects_cuda.push_back(dev_ptr);
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

    LightSource::Type light_type = light->getType();

    std::shared_ptr<LightSource> dev_ptr;

    if (light_type == AmbientLight::TYPE_ID) {
        dev_ptr = std::shared_ptr<LightSource>(createObjectOnDevice(*std::reinterpret_pointer_cast<AmbientLight>(light)), &cudaFree);
    }
    else if (light_type == DirectionalLight::TYPE_ID) {
        dev_ptr = std::shared_ptr<LightSource>(createObjectOnDevice(*std::reinterpret_pointer_cast<DirectionalLight>(light)), &cudaFree);
    }
    else if (light_type == PointLight::TYPE_ID) {
        dev_ptr = std::shared_ptr<LightSource>(createObjectOnDevice(*std::reinterpret_pointer_cast<PointLight>(light)), &cudaFree);
    }
    else {
        throw ExceptionFromHere("Unknown light source.");
    }

    m_lights_cuda_ready = false;

    m_lights.push_back(light);
    m_lights_cuda.push_back(dev_ptr);
}

const std::pair<GraphObject **, size_t> Scene::getObjectsCuda(void) const
{
    if (!m_objects_cuda_ready) {

        m_objects_cuda_ready = true;

        size_t n_objects = m_objects_cuda.size();

        m_objects_cuda_array.reset(allocDeviceMemory<GraphObject*>(n_objects), &cudaFree);

        for (size_t i = 0; i < n_objects; ++i) {
            GraphObject *temp = m_objects_cuda[i].get();
            void *src_ptr = reinterpret_cast<void*>(&temp);
            void *dst_ptr = reinterpret_cast<void*>(m_objects_cuda_array.get() + i);
            cudaMemcpy(dst_ptr, src_ptr, sizeof(GraphObject *), cudaMemcpyHostToDevice);
        }
    }

    return std::pair<GraphObject **, size_t>(m_objects_cuda_array.get(), m_objects_cuda.size());
}

const std::pair<LightSource **, size_t> Scene::getLightSourcesCuda(void) const
{
    if (!m_lights_cuda_ready) {

        m_lights_cuda_ready = true;

        size_t n_lights = m_lights_cuda.size();

        m_lights_cuda_array.reset(allocDeviceMemory<LightSource*>(n_lights), &cudaFree);

        for (size_t i = 0; i < n_lights; ++i) {
            LightSource *temp = m_lights_cuda[i].get();
            void *src_ptr = reinterpret_cast<void*>(&temp);
            void *dst_ptr = reinterpret_cast<void*>(m_lights_cuda_array.get() + i);
            cudaMemcpy(dst_ptr, src_ptr, sizeof(LightSource *), cudaMemcpyHostToDevice);
        }
    }

    return std::pair<LightSource **, size_t>(m_lights_cuda_array.get(), m_lights_cuda.size());;
}
