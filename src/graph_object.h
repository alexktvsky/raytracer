#ifndef GRAPH_OBJECT_H
#define GRAPH_OBJECT_H

#include <glm/glm.hpp>

#include "ray.h"
#include "intersect.h"
#include "cuda_helper.h"

class GraphObject {
public:
    CUDA_CALLABLE_MEMBER GraphObject(void) = default;
    CUDA_CALLABLE_MEMBER GraphObject(const GraphObject &other) = default;
    CUDA_CALLABLE_MEMBER GraphObject &operator=(const GraphObject &other) = default;
    CUDA_CALLABLE_MEMBER virtual Intersect intersect(const Ray &ray) const = 0;
    using Type = uint8_t;
    CUDA_CALLABLE_MEMBER virtual Type getType(void) const = 0;
    virtual ~GraphObject(void) = default;
protected:

private:

}; // End of class

#endif // GRAPH_OBJECT_H
