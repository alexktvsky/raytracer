#ifndef RENDER_KERNEL_H
#define RENDER_KERNEL_H

#if defined(HAVE_CUDA)

#include <vector>

std::vector<Color> renderKernel(size_t width, size_t height, const Camera &camera, const Scene &scene);

#endif // HAVE_CUDA

#endif // RENDER_KERNEL_H
