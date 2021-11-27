#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#if defined(__CUDACC__)
#define CUDA_CALLABLE_MEMBER  __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#if defined(HAVE_CUDA)

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "cuda_exception.h"

#define checkCudaError(expr)                                                  \
{                                                                             \
    cudaError_t err = expr;                                                   \
    if (err != cudaSuccess) {                                                 \
        throw CudaExceptionFromHere(err);                                     \
    }                                                                         \
}

template <typename T>
auto allocDeviceMemory(size_t n = 1)
{
    T *ptr;
    checkCudaError(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * n));
    return ptr;
}

template <typename T>
auto copyObjectToDevice(const T &obj)
{
    T *ptr;
    checkCudaError(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T)));
    checkCudaError(cudaMemcpy(ptr, &obj, sizeof(T), cudaMemcpyHostToDevice));
    return ptr;
}

#if defined(__CUDACC__)
template <typename T>
__global__ void createObjectOnDeviceKernel(T *ptr)
{
    T obj_copy(*ptr);
    memcpy(ptr, &obj_copy, sizeof(T));
}

template <typename T>
auto createObjectOnDevice(const T &obj)
{
    T *ptr;
    checkCudaError(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T)));
    checkCudaError(cudaMemcpy(ptr, &obj, sizeof(T), cudaMemcpyHostToDevice));
    createObjectOnDeviceKernel<<<1, 1>>>(ptr);
    checkCudaError(cudaDeviceSynchronize());
    return ptr;
}
#endif // __CUDACC__

#endif // HAVE_CUDA

#endif // CUDA_HELPER_H
