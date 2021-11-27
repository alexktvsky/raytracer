#include "cuda_exception.h"


#if defined(HAVE_CUDA)

CudaException::CudaException(const std::string &file, int line, cudaError_t code)
    : Exception(file, line, cudaGetErrorString(code))
    , m_code(code)
{}

#endif
