#ifndef CUDA_EXCEPTION_H
#define CUDA_EXCEPTION_H

#if defined(HAVE_CUDA)

#include <cuda_runtime.h>

#include "exception.h"

#define CudaExceptionFromHere(code)  CudaException(__FILE__, __LINE__, code)

class CudaException : public Exception {
public:
    CudaException(const std::string &file, int line, cudaError_t code);
    cudaError_t getCode(void) const;
private:
    cudaError_t m_code;
}; // End of class


inline cudaError_t CudaException::getCode(void) const
{
    return m_code;
}

#endif

#endif // CUDA_EXCEPTION_H
