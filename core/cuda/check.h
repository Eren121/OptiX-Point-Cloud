#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK(expr) checkImplCuda((expr), __FILE__, __LINE__)

void checkImplCuda(cudaError_t result, const char* file, int line);