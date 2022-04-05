#pragma once

#include <cuda_runtime.h>

#define CUDA_CHECK(expr) checkImplCuda((expr), __FILE__, __LINE__)
#define CUDA_CHECK_LAST_KERNEL() checkLastKernelCuda(__FILE__, __LINE__);

void checkImplCuda(cudaError_t result, const char* file, int line);

/**
 * @remarks
 * Définir la macro SYNCHRONIZE_CUDA_DEVICE_ON_ERROR_CHECK pour 
 * synchroniser le device après chaque kernel. Sinon l'indication
 * du fichier / ligne de l'erreur sera faux.
 */
void checkLastKernelCuda(const char* file, int line);