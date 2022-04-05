#include "check.h"
#include "core/utility/debug.h"
#include <cstdio>
#include <cstdlib>

void checkImplCuda(cudaError_t result, const char* file, int line)
{
    if(result != cudaSuccess)
    {
        std::fprintf(stderr, "%s:%d\n", file, line);
        std::fprintf(stderr, "CUDA error: %s", cudaGetErrorString(result));
        std::exit(1);
    }
}

void checkLastKernelCuda(const char* file, int line)
{
    // Vérifie les erreurs de LANCEMENT du kernel
    checkImplCuda(cudaPeekAtLastError(), file, line);

    #if DEBUG_ENABLED && defined(SYNCHRONIZE_CUDA_DEVICE_ON_ERROR_CHECK)
    {
        #warning Each kernel launch will be synchronized, it will affect performance on multiple streams
        
        // Vérifie les erreurs qui arrivent PENDANT le kernel
        // Sans ça, les erreurs seront détectées au prochain CUDA_CHECK()
        // ce qui n'est pas la bonne ligne d'erreur

        // Evidémment, comme on synchronise,
        // Cela peut ralentir, et empêcher la parallélisation pour notamment les streams,
        // à n'utiliser que pour debugger
        checkImplCuda(cudaDeviceSynchronize(), file, line);
    }
    #endif
}