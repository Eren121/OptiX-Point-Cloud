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

    {
        // Vérifie les erreurs qui arrivent PENDANT le kernel
        // Sans ça, les erreurs seront détectées au prochain CUDA_CHECK()
        // ce qui n'est pas la bonne ligne d'erreur

        // Evidémment, comme on synchronise,
        // Cela peut ralentir, et empêcher la parallélisation pour notamment les streams,
        // à n'utiliser que pour debugger
        checkImplCuda(cudaDeviceSynchronize(), file, line);
    }
}

#define CUDA_CHECK(expr) checkImplCuda((expr), __FILE__, __LINE__)
#define CUDA_CHECK_LAST_KERNEL() checkLastKernelCuda(__FILE__, __LINE__);


///////////////////////////////////////////////////////////////////////////////////

/**
 * @remarks
 * Définir la macro SYNCHRONIZE_CUDA_DEVICE_ON_ERROR_CHECK pour 
 * synchroniser le device après chaque kernel. Sinon l'indication
 * du fichier / ligne de l'erreur sera faux.
 */
void checkLastKernelCuda(const char* file, int line);

#define DEF_FUNC(X) \
    __device__ void X() { printf(#X " kernel\n"); } \
    __device__ void* X##_addr = &X;

DEF_FUNC(hi)
DEF_FUNC(ha)

struct Params
{
    void(*f)();
};

__global__ void myKernel(Params params)
{
    params.f();
}

void*& find()
{
    return hi_addr;
}

int main()
{
    void(*f)();
    Params params;

    CUDA_CHECK(cudaMemcpyFromSymbol(&f, find(), sizeof(void*)));

    printf("addr. = %p\n", (void*)f);
    params.f = f;

    myKernel<<<1, 1>>>(params);

    CUDA_CHECK_LAST_KERNEL();

    printf("end\n");
}