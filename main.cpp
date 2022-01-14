#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h> // On est pas dans un .cu sinon c'est inclus automatiquement
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h> // Nécessaire sinon erreur de link g_optixFunctionTable
#include "ray.cuh"

std::string readAllFile(const char *path)
{
    using iterator = std::istreambuf_iterator<char>;
    
    std::ifstream ifs(path);
    std::string content{(iterator(ifs)), iterator()};
    return content;
}

void CUDA_CHECK(cudaError_t result)
{
    if(result != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

void OPTIX_CHECK(OptixResult result)
{
    if(result != OPTIX_SUCCESS)
    {
        std::cerr << "OptiX error: " << optixGetErrorString(result) << std::endl;
        exit(1);
    }
}

void initCuda()
{
    // La doc montre que cela est une no-op qui initialise cuda
    CUDA_CHECK(cudaFree(nullptr));
}

void logCallback(unsigned int level, const char *tag, const char *message, void *cbdata)
{
    // Callback appelé pour les logs OptiX
    std::cerr << "[" << level << "][" << tag << "]" << " " << message << std::endl;
}

#ifndef MY_PTX_PATH
    #error Define MY_PTX_PATH
#endif

OptixModule createModule(OptixDeviceContext context)
{
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0; // OPTIX_COMPILE_OPTIMIZATION_LEVEL_DEFAULT en release
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL; // OPTIX_COMPILE_DEBUG_LEVEL_FULL_DEFAULT en release

    const std::string ptxContent = readAllFile(MY_PTX_PATH);

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags = 0;

    OptixModule module = nullptr;
    
    {
        size_t logStringSize = 2000;
        std::vector<char> logString(logStringSize);
        
        const OptixResult result = optixModuleCreateFromPTX(
            context,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            ptxContent.c_str(), ptxContent.size(),
            logString.data(), &logStringSize,
            &module
        );

        if(result != OPTIX_SUCCESS)
        {
            std::cerr << "OptiX module compilation error: " << logString.data() << std::endl;
        }

        OPTIX_CHECK(result);
    }

    return module;
}

OptixProgramGroup createProgramGroup(OptixDeviceContext context, OptixModule module)
{
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.entryFunctionName = "__raygen__my_program";
    pgDesc.raygen.module = module;

    OptixProgramGroup group = nullptr;

    {
        size_t logStringSize = 2000;
        std::vector<char> logString(logStringSize);
    
        OptixProgramGroupOptions pgOptions = {};
        const OptixResult result = optixProgramGroupCreate(
            context,
            &pgDesc,
            1,
            &pgOptions,
            logString.data(), &logStringSize,
            &group
        );

        if(result != OPTIX_SUCCESS)
        {
            std::cerr << "OptiX program group compilation error: " << logString.data() << std::endl;
        }

        OPTIX_CHECK(result);
    }

    return group;
}

OptixPipeline createPipeline(OptixDeviceContext context, OptixProgramGroup programGroup)
{
    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    // Doit être le même que lors de la création du module
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    pipelineCompileOptions.usesPrimitiveTypeFlags = 0;


    {
        size_t logStringSize = 1000;
        std::vector<char> logString(logStringSize);
    
        OptixProgramGroupOptions pgOptions = {};
        const OptixResult result = optixPipelineCreate(
            context,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            &programGroup,
            1,
            logString.data(), &logStringSize,
            &pipeline
        );

        if(result != OPTIX_SUCCESS)
        {
            std::cerr << "OptiX program group compilation error: " << logString.data() << std::endl;
        }

        OPTIX_CHECK(result);
    }

    return pipeline;
}

OptixDeviceContext createContext()
{
    // 0 signifie le contexte courant 
    // Les contextes cuda ne sont pas les mêmes que les contextes OptiX
    CUcontext cuContext = 0;

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &logCallback;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL; // Vérification des erreurs. Peut ralentir, à utiliser seulement pour le debugging
    OptixDeviceContext context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &context));

    return context;
}

int main(int argc, char **argv)
{
    initCuda();
    OPTIX_CHECK(optixInit());
    
    OptixDeviceContext context = createContext();

    OptixModule module = createModule(context);
    OptixProgramGroup group = createProgramGroup(context, module);
    OptixPipeline pipeline = createPipeline(context, group);

    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(group));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));

    CUstream stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUdeviceptr raygenRecord;
    void *d_pipelineParams = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pipelineParams, sizeof(Params)));
    

    unsigned int width = 100;
    unsigned int height = 100;
    unsigned int depth = 1;
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = raygenRecord;

    CUDA_CHECK(cudaFree(d_pipelineParams));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return EXIT_SUCCESS;
}