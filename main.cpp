#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h> // On est pas dans un .cu sinon c'est inclus automatiquement
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h> // Nécessaire sinon erreur de link g_optixFunctionTable
#include "ray.cuh"
#include "Record.hpp"

std::string readAllFile(const char *path)
{
    using iterator = std::istreambuf_iterator<char>;
    
    std::ifstream ifs(path);
    std::string content{(iterator(ifs)), iterator()};
    return content;
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
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG; // _NONE en release
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

std::vector<OptixProgramGroup> createProgramGroup(
    OptixDeviceContext context,
    OptixModule module)
{
    const size_t nb_programs = 3;
    OptixProgramGroupDesc pgDesc[nb_programs] = {};
    pgDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc[0].raygen.entryFunctionName = "__raygen__my_program";
    pgDesc[0].raygen.module = module;
    
    pgDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

    std::vector<OptixProgramGroup> groups(nb_programs, nullptr);

    {
        size_t logStringSize = 2000;
        std::vector<char> logString(logStringSize);
    
        OptixProgramGroupOptions pgOptions = {};
        const OptixResult result = optixProgramGroupCreate(
            context,
            &pgDesc[0],
            nb_programs,
            &pgOptions,
            logString.data(), &logStringSize,
            groups.data()
        );

        if(result != OPTIX_SUCCESS)
        {
            std::cerr << "OptiX program group compilation error: " << logString.data() << std::endl;
        }

        OPTIX_CHECK(result);
    }

    return groups;
}

OptixPipeline createPipeline(OptixDeviceContext context,
    const std::vector<OptixProgramGroup> &programGroup)
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
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG;
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
            programGroup.data(),
            static_cast<unsigned int>(programGroup.size()),
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

void printResult(const Params &params)
{
    printf("hello?\n");

    std::vector<uchar4> image(params.width * params.height);

    CUDA_CHECK(cudaMemcpy(image.data(),
        params.image, image.size() * sizeof(image[0]),
        cudaMemcpyDeviceToHost)
    );

    uchar4 px = image[0];
    uint4 u = make_uint4(px.x, px.y, px.z, px.w);
    printf("%d, %d, %d, %d\n", u.x, u.y, u.z, u.w);
}

int main(int argc, char **argv)
{
    initCuda();
    OPTIX_CHECK(optixInit());
    
    OptixDeviceContext context = createContext();

    OptixModule module = createModule(context);
    std::vector<OptixProgramGroup> groups = createProgramGroup(context,
        module);

    OptixProgramGroup raygenGroup = groups[0];

    OptixPipeline pipeline = createPipeline(context, groups);

    CUstream stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    Params params = {};
    params.width = 100;
    params.height = 100;
    
    CUDA_CHECK(cudaMalloc(
        &params.image,
        sizeof(*params.image) * params.width * params.height
    ));

    const unsigned int depth = 1;

    // Copy params sur le GPU
    CUdeviceptr d_params = 0;
    CUDA_CHECK(cudaMalloc(
        &reinterpret_cast<void*>(d_params),
        sizeof(Params))
    );

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*> (d_params),
        &params,
        sizeof(Params),
        cudaMemcpyHostToDevice)
    );

    Record<void*> raygenRecord(raygenGroup);
    raygenRecord.copyToDevice();

    Record<void*> missRecord(raygenGroup);
    missRecord.copyToDevice();
    
    // définir au minimum le raygenRecord et le missRecord
    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = raygenRecord.getDevicePtr();
    sbt.missRecordBase = missRecord.getDevicePtr();
    sbt.missRecordStrideInBytes = sizeof(missRecord);
    sbt.missRecordCount = 1;

    OPTIX_CHECK(optixLaunch(
        pipeline,
        stream,
        d_params,
        sizeof(Params),
        &sbt,
        params.width, params.height, depth
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));

    for(OptixProgramGroup group : groups) {
        OPTIX_CHECK(optixProgramGroupDestroy(group));
    }
    printResult(params);

    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(params.image));

    return EXIT_SUCCESS;
}