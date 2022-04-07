#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h> // On est pas dans un .cu sinon c'est inclus automatiquement
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "ray.cuh"
#include "Record.hpp"
#include "stb_image_write.h" // Pour écrire des PNG
#include "Application.h"
#include <cuda_gl_interop.h> // Pour pouvoir utiliser les fonctionalités d'interopérabilité OpenGL de CUDA
#include "imgui.h"
#include "PointsCloud.hpp"
#include "SimpleGLRect.hpp"
#include "OrbitalControls.hpp"
#include "Scene.hpp"
#include "Gui.hpp"
#include "core/utility/debug.h"
#include "core/utility/time.h"
#include "core/cuda/StreamScatter.h"
#include "core/cuda/Pbo.h"
#include "core/gl/check.h"
#include "ssaa/SuperSampling.h"
#include "ssaa/patterns.h"

std::unique_ptr<PointsCloud> points;

const bool optixDebugOn = DEBUG_ENABLED;

// Fonction utilitaire pour lire tout un fichier dans une string
std::string readAllFile(const char *path)
{
    using iterator = std::istreambuf_iterator<char>;
    
    std::ifstream ifs(path);
    std::string content{(iterator(ifs)), iterator()};
    return content;
}

// "Initialise" CUDA: pas vraiment nécessaire, mais vérifie une éventuelle erreur d'initialisation.
void initCuda()
{
    // cudaFree(nullptr) fonction n'effectue rien,
    // Mais elle initialise indirectement CUDA lors du premier appel.
    // La documentation explique que c'est la façon habituelle de vérifier que CUDA fonctionne.
    CUDA_CHECK(cudaFree(nullptr));
}

// Fonction de callback appelée lors d'une erreur OptiX.
// Certaines fonctions comme optixModuleCreateFromPTX() prennent un argument supplémentaire pour stocker l'éventuel erreur comme avec OpenGL,
// logCallback() permet d'éviter à spécifier ces arguments pour récupérer le message d'erreur.
// qui indique une string où stocker le message d'erreur plus précis (à voir ?), qu'il faut donc aussi utiliser en plus du log de callback.
// cbdata est une donnée utilisateur définie dans la variable logCallbackData lors de l'initialisation du contexte OptiX.
void logCallback(unsigned int level, const char *tag, const char *message, void *cbdata)
{
    // Callback appelé pour les logs OptiX
    std::cerr << "[" << level << "][" << tag << "]" << " " << message << std::endl;
}

// MY_PTX_PATH est défini dans le CMakeLists.txt
// Il s'agit du chemin qui contient le .cu compilé en .ptx, sorte de bytecode assembleur GPU intermédiaire
// (lisible par un humain, on peut même l'écrire soi-même mais c'est évidemment très bas niveau.)
// Le contenu des .ptx est un peu l'équivalent du code source fragment shader / vertex shader en OpenGL.
#ifndef MY_PTX_PATH
    #error J'ai besoin que MY_PTX_PATH soit défini
#endif

// Des options à donner lors de la création des modules et de la pipeline.
// On doit donner la même valeur pour tous les mêmes modules d'une pipeline et pour la pipeline,
// Donc autant factoriser le code dans une fonction.
const OptixPipelineCompileOptions* getPipelineCompileOptions()
{
    // Initialise tous les membres de la structure
    // On se permet d'utiliser une variable static car les fonctions OptiX attendent un pointeur constant,
    // Pas besoin de copier la variable à chaque fois et économise quelques lignes à créer la variable locale dans chaque fonction.
    static OptixPipelineCompileOptions pipelineCompileOptions = {};

    // On réécrit à chaque appel de getPipelineCompileOptions() les valeurs mais ce n'est pas très grave car ce sont des valeurs constantes
    // (en C++20 on pourra initialiser pipelineCompileOptions avec un "= { .useMotionBlur = false, ... }" comme en C)
    pipelineCompileOptions.usesMotionBlur = false;
   
    // Nombre de int32 possibles qu'on pourra utiliser pour le passage
    // de paramètres
    pipelineCompileOptions.numPayloadValues = 3;

    // Pour les intersections
    // Pour notre programme d'intersection de sphères,
    // on utilise 4 attributs pour la position (x, y, z) de collision
    // et l'index de la sphère
    pipelineCompileOptions.numAttributeValues = 3;

    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    // Si notre structure d'accélération ne possède que des triangles
    // ou des structures custom
    // on peut mettre un flag pour optimiser
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    // Pour debugger
    // _NONE en release
    if(optixDebugOn)
    {
        pipelineCompileOptions.exceptionFlags =
            OPTIX_EXCEPTION_FLAG_DEBUG |
            OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
            OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
            OPTIX_EXCEPTION_FLAG_USER;
    }
    else
    {
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    }

    // Comme cet exemple n'a qu'une seule structure d'accélération,
    // cette option est nécessaire sinon erreur OPTIX_EXCEPTION_CODE_UNSUPPORTED_SINGLE_LEVEL_GAS

    // CHANGEMENT: on utilise maintenant une IAS pour mixer custom primitives / curves
    // Donc on utilise OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
    // Cela marcherait aussi avec OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY
    // mais ça serait moins optimisé
    pipelineCompileOptions.traversableGraphFlags =
        //OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

    return &pipelineCompileOptions;
}

// Création du module OptiX sur un contexte.
OptixModule createModule(OptixDeviceContext context)
{
    // https://raytracing-docs.nvidia.com/optix7/api/struct_optix_module_compile_options.html
    // Options de compilation du module
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;

    // Un peu comme les options de gcc -O1, -O2, -O3.
    // En Debug il vaut mieux mettre à 0, et OPTIX_COMPILE_OPTIMIZATION_LEVEL_DEFAULT en Release.
    if(optixDebugOn)
    {
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    }
    else
    {
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    }

    // Informations de debugging. Un peu comme l'option de gcc -g.
    // En Debug il vaut mieux mettre à OPTIX_COMPILE_DEBUG_LEVEL_FULL, et en Release à OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT.
    if(optixDebugOn)
    {
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    }
    else
    {
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    }

    // Lit le programme assembleur du fichier
    const std::string ptxContent = readAllFile(MY_PTX_PATH);

    OptixModule module = nullptr;
    OPTIX_CHECK(optixModuleCreateFromPTX(
        context,
        &moduleCompileOptions,
        getPipelineCompileOptions(),
        ptxContent.c_str(), ptxContent.size(),
        nullptr, nullptr, // Log string / log size: pas besoin, on utilise logCallback()
        &module
    ));

    return module;
}

std::vector<OptixProgramGroup> createProgramGroup(
    OptixDeviceContext context,
    OptixModule module)
{
    const size_t nb_programs = 4;
    OptixProgramGroupDesc pgDesc[nb_programs] = {};
    pgDesc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc[0].raygen.entryFunctionName = "__raygen__my_program";
    pgDesc[0].raygen.module = module;
    
    // Le programme d'intersection n'est pas requis, en cas spécial, pour les primitives triangles
    // Comme ici on n'utilise que des triangles on n'a donc pas besoin de programm d'intersection. 
    pgDesc[1].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc[1].hitgroup.moduleCH = module;
    pgDesc[1].hitgroup.entryFunctionNameCH = "__closesthit__my_program";
    pgDesc[1].hitgroup.moduleAH = module;
    pgDesc[1].hitgroup.entryFunctionNameAH = "__anyhit__my_program";
    pgDesc[1].hitgroup.moduleIS = module;
    pgDesc[1].hitgroup.entryFunctionNameIS = "__intersection__my_program";
    
    pgDesc[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc[2].miss.entryFunctionName = "__miss__my_program";
    pgDesc[2].miss.module = module;

    pgDesc[3].kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
    pgDesc[3].exception.entryFunctionName = "__exception__my_program";
    pgDesc[3].exception.module = module;

    std::vector<OptixProgramGroup> groups(nb_programs, nullptr);

    OptixProgramGroupOptions pgOptions = {};
    
    OPTIX_CHECK(optixProgramGroupCreate(
        context,
        &pgDesc[0],
        nb_programs,
        &pgOptions,
        nullptr, nullptr, // Log string / log size: pas besoin, on utilise logCallback()
        groups.data()
    ));

    return groups;
}

OptixPipeline createPipeline(OptixDeviceContext context,
    const std::vector<OptixProgramGroup> &programGroup)
{
    OptixPipeline pipeline = nullptr;
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 3;

    if(optixDebugOn)
    {
        pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    }
    else
    {
        pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    }

    OptixProgramGroupOptions pgOptions = {};
    OPTIX_CHECK(optixPipelineCreate(
        context,
        getPipelineCompileOptions(),
        &pipelineLinkOptions,
        programGroup.data(),
        static_cast<unsigned int>(programGroup.size()),
        nullptr, nullptr, // Log string / log size: pas besoin, on utilise logCallback()
        &pipeline
    ));

    return pipeline;
}

// Création du contexte OptiX, semblable à un context OpenGL.
// OptixDeviceContext est à considérer comme un pointeur (donc léger à copier).
// C'est ici que l'on peut définir certaines options, toutes relatives au debugging et toutes optionnelles.
OptixDeviceContext createContext()
{
    // Quel contexte CUDA OptiX doit utiliser ?
    // 0 signifie le contexte courant, cela évite de devoir gérer le CUcontext nous même.
    // Attention, les contextes cuda ne sont pas les mêmes que les contextes OptiX (CUcontext != OptixDeviceContext).
    CUcontext cuContext = 0;

    // Structure pour stocker les options de création du contexte OptiX.
    // OptiX recommande de toujours initialiser les structures avec "= {}" pour initialiser la structure à zéro,
    // puis ensuite écraser uniquement les variables qui nous intéressent avec nos propres valeurs.
    OptixDeviceContextOptions options = {};

    // Quelle fonction de callback OptiX doit utiliser en cas d'erreur ?
    options.logCallbackFunction = &logCallback;

    // Quel niveau d'erreur maximum pour appeler la fonction de callback customisée ?
    // Plus le niveau est bas, plus l'erreur est "grave".
    // Tous les messages de log en dessus du niveau indiqué seront ignorés.
    // Les différents niveaux possibles sont:
    //      0: Désactiver totalement la fonctionalité de log
    //      1: [Erreurs fatales]: Après une erreur fatale, le contexte OptiX n'est plus garanti d'être valide
    //      2: [Erreur non-fatales] (par exemple, un paramètre de fonction invalide)
    //      3: [Avertissements]: quand par exemple OptiX détecte que le programme ne se comporte pas exactement comme demandé
    //         ou plus lentement que prévu (avec par exemple une mauvaise combinaison d'options)
    //      4: Info
    if(optixDebugOn)
    {
        options.logCallbackLevel = 3; // Ici, on affiche tout pour bien debugger
    }
    else
    {
        options.logCallbackLevel = 0;
    }


    // Vérifications d'erreurs en bonus à l'exécution
    // Cela permet encore de débugger, mais cela a un coût et peut ralentir le programme.
    // En Debug on pourra mettre OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL qui permet d'activer toutes les options bonus.
    // En Release on pourra mettre OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF.
    if(optixDebugOn)
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    }
    else
    {
        options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
    }

    // Création du contexte et vérification des erreurs
    OptixDeviceContext context = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(cuContext, &options, &context));

    return context;
}

// Créer la structure accélératrice OptiX
// Cette partie est la plus importante, elle permet d'importer les géométries dans OptiX.
// C'est la seule partie qui dépend vraiment de l'application, le reste des fonctions (createPipeline(), createContext()...)
// sera a peu près pareil quelque soit l'application

template<typename It>
TraversableHandleStorage createAccelerationStructure(OptixDeviceContext context, CUstream stream, It begin, It end)
{
    // ====== Options de la structure accélératrice ======

    // Tableau de l'ensemble des géométries de la scène
    // On peut mettre plusieurs géométries mais toutes les géométries doivent avoir le même type (triangles, custom, curves, instances)
    std::vector<OptixBuildInput> buildInputs;

    // Options de la structure accélératrice
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    options.operation = OPTIX_BUILD_OPERATION_BUILD; // On veut construire la structure accélératrice, pas l'actualiser pour l'instant
    options.motionOptions.numKeys = 0; // 0 pour pas de motion blur, que l'on utilise pas ici

    // ====== Création d'un triangle ======

    // Coordonnées ABC du triangle
    static const float3 vertices[] = {
        make_float3(0.25f, 0.25f, -1.0f),
        make_float3(0.75f, 0.75f, -1.0f),
        make_float3(0.25f, 0.75f, -1.0f),

        
        make_float3(0.25f+0.25f, 0.25f-0.25f, -1.0f),
        make_float3(0.75f+0.25f, 0.75f-0.25f, -1.0f),
        make_float3(0.25f+0.25f, 0.75f-0.25f, -1.0f),
    };

    static const unsigned int indices[] = {0, 1, 2};

    // Création & passage des vertices sur le GPU
    // d_vertices doit être aligné sur 4-bytes (peut importe car cudaMalloc aligne toujours sur 256-bytes)
    CUdeviceptr d_vertices = 0;
    CUDA_CHECK(cudaMalloc(&reinterpret_cast<void*>(d_vertices), sizeof(vertices)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices, sizeof(vertices), cudaMemcpyHostToDevice));

    CUdeviceptr d_indices = 0;
    CUDA_CHECK(cudaMalloc(&reinterpret_cast<void*>(d_indices), sizeof(indices)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_indices), indices, sizeof(indices), cudaMemcpyHostToDevice));

    // Convertit le vecteur de points en vecteur de AABB
    std::vector<OptixAabb> aabbs;
    for(It it = begin; it != end; ++it)
    //for(const Point& point : points->getPoints())
    {
        const Point& point = *it;
        aabbs.push_back(point.toAabb());
    }

    // Envoie le vecteur de AABB sur le GPU
    managed_device_ptr d_aabbs(aabbs.data(), sizeof(*aabbs.data()) * aabbs.size());

    // Triangles
    if(false)
    {
        // On ajoute un élément (qui doit être initialisé à zéro)
        // On peut initialiser une structure automatiquement à zéro avec les accolades (par exemple "MyStruct s{}")
        buildInputs.push_back(OptixBuildInput{});
        OptixBuildInput& buildInput = buildInputs.back();

        // On veut que cet élément soit un triangle
        // ERREUR Dans la doc OptiX:
        //      https://raytracing-docs.nvidia.com/optix7/guide/index.html#acceleration_structures -> Listing 5.2 -> Ligne 3
        //      Ce ne doit pas être "buildInput.type" mais "buildInputs[0].type"
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // On choisit la partie triangleArray correspondante au type, OptixBuildInput se comporte comme une union
        // "triangles" au pluriel pour montrer que l'on peut stocker plusieurs triangles, mais ici l'on n'en créé qu'un seul.
        OptixBuildInputTriangleArray& triangles = buildInput.triangleArray;
        triangles.numVertices = sizeof(vertices) / sizeof(*vertices); // 1 seul triangle, donc 3 sommets
        triangles.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3; // On donne les sommets sous forme de float3

        // On peut donner plusieurs tableaux de vertices (ici, 1 seul)
        // donc on doit donner le pointeur vers le tableau de vertices
        triangles.vertexBuffers = &d_vertices;

        // Nombre de bytes entre chaque sommet (comme OpenGL),
        // Décalage par défaut avec 0 (tightly packed)
        triangles.vertexStrideInBytes = 0;

        // Indices
        // Optionnel, mais attention: dans ce cas on ne pas donner de valeur à indexFormat

        // Pointeur optionnel vers les indices (comme OpenGL)
        #if 0
        triangles.indexBuffer = d_indices;
        triangles.numIndexTriplets = sizeof(indices) / sizeof(*indices) / 3; // numIndexTriplets correspond au nombre de triangles
        triangles.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangles.indexStrideInBytes = 0;
        #endif

        // Pointeur optionnel vers une matrice de transformation (3x4 row-major), pas utilisé ici
        // On ne peut pas mettre "nullptr" car il s'agit d'un CUdeviceptr défini comme un entier, mais c'est un type opaque pour un pointeur
        triangles.preTransform = 0;

        // Chaque triangle se map à un ou plusieurs SBT Records
        // Si on a besoin que d'un seul SBT Record, tous les triangles référencent le même SBT Record.
        // Gérer plusieurs SBT Records est plus compliqué mais possible aussi
        triangles.numSbtRecords = 1;
        
        static const unsigned int flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
        triangles.flags = flags; // Un flag par SBT Record

    }
    else
    {
        buildInputs.push_back(OptixBuildInput{});
        OptixBuildInput& buildInput = buildInputs.back();
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        buildInput.customPrimitiveArray.aabbBuffers = &static_cast<const CUdeviceptr&>(d_aabbs);
        buildInput.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(aabbs.size()); // Taille du tableau aabbBuffers
        
        // Notre géométrie customisée a besoin que d'une seule donnée donc 1 SBT Record
        buildInput.customPrimitiveArray.numSbtRecords = 1;

        static const unsigned int flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
        buildInput.customPrimitiveArray.flags = flags; // Un flag par SBT Record

    }
    
    // ====== Allocation de la structure accélératrice sur mémoire GPU ======

    // OptiX n'alloue pas la mémoire lui-même, on doit allouer soi-même la mémoire
    // OptiX nous donne la taille minimale nécessaire avec optixAccelComputeMemoryUsage()
    // OptiX a besoin de 3 pointeurs de données:
    //      - Un pointeur pour stocker la structure accélératrice en elle-même
    //      - Un pointeur temporaire pour stocker les données nécessaires pour l'initialisation de la structure accélératrice
    //      - Un pointeur temporaire pour stocker les données nécessaires pour l'actualisation de la structure accélératrice
    //          (si par exemple une géometrie bouge dans la scène. On ne l'utilise pas ici)

    TraversableHandleStorage outputHandle; // Où stocker la structure accélératrice (outputHandle.d_data)
    CUdeviceptr d_temp = 0; // Où stocker les données temporaires pour l'initialisation de la structure accélératrice

    // bufferSizes nous donnera la taille de d_output et d_temp que l'on devra allouer
    OptixAccelBufferSizes bufferSizes;

    // Remplit bufferSizes
    // Cette fonction n'utilise jamais le GPU, les pointeurs sur le GPU de buildInputs comme vertexBuffers n'ont pas besoin d'être initialisés ici
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &options,
        buildInputs.data(),
        static_cast<unsigned int>(buildInputs.size()),
        &bufferSizes
    ));

    // Maintenant que l'on a la taille, on peut allouer les données
    // output et temp doivent être alignés sur 128-bytes (peu importe car cudaMalloc() aligne toujours sur 256-bytes)
    outputHandle.d_storage = managed_device_ptr(bufferSizes.outputSizeInBytes);
    CUDA_CHECK(cudaMalloc(&reinterpret_cast<void*>(d_temp), bufferSizes.tempSizeInBytes));

    std::cout << "accel_mem_size=" << bufferSizes.outputSizeInBytes << " bytes" << std::endl;
    std::cout << "accel_mem_size_temp=" << bufferSizes.tempSizeInBytes << " bytes" << std::endl;

    // ====== Construit l'AS ======

    // Toutes les données des buffers sont copiées vers une structure interne dont le handle est outputHandle, aucune référence n'est gardée
    //      (sauf pour les instances-AS mais ici ce n'est pas utilisé)
    // On peut donc libérer toute la mémoire allouée dans cette fonction après avec construit l'AS
    // SAUF d_output: il contient la structure de données, OptiX n'alloue aucune mémoire.
    // Donc pour détruire l'OptixTraversableHandle, il faudra désallouer d_output.
    // Pour tester, j'ai affiché la valeur de outputHandle.handle et d_output et j'obtiens:
    // outputHandle.handle=30115103232, d_output=30115103242
    // On voit bien que outputHandle référence une mémoire proche de d_output.
    OPTIX_CHECK(optixAccelBuild(
        context,
        stream,
        &options,
        buildInputs.data(), static_cast<unsigned int>(buildInputs.size()),
        d_temp, bufferSizes.tempSizeInBytes,
        outputHandle.d_storage, outputHandle.d_storage.size(),
        &outputHandle.handle,
        nullptr, 0 // Propriétés non utilisées ici
    ));

    // ====== Désallocation ======

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_indices)));
    
    return outputHandle;
}

bool isShiftDown(GLFWwindow* win)
{
    return glfwGetKey(win, GLFW_KEY_LEFT_SHIFT) || glfwGetKey(win, GLFW_KEY_RIGHT_SHIFT);
}

bool isCtrlDown(GLFWwindow* win)
{
    return glfwGetKey(win, GLFW_KEY_LEFT_CONTROL) || glfwGetKey(win, GLFW_KEY_RIGHT_CONTROL);
}

float userScalable(GLFWwindow* win, float f)
{
    if(isShiftDown(win))
    {
        f *= 10.0f;
    }
    if(isCtrlDown(win))
    {
        f /= 10.0f;
    }

    return f;
}

int main(int argc, char **argv)
{
    Gui gui;

    // ===== Création des streams

    const int numStreams = 1;
    cudaStream_t streams[numStreams];

    for(int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // ===== Chargement du nuage de points

    const char* path = R"(../../data/bunny/reconstruction/bun_zipper.ply)";
    points = std::make_unique<PointsCloud>(path);
    //points->randomizeColors();
    // ====== Initialisation ======

    initCuda();
    OPTIX_CHECK(optixInit());
    
    // ====== Création du contexte -> module -> groupe de programmes -> pipeline ======

    OptixDeviceContext context = createContext();

    OptixModule module = createModule(context);
    std::vector<OptixProgramGroup> groups = createProgramGroup(context,
        module);

    OptixProgramGroup raygenGroup = groups[0];
    OptixProgramGroup hitGroup = groups[1];
    OptixProgramGroup missGroup = groups[2];
    OptixProgramGroup exceptionGroup = groups[3];

    // On peut lancer des optixLaunch() en parallèle,
    // mais il faut que ce soit sur des pipelines différentes
    // Source: https://forums.developer.nvidia.com/t/optix7-0-could-i-use-two-streams-for-two-optixlaunch-operation-in-two-threads-for-speed-optimize/156764
    OptixPipeline pipelines[numStreams];
    for(int s = 0; s < numStreams; s++)
    {
        pipelines[s] = createPipeline(context, groups);
    }

    CUstream stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ====== Créer la structure accélératrice = envoyer les géométries sur le GPU ======

    auto beginPoints = begin(points->getPoints());
    auto endPoints = end(points->getPoints());
    const auto numPoints = points->getPoints().size();

    TraversableHandleStorage traversableHandleStorage = createAccelerationStructure(
        context, stream, beginPoints, endPoints);

    //TraversableHandleStorage traversableHandleStorage = createAccelerationStructure(
    //    context, stream, beginPoints, beginPoints + numPoints / 2);

    //TraversableHandleStorage traversableHandleStorage2 = createAccelerationStructure(
    //    context, stream, beginPoints + numPoints / 2, endPoints);

    //TraversableHandleStorage ias = mergeGASIntoIAS(context, stream, traversableHandleStorage.handle, traversableHandleStorage2.handle);

    /*
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK( optixUtilAccumulateStackSizes(raygenGroup, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes(hitGroup, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes(missGroup, &stack_sizes ) );
    OPTIX_CHECK( optixUtilAccumulateStackSizes(exceptionGroup, &stack_sizes ) );

    const uint32_t max_trace_depth = 2;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK( optixUtilComputeStackSizes( &stack_sizes, max_trace_depth,
                                             0,  // maxCCDepth
                                             0,  // maxDCDEpth
                                             &direct_callable_stack_size_from_traversal,
                                             &direct_callable_stack_size_from_state, &continuation_stack_size ) );
    OPTIX_CHECK( optixPipelineSetStackSize( pipeline, direct_callable_stack_size_from_traversal,
                                            direct_callable_stack_size_from_state, continuation_stack_size,
                                            2  // maxTraversableDepth
                                            ) );
    */

    // ====== Création des paramètres à envoyer au GPU pour le ray tracing ======

    managed_device_ptr d_pointsCloud(points->data(), sizeof(Point) * points->size());

    Record<CUdeviceptr> raygenRecord;
    raygenRecord.fill(raygenGroup);
    raygenRecord.data = d_pointsCloud;

    Record<void*> missRecord;
    missRecord.fill(missGroup);

    Record<CUdeviceptr> hitgroupRecord;
    // Alloue les informations de chaque points sur le GPU
    // Le raygen et hitgroup record on la même donnée, pour qu'ils puissent tous les deux accéder aux points
    hitgroupRecord.data = d_pointsCloud;
    hitgroupRecord.fill(hitGroup);

    Record<void*> exceptionRecord;
    exceptionRecord.fill(exceptionGroup);

    managed_device_ptr d_raygenRecord = managed_device_ptr::create_from(raygenRecord);
    managed_device_ptr d_missRecord = managed_device_ptr::create_from(missRecord);
    managed_device_ptr d_hitgroupRecord = managed_device_ptr::create_from(hitgroupRecord);
    managed_device_ptr d_exceptionRecord = managed_device_ptr::create_from(exceptionRecord);

    // définir au minimum le raygenRecord et le missRecord
    OptixShaderBindingTable sbt = {};

    sbt.raygenRecord = d_raygenRecord;

    sbt.missRecordBase = d_missRecord;
    sbt.missRecordStrideInBytes = sizeof(missRecord);
    sbt.missRecordCount = 1;

    sbt.hitgroupRecordBase = d_hitgroupRecord;
    sbt.hitgroupRecordStrideInBytes = sizeof(hitgroupRecord);
    sbt.hitgroupRecordCount = 1;

    sbt.exceptionRecord = d_exceptionRecord;

    
    {
        // Initialement, on créé la fenêtre à ratio 1:1
        Application app(maxWinTexWidth, maxWinTexHeight);

        // static pour permettre d'être utilisé dans les callbacks GLFW pour les inputs
        static Params params = {};

        params.camera.origin.z = 300;

        {
            const float ratio = 10.0f;
            params.width = maxWinTexWidth / ratio;
            params.height = maxWinTexHeight / ratio;
        }

        // Contient la texture qui sera affiché sur toute la fenêtre à chaque Frame, et le VAO et le VBO associés à la texture.        
        static std::unique_ptr<SimpleGLRect> rect
            = std::make_unique<SimpleGLRect>(maxWinTexWidth, maxWinTexHeight);
        rect->setRenderSize(params.width, params.height);
                    
        // Une exception est le PBO: on ré-alloue la taille à chaque fois que celle-ci change
        // Obligé avec OpenGL, mais cela devrait être relativement rapide
        // Heureusement, la taille du PBO ne dépend pas du nombre de pixels
        Pbo pbo(params.width, params.height);

            
        static OrbitalControls orbitalControls(make_float3(0.0f, 0.0f, 0.0f), 100.0f);

        // Définir le centre de la vue comme le centre estimé du model
        // On estime le centre en calculant la moyenne de la position des points
        // (peut ne pas toujours marcher, ce n'est qu'une heuristique)
        float3 averagePointPos = make_float3(0.0f);
        for(const Point& point : points->getPoints())
        {
            averagePointPos += point.pos;
        }

        averagePointPos /= points->size();

        orbitalControls.cameraTarget = averagePointPos;

        // Maintenant que l'on a le centre de la vue,
        // on estime encore une fois avec une heuristique le zoom adéquat
        // on prend le point le plus éloigné de la scène et on en définit le zoom de la caméra
        const Point& furthestPoint = *std::max_element(
            begin(points->getPoints()), end(points->getPoints()),
            [&](const Point& a, const Point& b) {
            return lengthSquared(a.pos - orbitalControls.cameraTarget) < 
                   lengthSquared(b.pos - orbitalControls.cameraTarget);
        });

        const float arbitraryDistanceFactor = 1.0f;
        orbitalControls.cameraDistanceToTarget = length(furthestPoint.pos - orbitalControls.cameraTarget) * arbitraryDistanceFactor;

        orbitalControls.horizontalAngle = my::pi / 4.0f;
        orbitalControls.verticalAngle = my::pi / 4.0f;

        params.traversableHandle = traversableHandleStorage.handle;

        glfwSetScrollCallback(app.getWindow(), [](GLFWwindow* window, double xoffset, double yoffset) {

            // Ne pas scroller si on focus une fenêtre ImGUI
            if(!ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow))
            {
                float zoomSpeed = userScalable(window, 0.05f);

                orbitalControls.cameraDistanceToTarget *= (1.0f - yoffset * zoomSpeed);
            }
        });

        glfwSetCursorPosCallback(app.getWindow(), [](GLFWwindow* window, double xpos, double ypos) {
            
            static float2 previousPos = make_float2(xpos, ypos);
            const float2 currentPos = make_float2(xpos, ypos);

            // Déplacer uniquement si le bouton gauche est cliqué
            if( glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
            {
                const float speed = userScalable(window, 0.02f);

                const float2 deltaPos = currentPos - previousPos;

                orbitalControls.horizontalAngle += -deltaPos.x * speed;
                orbitalControls.verticalAngle += deltaPos.y * speed;
            }

            // Même si le bouton n'est pas appuyé, on enregistre la position précédente
            // Sinon quand l'utilisateur bouge la souris sans cliquer, cela fera un "saut" dans le déplacement
            previousPos = currentPos;
        });

        // On créé directement la texture la plus grande possible
        // Cela permet de ne pas réallouer la structure quand numRays change
        SuperSampling ssaa(maxWinTexWidth, maxWinTexHeight, SSAA_NUM_RAYS_MAX);
        ssaa.setNumRays(params.ssaaParams.numRays());
        ssaa.setSize(params.width, params.height);
        
        // Créer le tableau de curandState le + grand possible aussi
        managed_device_ptr rngStorage(sizeof(curandState) * maxWinTexWidth * maxWinTexHeight);

        params.rand = rngStorage.as<curandState>();

        app.onDraw = [&]() {

            orbitalControls.applyToCamera(params.camera, my::radians(gui.verticalFieldOfView), static_cast<float>(params.width) / params.height);

            {
                GuiArgs args;
                args.params = &params;
                args.orbitalControls = &orbitalControls;
                args.ssaa = &ssaa;
                args.rect = rect.get();
                args.pbo = &pbo;
                
                gui.draw(args);
            }

            // Map la texture OpenGL dans un tableau CUDA,
            // Puis effectue l'interpolation sur ce tableau
            // http://cuda-programming.blogspot.com/2013/02/cuda-array-in-cuda-how-to-use-cuda.html
            
            if(OPTIMIZE_SUPERSAMPLE)
            {
                params.image = ssaa.getBufferDeviceData();

                {
                    const StreamScatter scatter(params.width, params.height, numStreams);

                    // Tableau de tous les paramètres des streams
                    // Allouer / free / copier avant car
                    // elles sont effectuées de façon synchrone / bloquante
                    // Et aussi cela permet de n'effectuer qu'une seule allocation, plus rapide

                    managed_device_ptr d_param_array_storage(sizeof(Params) * numStreams);
                    Params* d_param_array = d_param_array_storage.as<Params>();
                    
                    for(int s = 0; s < numStreams; s++)
                    {
                        params.offsetIdx = scatter.start(s);
                        d_param_array_storage.subfill(&params, sizeof(params), sizeof(params) * s);
                    }

                    for(int s = 0; s < numStreams; s++)
                    {
                        cudaStream_t stream = streams[s];

                        const uint2 numThreads = scatter.size(s);

                        // Copier params sur le GPU de façon asynchrone
                        // Sinon cela empêche la parallélisation des streams

                        OPTIX_CHECK(optixLaunch(
                            pipelines[s],
                            stream,
                            reinterpret_cast<CUdeviceptr>(&d_param_array[s]), sizeof(Params),
                            &sbt,
                            numThreads.x, numThreads.y, params.ssaaParams.numRays()
                        ));
                    }
                }

                // On ne synchronise pas là
                // Car chaque stream est indépendant:
                // Un stream interpolera les mêmes pixels que ceux qu'il a ray tracé

                {
                    void* d_image = pbo.map(stream);
                    
                    const double start = getTimeInSeconds();

                    ssaa.interpolate(reinterpret_cast<uchar3*>(d_image), streams, numStreams);
                    CUDA_CHECK(cudaDeviceSynchronize());
                    
                    performanceInfo.interpolationTimeInSeconds = getTimeInSeconds() - start;
                    
                    // On travaille en RGB (3 bytes), or par défaut OpenGL fonctionne avec 4 bytes
                    // Demande à travailler sur des muliples de 1 (donc n'importe quel nombre dont 3)
                    GL_CHECK(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));

                    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo.id()));
                    GL_CHECK(glBindTexture(GL_TEXTURE_2D, rect->getTexture()));
                    
                    // Actualise la sous-partie de la texture qui est active
                    // Contrairement à glTexImage2D(), glTexSubImage2D() ne redimensionne pas la texture 
                    GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, params.width, params.height, GL_RGB, GL_UNSIGNED_BYTE, 0));
                    GL_CHECK(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));

                    pbo.unmap(stream);
                }
            }
            else
            {
                void* d_image = pbo.map(stream);
                    
                params.image = reinterpret_cast<uchar3*>(d_image);
                managed_device_ptr d_params(&params, sizeof(params)); // Copier params sur le GPU

                const unsigned int depth = 1;
                OPTIX_CHECK(optixLaunch(
                    pipelines[0],
                    stream,
                    d_params, d_params.size(),
                    &sbt,
                    params.width, params.height, depth
                ));

                glBindTexture(GL_TEXTURE_2D, rect->getTexture());
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo.id());
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, params.width, params.height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

                pbo.unmap(stream);
            }
            
            // Utile ?
            CUDA_CHECK(cudaDeviceSynchronize());

            rect->draw();
            
            params.frame++;
        };

        app.display();
    }
    

    // ====== Destruction ======

    for(int s = 0; s < numStreams; s++)
    {
        OPTIX_CHECK(optixPipelineDestroy(pipelines[s]));
    }

    for(OptixProgramGroup group : groups) {
        OPTIX_CHECK(optixProgramGroupDestroy(group));
    }

    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));
    CUDA_CHECK(cudaStreamDestroy(stream));

    for(int i = 0; i < numStreams; i++)
    {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    printf("Programme termine avec succes.\n");
    return EXIT_SUCCESS;
}