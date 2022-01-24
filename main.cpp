#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h> // On est pas dans un .cu sinon c'est inclus automatiquement
#include <optix.h>
#include <optix_stubs.h>
#include "ray.cuh"
#include "Record.hpp"
#include "stb_image_write.h" // Pour écrire des PNG
#include "Application.h"
#include <cuda_gl_interop.h> // Pour pouvoir utiliser les fonctionalités d'interopérabilité OpenGL de CUDA
#include "imgui.h"
#include "PointsCloud.hpp"
#include "SimpleGLRect.hpp"

std::unique_ptr<PointsCloud> points;

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
    pipelineCompileOptions.numAttributeValues = 0;

    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    // Si notre structure d'accélération ne possède que des triangles
    // ou des structures custom
    // on peut mettre un flag pour optimiser
    pipelineCompileOptions.usesPrimitiveTypeFlags = 0;

    // Pour debugger
    // _NONE en release
    pipelineCompileOptions.exceptionFlags =  OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;

    // Comme cet exemple n'a qu'une seule structure d'accélération,
    // cette option est nécessaire sinon erreur OPTIX_EXCEPTION_CODE_UNSUPPORTED_SINGLE_LEVEL_GAS
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;

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
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;

    // Informations de debugging. Un peu comme l'option de gcc -g.
    // En Debug il vaut mieux mettre à OPTIX_COMPILE_DEBUG_LEVEL_FULL, et en Release à OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT.
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

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
    const size_t nb_programs = 3;
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
    pipelineLinkOptions.maxTraceDepth = 1;
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

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
    options.logCallbackLevel = 3; // Ici, on affiche tout pour bien debugger


    // Vérifications d'erreurs en bonus à l'exécution
    // Cela permet encore de débugger, mais cela a un coût et peut ralentir le programme.
    // En Debug on pourra mettre OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL qui permet d'activer toutes les options bonus.
    // En Release on pourra mettre OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF.
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;

    // Création du contexte et vérification des erreurs
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


    const int strideInBytes = params.width * sizeof(params.image[0]);
    stbi_write_png("output.png", params.width, params.height, 4, image.data(), 0);
    system("output.png");
}

// Créer la structure accélératrice OptiX
// Cette partie est la plus importante, elle permet d'importer les géométries dans OptiX.
// C'est la seule partie qui dépend vraiment de l'application, le reste des fonctions (createPipeline(), createContext()...)
// sera a peu près pareil quelque soit l'application


TraversableHandleStorage createAccelerationStructure(OptixDeviceContext context, CUstream stream)
{
    // ====== Options de la structure accélératrice ======

    // Tableau de l'ensemble des géométries de la scène
    // On peut mettre plusieurs géométries mais toutes les géométries doivent avoir le même type (triangles, custom, curves, instances)
    std::vector<OptixBuildInput> buildInputs;

    // Options de la structure accélératrice
    OptixAccelBuildOptions options = {};
    options.buildFlags = OPTIX_BUILD_FLAG_NONE;
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
    for(const Point& point : points->getPoints())
    {
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
        buildInput.customPrimitiveArray.aabbBuffers = &d_aabbs.device_ptr;
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
    // d_output et d_temp doivent être alignés sur 128-bytes (peu importe car cudaMalloc() aligne toujours sur 256-bytes)
    CUDA_CHECK(cudaMalloc(&reinterpret_cast<void*>(outputHandle.d_output), bufferSizes.outputSizeInBytes));
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
        outputHandle.d_output, bufferSizes.outputSizeInBytes,
        &outputHandle.handle,
        nullptr, 0 // Propriétés non utilisées ici
    ));

    // ====== Désallocation ======

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_indices)));
    
    return outputHandle;
}

int main(int argc, char **argv)
{
    const char* path = R"(C:/Users/Raphael/Documents/RT1001_ProjetOptiX/data/bunny/reconstruction/bun_zipper.ply)";
    points = std::make_unique<PointsCloud>(path);

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

    OptixPipeline pipeline = createPipeline(context, groups);

    CUstream stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ====== Créer la structure accélératrice = envoyer les géométries sur le GPU ======

    TraversableHandleStorage traversableHandleStorage = createAccelerationStructure(context, stream); 

    // ====== Création des paramètres à envoyer au GPU pour le ray tracing ======

    RaygenRecord raygenRecord;
    raygenRecord.fill(raygenGroup);
    raygenRecord.data = make_uchar3(255, 255, 0);

    Record<void*> missRecord;
    missRecord.fill(missGroup);

    Record<CUdeviceptr> hitgroupRecord;

    // Alloue les informations de chaque points sur le GPU
    managed_device_ptr d_pointsCloud(points->data(), sizeof(Point) * points->size());
    hitgroupRecord.data = d_pointsCloud.device_ptr;
    hitgroupRecord.fill(hitGroup);

    managed_device_ptr d_raygenRecord(raygenRecord);
    managed_device_ptr d_missRecord(missRecord);
    managed_device_ptr d_hitgroupRecord(hitgroupRecord);
    
    // définir au minimum le raygenRecord et le missRecord
    OptixShaderBindingTable sbt = {};

    sbt.raygenRecord = d_raygenRecord.device_ptr;

    sbt.missRecordBase = d_missRecord.device_ptr;
    sbt.missRecordStrideInBytes = sizeof(missRecord);
    sbt.missRecordCount = 1;

    sbt.hitgroupRecordBase = d_hitgroupRecord.device_ptr;
    sbt.hitgroupRecordStrideInBytes = sizeof(hitgroupRecord);
    sbt.hitgroupRecordCount = 1;

    
    {
        const int width = 1920;
        const int height = 1080;
        Application app(width, height);

        // Contient la texture qui sera affiché sur toute la fenêtre à chaque Frame, et le VAO et le VBO associés à la texture.        
        SimpleGLRect rect(width, height);


        GLuint pbo;
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * 3, NULL, GL_STREAM_DRAW);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Contient les pixels de la fenêtre (= framebuffer) sous forme d'un pointeur vers le GPU.
        cudaGraphicsResource *textureResource = nullptr;

        // Indique à OpenGL que la texture sera utilisée à la fois par CUDA et OpenGL
        // On veut écrire l'image avec CUDA et l'afficher avec OpenGL d'où CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD.
        // Aide: https://forums.developer.nvidia.com/t/reading-opengl-texture-data-from-cuda/110370/3
        CUDA_CHECK(cudaGraphicsGLRegisterBuffer(
            &textureResource, pbo,
            cudaGraphicsRegisterFlagsWriteDiscard));


            
        Params params = {};
        params.width = width;
        params.height = height;
        params.traversableHandle = traversableHandleStorage.handle;

        app.onDraw = [&]() {

            
            CUDA_CHECK(cudaGraphicsMapResources(1, &textureResource, stream));

            // http://cuda-programming.blogspot.com/2013/02/cuda-array-in-cuda-how-to-use-cuda.html
            void *d_image = nullptr;
            size_t sizeInBytes;
            
            // Récupère le pointeur de la texture mappé sur CUDA (aussi dans le GPU...)
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&d_image, &sizeInBytes, textureResource));
            
            glBindTexture(GL_TEXTURE_2D, rect.getTexture());
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            // Détruit la ressource d'interoperabilité CUDA, à chaque frame
            CUDA_CHECK(cudaGraphicsUnmapResources(1, &textureResource, stream));

            const unsigned int depth = 1;

            params.image = reinterpret_cast<uchar3*>(d_image);
            
            // Copier params sur le GPU
            managed_device_ptr d_params(params);
    

            OPTIX_CHECK(optixLaunch(
                pipeline,
                stream,
                d_params.device_ptr,
                sizeof(Params),
                &sbt,
                params.width, params.height, depth
            ));

            // Utile ?
            CUDA_CHECK(cudaDeviceSynchronize());

            rect.draw();

            static bool showDemoWindow = false;

            if(showDemoWindow)
            {
                ImGui::ShowDemoWindow(&showDemoWindow);
            }
            
            /*
            static float t = 0.0f;
            t += 0.01f;

            params.camera.origin.x = cos(t) * SCALE;
            params.camera.origin.y = 0.0f;
            params.camera.origin.z = sin(t) * SCALE;
            params.camera.direction = normalize(-params.camera.origin);
            
            static float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
            params.camera.computeUVFromUpVector(worldUp, make_float2(width, height) / SCALE / 3.0f);
            */

            static bool showInterface = true;

            const ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_DefaultOpen;

            if(ImGui::Begin("Interface", &showInterface))
            {
                if(ImGui::CollapsingHeader("Nuage de points", nodeFlags))
                {
                }
                if(ImGui::CollapsingHeader("Caméra", nodeFlags))
                {
                    ImGui::SliderFloat3("Position", &params.camera.origin.x,
                        -2.0f * SCALE, 2.0f * SCALE);


                    if(ImGui::SliderFloat3("Direction", &params.camera.direction.x, -1.0f, 1.0f))
                    {
                        // Exécuté si l'utilisateur change la valeur
                        // Le vecteur doit être normalisé, on le normalise alors ici
                        params.camera.direction = normalize(params.camera.direction);
                    }

                    static float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
                    if(ImGui::SliderFloat3("World Up", &worldUp.x, -1.0f, 1.0f))
                    {
                        worldUp = normalize(worldUp);
                    }

                    static float2 cameraSize = make_float2(1.0f, 1.0f);
                    ImGui::SliderFloat("Viewport", &cameraSize.x, 0.1f, 3.0f);
                    cameraSize.y = cameraSize.x;
                    
                    params.camera.computeUVFromUpVector(worldUp, cameraSize);
                }
                if(ImGui::CollapsingHeader("OpenGL", nodeFlags))
                {
                    ImVec2 wsize(200, 200);
                    ImGui::Image(reinterpret_cast<ImTextureID>(rect.getTexture()), wsize, ImVec2(0, 1), ImVec2(1, 0));
                }
            }
            ImGui::End();
        };

        app.display();

        CUDA_CHECK(cudaGraphicsUnregisterResource(textureResource));
        textureResource = nullptr;

        glDeleteBuffers (1, &pbo);
    }
    

    // ====== Destruction ======

    // Détruit la structure accélératrice
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(traversableHandleStorage.d_output)));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));

    for(OptixProgramGroup group : groups) {
        OPTIX_CHECK(optixProgramGroupDestroy(group));
    }

    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));
    CUDA_CHECK(cudaStreamDestroy(stream));

    printf("Programme termine avec succes.\n");
    return EXIT_SUCCESS;
}