#include "Scene.hpp"
#include "common.hpp"

namespace
{
    /**
     * @return Les options de optixAccelBuild() par défaut.
     */
    OptixAccelBuildOptions getDefaultBuildOptions()
    {
        OptixAccelBuildOptions options = {};
        options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        options.operation = OPTIX_BUILD_OPERATION_BUILD; // Créer la structure et pas l'actualiser
        options.motionOptions.numKeys = 0; // Pas de motion blur
        return options;
    }

    /**
     * @param traversableHandle Le GAS à utiliser par cette IAS.
     * @return OptixInstance Les options d'une IAS par défaut en utilisant le GAS donné.
     */
    OptixInstance getDefaultInstance(OptixTraversableHandle traversableHandle)
    {
        OptixInstance instance = {};
        Scene::getRowMajorIdentityMatrix(instance.transform);
        instance.instanceId = 0;
        instance.visibilityMask = 0xff;
        instance.sbtOffset = 0;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = traversableHandle;
        return instance;
    }

    /**
     * @brief Créer le OptixBuildInput pour stocker 1 IAS.
     * @param d_instance Tableau de OptixInstance de taille numInstances
     */
    OptixBuildInput createBuildInputIAS(const managed_device_ptr& d_instance, unsigned int numInstances)
    {
        OptixBuildInput buildInput = {};

        buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances = d_instance;
        buildInput.instanceArray.numInstances = numInstances;

        return buildInput;
    }

    struct AccelBuildBuffers {
        managed_device_ptr outputBuffer;
        managed_device_ptr tempBuffer;
    };

    /**
     * @brief Alloue la mémoire nécessaire pour construire la structure accélératrice (wrapper de optixAccelComputeMemoryUsage()).
     */
    AccelBuildBuffers accelComputeMemoryUsage(
        OptixDeviceContext context, const OptixAccelBuildOptions* options, const OptixBuildInput* input, unsigned int numInputs)
    {
        OptixAccelBufferSizes bufferSizes = {};

        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            context,
            options,
            input,
            numInputs,
            &bufferSizes
        ));

        AccelBuildBuffers buffers;
        buffers.tempBuffer = managed_device_ptr(bufferSizes.tempSizeInBytes);
        buffers.outputBuffer = managed_device_ptr(bufferSizes.outputSizeInBytes);
        return buffers;
    }

    /**
     * @brief Wrapper de optixAccelBuild().
     */
    void accelBuildInstanceArray(
        OptixDeviceContext context, CUstream stream, const OptixAccelBuildOptions* accelOptions,
        const OptixBuildInput* buildInputs, unsigned int numBuildInputs,
        AccelBuildBuffers& buffers, OptixTraversableHandle* outputHandle)
    {
        // Note:
        // Pour les "Instance Acceleration Structure" (IAS),
        // numBuildInput == 1 toujours

        OPTIX_CHECK(optixAccelBuild(
            context, stream,
            accelOptions,
            buildInputs, numBuildInputs,
            buffers.tempBuffer, buffers.tempBuffer.size(),
            buffers.outputBuffer, buffers.outputBuffer.size(),
            outputHandle,
            nullptr, 0 // Emitted properties, not used
        ));
    }
}

/**
 * Merge deux geometry-acceleration-structure dans un instance-acceleration-structure.
 * L'avantage est que les deux GAS sont indépendants, peuvent être modifiés et aussi contenir des
 * primitives différentes. C'est un moyen de mixer différentes primitives car un seul GAS ne peut contenir
 * qu'un même type de géométrie (triangles, curves ou custom).
 */
TraversableHandleStorage mergeGASIntoIAS(OptixDeviceContext context, CUstream stream,
    OptixTraversableHandle geometry1, OptixTraversableHandle geometry2)
{
    const OptixAccelBuildOptions buildOptions = getDefaultBuildOptions();

    const unsigned int numGeometries = 2;
    const unsigned int numBuildInputs = 1;
    const OptixInstance instances[numGeometries] = {
        getDefaultInstance(geometry1),
        getDefaultInstance(geometry2)
    };

    const managed_device_ptr d_instances(instances, sizeof(OptixInstance) * numGeometries);
    OptixBuildInput buildInput = createBuildInputIAS(d_instances, numGeometries);
    AccelBuildBuffers buffers = accelComputeMemoryUsage(context, &buildOptions, &buildInput, numBuildInputs);

    TraversableHandleStorage accelerationStructure;
    accelBuildInstanceArray(context, stream, &buildOptions, &buildInput, numBuildInputs, buffers, &accelerationStructure.handle);

    // ATTENTION
    // Quand le AccelBuildBuffers est détruit, les buffers qu'il contient sont détruit.
    // Il contient à la fois le buffer temporaire et le buffer de sortie.
    // Le buffer temporaire peut être détruit à la fin, il n'est utile que pendant l'initialisation.
    // Mais le buffer de sortie doit être gardé tout au long de l'application!
    // On donne alors le relai à TraversableHandleStorage, grâce à la "move semantic":
    // La responsabilité du buffer est passé au TraversableHandleStorage pour éviter d'être détruit à la fin de cette fonction
    // (4h à chercher d'ou venait le bug, donc je pense que c'est important...)

    accelerationStructure.d_storage = std::move(buffers.outputBuffer);

    return accelerationStructure;
}

Scene::Scene(OptixDeviceContext context, CUstream stream)
{
}

const float* Scene::getRowMajorIdentityMatrix()
{
    static const float rowMajorIdentityMatrix[12] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    };

    return rowMajorIdentityMatrix;
}

void Scene::getRowMajorIdentityMatrix(float output[12])
{
    memcpy(output, getRowMajorIdentityMatrix(), sizeof(float) * 4 * 3);
}