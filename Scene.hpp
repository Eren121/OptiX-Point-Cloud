#ifndef SCENE_HPP
#define SCENE_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>
#include "common.hpp"

TraversableHandleStorage mergeGASIntoIAS(OptixDeviceContext context, CUstream stream,
    OptixTraversableHandle geometry1, OptixTraversableHandle geometry2);

class Scene
{
public:
    Scene(OptixDeviceContext context, CUstream stream);

    static const float* getRowMajorIdentityMatrix();

    /**
     * @brief Copier une matrice identité 4x3 dans output
     * @param output 4*3 matrix row-major (of size 12).
     */
    static void getRowMajorIdentityMatrix(float output[12]);
};

#endif /* SCENE_HPP */
