#include "SuperSampling.h"
#include "core/cuda/math.h"
#include "core/utility/ArrayView.h"

/**
 * @param outputBuf L'image de sortie, interpolée de taille width x height
 * @param inputBuf L'image non-interpolée de taille width x height x subPixelsCount
 * @param width, height Taille de l'image de sortie
 * @param subPixelsCount Le nombre de sous-pixels.
 *
 * @remarks Les GPU sont 32 bits, donc int normaleemnt plus rapides que size_t
 * https://forums.developer.nvidia.com/t/signed-vs-unsigned-int-for-indexes-and-sizes/36054/2.
 */
__global__ void kernelSuperSamplingInterpolation(
    SuperSampling::Pixel* outputBuf,
    const SuperSampling::Pixel* inputBuf,
    int width, int height, int subPixelsCount)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;

    ArrayView<SuperSampling::Pixel, 2> output(outputBuf, width, height);
    ArrayView<const SuperSampling::Pixel, 3> input(inputBuf, width, height, subPixelsCount);

    if(px < width && py < height)
    {
        float3 total = make_float3(0.0f, 0.0f, 0.0f);
    
        for(int subPx = 0; subPx < subPixelsCount; subPx++)
        {
            const uchar3 sub = input(px, py, subPx);
            total.x += sub.x;
            total.y += sub.y;
            total.z += sub.z;
        }
        
        const float3 avg = total / static_cast<float>(subPixelsCount);
        output(px, py) = make_uchar3(avg.x, avg.y, avg.z);
    }
}

SuperSampling::SuperSampling(int imageWidth, int imageHeight, int subPixelsCount)
    : m_width(imageWidth),
      m_height(imageHeight),
      m_subPixelsCount(subPixelsCount)
{
    // Potentiellement plusieurs Go de mémoire
    // On utilise size_t et on évite int

    const size_t bytes = m_width * m_height * m_subPixelsCount * sizeof(Pixel);
    m_d_buffer = managed_device_ptr(bytes);
}

void SuperSampling::interpolate(Pixel* d_output)
{
    const uint w = static_cast<uint>(m_width);
    const uint h = static_cast<uint>(m_height);
    const uint sub = static_cast<uint>(m_subPixelsCount);

    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks(ceil_div(w, threadsPerBlock.x), ceil_div(h, threadsPerBlock.y));
    kernelSuperSamplingInterpolation<<<numBlocks, threadsPerBlock>>>(
        d_output, m_d_buffer.as<Pixel>(),
        w, h, sub);
}