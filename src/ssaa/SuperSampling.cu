#include "SuperSampling.h"
#include "core/cuda/math.h"
#include "core/cuda/check.h"
#include "core/utility/ArrayView.h"
#include <cstdio>
#include <cstdlib>

/**
 * @param outputBuf L'image de sortie, interpolée de taille width x height
 * @param inputBuf L'image non-interpolée de taille width x height x subPixelsCount
 * @param width, height Taille de l'image de sortie
 * @param subPixelsCount Le nombre de sous-pixels.
 *
 * @remarks Les GPU sont 32 bits, donc int normaleemnt plus rapides que size_t
 * https://forums.developer.nvidia.com/t/signed-vs-unsigned-int-for-indexes-and-sizes/36054/2.
 *
 * @remarks
 * Comme les sous-pixels sont adjacents en mémoire, il est possible de traiter une seule partie
 * d'une image.
 */
__global__ void kernelSuperSamplingInterpolation(
    SuperSampling::Pixel* outputBuf,
    const SuperSampling::Pixel* inputBuf,
    int width, int height, int subPixelsCount)
{
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;

    // OpenGL / CUDA est row-major
    // Donc la height en 1ière dimension
    ArrayView<SuperSampling::Pixel, 2> output(outputBuf, height, width);
    ArrayView<const SuperSampling::Pixel, 3> input(inputBuf, height, width, subPixelsCount);

    if(px < width && py < height)
    {
        float3 total = make_float3(0.0f, 0.0f, 0.0f);
    
        for(int subPx = 0; subPx < subPixelsCount; subPx++)
        {
            const uchar3& subColor = input(py, px, subPx);
            total.x += subColor.x;
            total.y += subColor.y;
            total.z += subColor.z;
        }
        
        const float3 avg = total / static_cast<float>(subPixelsCount);
        output(py, px) = make_uchar3(avg.x, avg.y, avg.z);
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
    CUDA_CHECK_LAST_KERNEL();
}

void SuperSampling::setSize(int width, int height)
{
    // Vérifie que le stockage interne est assez grand
    const size_t targetSize =
        static_cast<size_t>(width) *
        static_cast<size_t>(height) *
        m_subPixelsCount * sizeof(Pixel);
    
    if(targetSize > m_d_buffer.size())
    {
        const char* fmt = "SuperSampling: impossible d'utiliser une taille (%d, %d) (capacity: %zu bytes)\n";
        std::fprintf(stderr, fmt, width, height, m_d_buffer.size());
        std::exit(1);
    }
    else
    {
        m_width = width;
        m_height = height;
    }
}

void SuperSampling::setNumRays(int numRays)
{
    // Vérifie que le stockage interne est assez grand
    const size_t targetSize =
        static_cast<size_t>(m_width) *
        static_cast<size_t>(m_height) *
        numRays * sizeof(Pixel);
    
    if(targetSize > m_d_buffer.size())
    {
        const char* fmt = "SuperSampling: impossible d'utiliser %d rays par pixel (capacity: %zu bytes)\n";
        std::fprintf(stderr, fmt, numRays, m_d_buffer.size());
        std::exit(1);
    }
    else
    {
        m_subPixelsCount = numRays;
    }
}

void SuperSampling::interpolate(Pixel* d_outputBuf, cudaStream_t* streams, int count)
{
    const uint w = static_cast<uint>(m_width);
    const uint h = static_cast<uint>(m_height);
    const uint sub = static_cast<uint>(m_subPixelsCount);
    
    // C'est parallélisable en streams car on effectue chaque somme de sous-pixels
    // Qui est entièrement indépendant pour chaque pixel
    // Même pas besoin de copie de données, toutes les données sont déjà sur le GPU

    // Vaut pour tous les streams sauf potentiellement le dernier si la taille
    // du tableau n'est pas un multiple du nombre de streams, et il en fait alors moins
    const int sub_img_max_size_y = ceil_div(h, count);

    ArrayView<SuperSampling::Pixel, 2> output(d_outputBuf, h, w);
    ArrayView<SuperSampling::Pixel, 3> input(m_d_buffer.as<Pixel>(), h, w, sub);

    for(int s = 0; s < count; s++)
    {
        // Division des données par stream
        // Chaque stream ne s'occupe pas d'une sous-région carré mais d'un certain nombre de lignes    
        // Note:
        // Chaque stream s'occupe de N lignes consécutives,
        // qui sont aussi consécutives en mémoire,
        // donc on exploite la localité des données et on évite les pagefaults => optimise

        int2 sub_img_start;
        sub_img_start.x = 0;
        sub_img_start.y = s * ceil_div(h, count);
        
        int2 sub_img_end;
        sub_img_end.x = w;
        sub_img_end.y = min(h, sub_img_start.y + sub_img_max_size_y);

        const int2 sub_img_size = sub_img_end - sub_img_start;
        
        Pixel& d_sub_img_output = output(sub_img_start.y, sub_img_start.x);
        const Pixel& d_sub_img_input = input(sub_img_start.y, sub_img_start.x, 0);
        
        const dim3 threadsPerBlock(16, 16);
        dim3 numBlocks;
        numBlocks.x = ceil_div(sub_img_size.x, threadsPerBlock.x);
        numBlocks.y = ceil_div(sub_img_size.y, threadsPerBlock.y);

        kernelSuperSamplingInterpolation<<<numBlocks, threadsPerBlock, 0, streams[s]>>>(
            &d_sub_img_output, &d_sub_img_input,
            sub_img_size.x, sub_img_size.y, sub);
        CUDA_CHECK_LAST_KERNEL();
    }
}