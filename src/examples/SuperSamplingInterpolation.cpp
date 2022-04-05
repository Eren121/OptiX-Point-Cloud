#include "ssaa/SuperSampling.h"
#include "core/utility/ArrayView.h"
#include "stb_image_write.h"
#include <vector>

static std::vector<uchar3> generate_image(int w, int h, int depth)
{
    size_t size = w * h * depth;
    std::vector<uchar3> image(size);
    ArrayView3D<uchar3> view(image.data(), w, h, depth);

    for(int x = 0; x < w; x++)
    {
        for(int y = 0; y < h; y++)
        {
            uchar3 px;
            px.x = x % 255;
            px.y = px.x;

            for(int z = 0; z < depth; z++)
            {
                px.z = (z * 100) % 255;
                view(x, y, z) = px;
            }
        }
    }

    return image;
}

int main(int argc, char* argv[])
{
    const int width = 200;
    const int height = 200;
    const int subPixelsCount = 16;

    managed_device_ptr d_interpolated(width * height * sizeof(uchar3));

    SuperSampling ss(width, height, subPixelsCount);
    
    std::vector<uchar3> original = generate_image(width, height, 1);

    // We can't save the supersampled image because images are 2D but the supersampled one is 3D
    stbi_write_png("supersampled_original.png", width, height, 3, original.data(), 0);

    std::vector<uchar3> supersampled = generate_image(width, height, subPixelsCount);
    ss.getBufferDevice().fill(supersampled.data(), supersampled.size() * sizeof(uchar3));
    ss.interpolate(d_interpolated.as<uchar3>());

    std::vector<uchar3> interpolated(width * height);
    d_interpolated.download(interpolated.data());

    stbi_write_png("supersampled_interpolated.png", width, height, 3, interpolated.data(), 0);

    return 0;
}