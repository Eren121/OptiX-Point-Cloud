#include "core/utility/ArrayView.h"
#include <iostream>

int main(int argc, char* argv[])
{
    const size_t X = 2;
    const size_t Y = 3;
    const size_t Z = 4;

    int data[X][Y][Z] = {};

    int value = 0;

    for(size_t x = 0; x < X; x++)
    {
        for(size_t y = 0; y < Y; y++)
        {
            for(size_t z = 0; z < Z; z++)
            {
                data[x][y][z] = value;
                value++;
            }
        }
    }

    ArrayView<int, 3> view(reinterpret_cast<int*>(data), X, Y, Z);

    std::cout << view(1, 2, 3) << std::endl; // 19

    return 0;
}