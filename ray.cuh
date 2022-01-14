#pragma once

struct Params
{
    uchar4 *image;
    unsigned int width, height;
};

extern "C" __constant__ Params params;
