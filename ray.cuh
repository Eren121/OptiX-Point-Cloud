#pragma once

// La structure est read-only sur le GPU!!
// Mais elle peut contenir un pointeur vers une zone writable
struct Params
{
    uchar4 *image = nullptr;
    unsigned int width, height;
};

extern "C" __constant__ Params params;
