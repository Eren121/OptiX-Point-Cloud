#pragma once

#include "Parameters.h"
#include <vector>
#include <cuda_runtime.h>

class SsaaGuiPreview
{
public:
    void draw();

    bool isOpen = false;

private:
    void drawCanvas();
    void reload();

    SsaaParameters m_params;
    std::vector<float2> m_points; // coordinates are pixels
    int m_numPixels = 2; // Along one dimension
    bool m_firstDraw = true;
};