#pragma once

/**
 * All options for all patterns
 */
struct SsaaOptions
{
    struct
    {
        float dispersion = 0.5f; // by default can't go outside the pixel
    } random;
};

// Returns true if was changed by user
bool drawGui(SsaaOptions& options);