#pragma once

#include <optix_stubs.h>

#define OPTIX_CHECK(expr) checkImplOptiX((expr), __FILE__, __LINE__)

void checkImplOptiX(OptixResult result, const char* file, int line);