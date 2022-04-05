#include "time.h"
#include <chrono>

double getTimeInSeconds()
{
    static const auto origin = std::chrono::steady_clock::now();
    
    const auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = now - origin;
    return dt.count();
}