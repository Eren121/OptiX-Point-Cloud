#pragma once

#include <string>
#include <sstream>

template<typename... T>
std::string join(T&&... args)
{
    std::ostringstream ss;
    (ss << ... << std::forward<T>(args));
    return ss.str();
}