#pragma once

// Permet de détecter si est en build debug ou release
// En release: DEBUG_ENABLED=1
// En debug: DEBUG_ENABLED=0

// En fait on a besoin de ne rien faire que le CMakeLists.txt définit déjà
// la variable en mode Debug

#ifdef DEBUG // Macro définie par CMake en mode Debug
    #define DEBUG_ENABLED 1
#else
    #define DEBUG_ENABLED 0
#endif
