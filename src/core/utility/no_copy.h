#pragma once

/**
 * Ajouter dans le corps d'une classe X NO_COPY(X) pour empêcher la copie.
 * La sémantique de mouvement est toujours possible si on définit T(T&&) et operator=(T&&)
 * en plus d'ajouter cette macro. Cependant, s'ils ne sont pas définis,
 * cela empêche la copie + le mouvement.
 */
#define NO_COPY(T) \
    T(const T&) = delete; \
    T& operator=(const T&) = delete;