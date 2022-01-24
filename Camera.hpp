#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "helper_math.h"

struct Camera
{
    // Position de la caméra (worldSpace)
    float3 origin    = make_float3(0.0f, 0.0f, 1.0f);

    // Dans quelle direction regarde la caméra relativement à origin (worldSpace)
    // direction doit être unitaire
    float3 direction = make_float3(0.0f, 0.0f, -1.0f);

    // Pour pouvoir savoir comment orienter l'image, on doit savoir
    // quel direction est le haut
    // Il s'agit d'une paire de vecteurs qui indique à quelle position
    // mapper chaque pixel.
    // Le pixel tout à gauche se déplacera de u.
    // Le pixel tout à droite se déplacera de -u.
    // Le pixel en haut se déplacera de v.
    // Le pixel en bas se déplacera de -v.
    // u, v doivent être unitaires
    float3 u = make_float3(1.0f, 0.0f, 0.0f);
    float3 v = make_float3(0.0f, 1.0f, 0.0f);

    // Calcul les vecteurs u et v selon le vecteur "up" (normalisé) local à la caméra
    // et la direction courante de la caméra
    // @param size La taille de la caméra en projection ortographique =
    // taille du volume que la caméra peut voir
    void computeUVFromUpVector(const float3& up, float2 size)
    {
        u = -cross(up, direction);
        v = cross(u, direction);

        // A ce moment, u et v doivent être des vecteurs unitaires
        // On ajuste leur taille pour s'adapter au viewport donné
        // On divise par 2 car chaque thread peut aller de -1.0f à 1.0f dans chaque dimension

        u *= size.x / 2.0f;
        v *= size.y / 2.0f;
    }
};

#endif /* CAMERA_HPP */
