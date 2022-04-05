#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "helper_math.h"
#include "Transform.hpp"

struct Camera
{
    /**
     * La même fonction utilitaire que Unity.
     * Souvent, on fixe le FOV vertical et le FOV horizontal est calculé
     * en fonction de la taille de la fenêtre.
     * Cette fonction permet d'obtenir le FOV horizontal en connaissant
     * ces deux autres paramètres.
     * @param verticalFieldOfView Le FOV sur l'axe vertical de l'écran.
     * @param aspectRatio Le ratio écran width/height en pixels
     */
    static float verticalToHorizontalFieldOfView(float verticalFieldOfView, float aspectRatio)
    {
        return 2.0f * atan(tan(verticalFieldOfView / 2.0f) * aspectRatio);
    }

    // Position de la caméra (worldSpace)
    float3 origin = make_float3(0.0f, 0.0f, 1.0f);

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

    float horizontalFieldOfView;
    float verticalFieldOfView;

    __host__ __device__ float3 getLook() const
    {
        return -transform.w;
    }

    __host__ __device__ float3 getLeft() const
    {
        return -transform.u;
    }

    __host__ __device__ float3 getRight() const
    {
        return transform.u;
    }

    __host__ __device__ float3 getUp() const
    {
        return transform.v;
    }

    __host__ __device__ float3 getDown() const
    {
        return -transform.v;
    }

    /**
     * Position et système de coordonnées de la caméra.
     * 
     * transform.position exprime la position de la caméra.
     * transform.u exprime la direction droite de la caméra.
     * transform.v exprime la direction haut de la caméra.
     * -transform.w exprime la direction de la caméra (vers où elle regarde).
     * 
     * @remarks
     * Exprimé en coordonnées monde.
     */
    Transform transform;
    
    // Calcul les vecteurs u et v selon le vecteur "up" (normalisé) local à la caméra
    // et la direction courante de la caméra
    // @param size La taille de la caméra en projection ortographique =
    // taille du volume que la caméra peut voir
    void computeUVFromUpVector(const float3& up, float2 size)
    {
        u = -normalize(cross(up, direction));
        v = cross(u, direction);

        // A ce moment, u et v doivent être des vecteurs unitaires
        // On ajuste leur taille pour s'adapter au viewport donné
        // On divise par 2 car chaque thread peut aller de -1.0f à 1.0f dans chaque dimension

        u *= size.x / 2.0f;
        v *= size.y / 2.0f;
    } 

    /**
     * Construit un repère en connaissant seulement la direction et le vecteur monde qui indique le haut.
     * Procédé connu sous le nom de algorithme de Gram-Schmidt.
     * 
     * @return Une Transform à donner en valeur à Camera::transform.
     * 
     * @param position La position de la caméra.
     * @param lookDirection La direction du regard de la caméra (normalisé).
     * @param worldUpVector Le vecteur qui indique le haut du monde (normalisé).
     */
    static Transform buildCameraTransform(float3 position, float3 lookDirection, float3 worldUpVector)
    {
        const float3 w = -lookDirection;
        const float3 u = normalize(cross(worldUpVector, -lookDirection));
        const float3 v = cross(w, u);

        return Transform(position, u, v, w);
    }
};

#endif /* CAMERA_HPP */
