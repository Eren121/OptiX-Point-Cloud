#ifndef ORBITAL_CONTROLS_HPP
#define ORBITAL_CONTROLS_HPP

#include <cuda_runtime.h>
#include "Camera.hpp"

/**
 * Permet de contrôler une caméra orbitale.
 */
class OrbitalControls
{
public:
    /**
     * Constructeur.
     * Permet d'initialiser les contrôles suivant une position donnée,
     * une distance à cette position et le direction de la caméra.
     * La caméra regarde vers le centre et a pour origine
     * (center + direction * distance).
     * 
     * @param cameraTarget Le point cible que la caméra regarde.
     * @param distanceFromTarget La distance de la caméra au point cible.
     * @param directionFromTarget La direction pour aller de la cible à la caméra.
     */ 
    OrbitalControls(float3 cameraTarget, float distanceToTarget, float3 worldUpVector = make_float3(0.0f, 1.0f, 0.0f))
        : cameraDistanceToTarget(distanceToTarget),
          verticalAngle(0.0f),
          horizontalAngle(0.0f),
          cameraTarget(cameraTarget),
          worldUp(worldUpVector)
    {
    }

    /**
     * @return La position de la caméra relativement à la target.
     */
    float3 getCameraRelativePosition() const
    {
        return cameraDistanceToTarget * -getCameraLook();
    }

    float3 getCameraLook() const
    {
        return -aircraftAxes(horizontalAngle, verticalAngle);
    }

    float3 getCameraPosition() const
    {
        return cameraTarget + getCameraRelativePosition();
    }

    /**
     * Actualise la caméra avec les paramètres stockés dans cette classe.
     * @param camera La caméra sur laquelle appliquer la vue.
     * @param verticalFieldOfView Le FOV sur l'axe vertical de l'écran.
     * @param aspectRatio Le ratio écran x/y,
     * permet de calculer le fov de l'écran horizontalement (fovx).
     */
    void applyToCamera(Camera& camera, float verticalFieldOfView, float aspectRatio) const
    {
        camera.verticalFieldOfView = verticalFieldOfView;
        camera.horizontalFieldOfView = Camera::verticalToHorizontalFieldOfView(verticalFieldOfView, aspectRatio);
        camera.transform = Camera::buildCameraTransform(getCameraPosition(), getCameraLook(), worldUp);
    }

    /**
     * Coordonnées sphériques (r, phi, theta) de la position de la caméra centrée sur la target.
     */
    float cameraDistanceToTarget;
    float verticalAngle;
    float horizontalAngle;
    
    float3 cameraTarget;
    float3 worldUp;

private:
    /**
     * @param yaw Rotation horizontale (angle, radians).
     * @param pitch Rotation verticale (angle, radians).
     */
    static float3 aircraftAxes(float yaw, float pitch)
    {
        float x = cos(yaw);
        float z = -sin(yaw);
        
        const float y = sin(pitch);
        x *= cos(pitch);
        z *= cos(pitch);

        return make_float3(x, y, z);
    }
};

#endif