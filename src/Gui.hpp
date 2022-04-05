#ifndef GUI_HPP
#define GUI_HPP

#include "OrbitalControls.hpp"
#include "ray.cuh"

#include "imgui.h"

/**
 * Stocke toutes les données utiles pour le GUI.
 * Evite de polluer l'espace locale avec pleins de variables statiques (nécessaire pour se rappeler de la précédente valeur).
 */
class Gui
{
public:
    /**
     * Si l'interface demo de ImGUI (pour tester les widgets) doit être visible ou pas.
     */
    bool showDemoWindow = false;

    /**
     * Si l'interface doit être visible ou pas.
     */
    bool showInterface = true;

    /**
     * Le FOV vertical.
     * Le FOV horizontal ne peut pas être modifié car il est calculé en fonction du FOV vertical et de la taille de la fenêtre.
     */
    float verticalFieldOfView = 70.0f;

    /** 
     * @param params Les paramètres à envoyer au programme OptiX.
     */
    void draw(Params& params, OrbitalControls& orbitalControls);
};

#endif /* GUI_HPP */
