#include "Gui.hpp"
#include "imgui.h"
#include "helper_math.h"
#include "Distribution.hpp"
#include "core/utility/time.h"

void Gui::draw(Params& params, OrbitalControls& orbitalControls)
{
    const ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_DefaultOpen;
    
    if(showDemoWindow)
    {
        ImGui::ShowDemoWindow(&showDemoWindow);
    }

    if(ImGui::Begin("Interface", &showInterface))
    {
        if(ImGui::CollapsingHeader("Géométrie"))
        {
            // On laisse un peu de marge pour montrer que ça ne change pas si > sqrt(3)
            // La couleur peut changer quand même un peu à cause du calcul comme Phong et les vecteurs associés calculés
            ImGui::SliderFloat("point radius", &params.pointRadiusModifier, 0.0f, sqrt(3.0f) * 1.2f);
        }

        if(ImGui::CollapsingHeader("Lumière"))
        {
            static float3 lightDirection = params.lightDirection;
            if(ImGui::SliderFloat3("Direction", &lightDirection.x, -1.0f, 1.0f))
            {
                params.lightDirection = normalize(lightDirection);
            }

            ImGui::Checkbox("Shadow ray", &params.shadowRayEnabled);
        }

        if(ImGui::CollapsingHeader("Caméra"))
        {   
            // éviter d'être complètement à la verticale (+/- 180°),
            // sinon le produit vectoriel avec worldUp sera toujours nul
            // et donc il y aura un "gap" dans l'affichage et la caméra
            // sera orientée n'importe comment
            const float pitchLimit = (my::pi / 2.0f - 0.001f);

            ImGui::SliderFloat("Angle horizontal", &orbitalControls.horizontalAngle, -my::pi, my::pi);
            ImGui::SliderFloat("Angle vertical", &orbitalControls.verticalAngle, -pitchLimit, pitchLimit);
            ImGui::SliderFloat("distance", &orbitalControls.cameraDistanceToTarget, 0.0001f, 1.0f);
            
            float3 rel = orbitalControls.getCameraRelativePosition();
            ImGui::InputFloat3("position", &rel.x);

            float realDist = length(rel);
            ImGui::InputFloat("real distance", &realDist);

            float2 fov = make_float2(
                my::degrees(params.camera.verticalFieldOfView),
                my::degrees(params.camera.horizontalFieldOfView)
            );
            ImGui::InputFloat2("FOV", &fov.x);

            
            static float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
            if(ImGui::SliderFloat3("World Up", &worldUp.x, -1.0f, 1.0f))
            {
                orbitalControls.worldUp = normalize(worldUp);
            }   
        }
        if(ImGui::CollapsingHeader("Ray casting"))
        {
            glm::ivec2 countRaysPerPixel = params.countRaysPerPixel;
            ImGui::SliderInt2("Rayons par pixel", &countRaysPerPixel.x, 1, 10);
                
            /*
            // Inspiré en grande partie de la partie "Canvas" de la démo ImGui.
            // Note: pour ImGUI, le curseur signifie la position actuel de rendu (et pas la souris de l'utilisateur)

            ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
            ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
            ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
s   
            // Draw border and background color
            ImGuiIO& io = ImGui::GetIO();
            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
            draw_list->AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

            if(canvas_sz.x != 0 && canvas_sz.y != 0) // Evite un crash si la taille est zéro
            {
                // Permet de donner une taille réelle à notre canvas et d'avancer le curseur ImGui 
                ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
                

                for(int x = 0; x < countRaysPerPixel.x; ++x)
                {
                    for(int y = 0; y < countRaysPerPixel.y; ++y)
                    {
                        const glm::vec2 pos = Distribution::linspace<glm::vec2>(glm::ivec2(x, y), countRaysPerPixel);
                        ImVec2 center;
                        center.x = canvas_p0.x + pos.x * canvas_sz.x;
                        center.y = canvas_p0.y + pos.y * canvas_sz.y;
                        draw_list->AddCircleFilled(center, 1.0f, IM_COL32(255, 255, 255, 255));
                    }
                }

                draw_list->PushClipRect(canvas_p0, canvas_p1, true);
            }*/

            params.countRaysPerPixel = countRaysPerPixel;
        }
        if(ImGui::CollapsingHeader("Debug"))
        {
            ImGui::Checkbox("ImGUI demo", &showDemoWindow);

            {
                static double previousTime = getTimeInSeconds();
                const double currentTime = getTimeInSeconds();
                
                const double frameTimeMillis = (currentTime - previousTime) * 1000.0;

                ImGui::Text("frame time: %.2lf ms", frameTimeMillis);

                previousTime = currentTime;
            }
        }
    }
    ImGui::End();
}