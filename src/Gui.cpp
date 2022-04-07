#include "Gui.hpp"
#include "SimpleGLRect.hpp"
#include "ray.cuh"
#include "imgui.h"
#include "helper_math.h"
#include "Distribution.hpp"
#include "core/utility/time.h"
#include <algorithm>
#include <sstream>
#include <string>
#include "git.h"

void Gui::draw(GuiArgs& args)
{
    Params& params = *args.params;
    OrbitalControls& orbitalControls = *args.orbitalControls;
    SuperSampling& ssaa = *args.ssaa;
    SimpleGLRect& rect = *args.rect;
    Pbo& pbo = *args.pbo;


    ///////////// Custom style
    {
        static bool first = true;
        if(first)
        {
            first = false;
            ImGuiStyle& style = ImGui::GetStyle();
            style.FrameRounding = 24.0f;
        }
    }
    /////////////

    const ImGuiTreeNodeFlags nodeFlags = ImGuiTreeNodeFlags_DefaultOpen;
    
    static double previousTime = getTimeInSeconds();
    const double currentTime = getTimeInSeconds();
    const double frameTimeMillis = (currentTime - previousTime) * 1000.0;

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
            
            ImGui::BeginDisabled();
            float3 rel = orbitalControls.getCameraRelativePosition();
            ImGui::InputFloat3("position", &rel.x);
            ImGui::EndDisabled();
            
            // >= 180.0f: peut entrainer des lags
            // pt. entraîne trop de NaN dans les fonctions trigo?
            ImGui::SliderFloat("FOVY", &verticalFieldOfView, 1.0f, 179.0f);
            ImGui::TextDisabled("FOVX = %.2f\n", my::degrees(params.camera.horizontalFieldOfView));

            
            static float3 worldUp = make_float3(0.0f, 1.0f, 0.0f);
            if(ImGui::SliderFloat3("World Up", &worldUp.x, -1.0f, 1.0f))
            {
                orbitalControls.worldUp = normalize(worldUp);
            }   
        }
        if(ImGui::CollapsingHeader("Ray casting"))
        {
            ImGui::Checkbox("show preview window", &m_ssaaPreview.isOpen);
            ImGui::Separator();

            if(drawGui(params.ssaaParams))
            {
                ssaa.setNumRays(params.ssaaParams.numRays());
            }
        }
        if(ImGui::CollapsingHeader("Performance"))
        {
            {
                // Normalement maxHeight / height a le même ratio,
                // ici on calcule sur width arbitrairement

                float r = static_cast<float>(params.width) / maxWinTexWidth;

                std::string format;
                {  
                    std::ostringstream ss;
                    ss << params.width << "x" << params.height;
                    format = ss.str();
                }

                if(ImGui::SliderFloat("num. pixels", &r, 0.0, 1.0f, format.c_str()))
                {
                    params.width = std::max(1, static_cast<int>(static_cast<float>(maxWinTexWidth) * r));
                    params.height = std::max(1, static_cast<int>(static_cast<float>(maxWinTexHeight) * r));
                    
                    ssaa.setSize(params.width, params.height);
                    rect.setRenderSize(params.width, params.height);
                    
                    // On est obligé de ré-allouer le PBO, mais cela devrait être relativement rapide
                    pbo.resize(params.width, params.height);
                }


                ImGui::TextColored(ImVec4(0, 1, 0, 1), "Num. pixels max: %dx%d Current: %dx%d (%d%%) ",
                    maxWinTexWidth, maxWinTexHeight,
                    params.width, params.height,
                    static_cast<int>(r * 100.0f));

                ImGui::Separator();
                
                {
                    const int fps = static_cast<int>(1000.0 / frameTimeMillis);
                    ImGui::TextColored(ImColor::HSV(0.3f, 0.8f, 0.8f), "frame time: %.2lf ms (fps: %d)", frameTimeMillis, fps);
                }

                ImGui::TextColored(ImColor::HSV(0.2f, 0.8f, 0.8f), "interpolation time: %.2lf ms", performanceInfo.interpolationTimeInSeconds * 1000.0);

                {
                    const int numRays = params.ssaaParams.numRays();
                    const size_t totNumRays = static_cast<size_t>(params.width) * params.height * numRays;
                    ImGui::TextColored(ImColor::HSV(0.857f, 0.8f, 0.8f), "num. rays traced per frame: %dx%dx%d=%zu", params.width, params.height, numRays, totNumRays);
                }
            }
        }
        if(ImGui::CollapsingHeader("Debug"))
        {
            ImGui::Checkbox("ImGUI demo", &showDemoWindow);
        }
        if(ImGui::CollapsingHeader("Build"))
        {
            ImGui::Text("Ray Tracer using OptiX");
            ImGui::Text("sha1: %s", GIT_HEAD_SHA1);
            ImGui::Text("date: %s", GIT_COMMIT_DATE_ISO8601);
            ImGui::Text("tag: %s", GIT_DESCRIBE);
        }
    }
    ImGui::End();

    m_ssaaPreview.draw();

    previousTime = currentTime;
}