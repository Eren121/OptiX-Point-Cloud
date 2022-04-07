#include "Parameters.h"
#include <imgui.h>
#include <sstream>
#include "core/utility/string.h"

bool drawGui(SsaaParameters& params)
{
    bool changed = false;

    // Type
    if(ImGui::BeginListBox("Pattern"))
    {
        // Comme on utilise des bords très arrondis,
        // Evite de déborder pour la première et la dernière ligne
        ImGui::NewLine();

        for(int i = 0; i < SSAA_Count; i++)
        {
            const bool is_selected = (params.type == i);
            if(ImGui::Selectable(ssaaNames[i], is_selected))
            {
                changed = true;
                params.type = static_cast<SsaaType>(i);
            }
            
            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if (is_selected)
            {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::NewLine();
        ImGui::EndListBox();
    }

    // Ray count

    const int maxRaysSqrt = sqrt(SSAA_NUM_RAYS_MAX);

    std::string format;
    {
        std::ostringstream ss;
        
        if(ssaaIs2D(params.type))
        {
            ss << params.numRaysSqrt() << "x" << params.numRaysSqrt() << " (" << params.numRays() << ")";
        }
        else
        {
            ss << params.numRays();
        }

        format = ss.str();
    }

    int numRaysSqrt = params.numRaysSqrt();
    
    if(ImGui::SliderInt("num. rays per pixel", &numRaysSqrt, 1, maxRaysSqrt, format.c_str()))
    {
        changed = true;
        params.setNumRays(numRaysSqrt * numRaysSqrt);
    }

    ImGui::Indent();
    if(ImGui::CollapsingHeader("Options"))
    {
        int x = 0;
        
        if(drawGui(params.options))
        {
            changed = true;
        }
    }
    ImGui::Unindent();

    return changed;
}