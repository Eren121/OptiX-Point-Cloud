#include "Options.h"
#include "Type.h"
#include <imgui.h>

bool drawGui(SsaaOptions& options)
{
    bool changed = false;

    ImGui::Indent(); // CollapsingHeader() doesn't indent

    if(ImGui::CollapsingHeader(ssaaNames[SSAA_RANDOM]))
    {
        if(ImGui::DragFloat("dispersion", &options.random.dispersion, 0.001f, 0.0f, 100.0f, "%.3f px", 1.0f))
        {
            changed = true;
        }
    }

    ImGui::Unindent();

    return changed;
}