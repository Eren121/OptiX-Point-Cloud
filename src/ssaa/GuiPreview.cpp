#include "GuiPreview.h"
#include <imgui.h>
#include <curand_kernel.h>
#include "core/cuda/math.h"
#include "core/utility/string.h"
#include <algorithm>

void SsaaGuiPreview::draw()
{
    if(isOpen)
    {    
        if(m_firstDraw)
        {
            m_firstDraw = false;
            reload();
        }

        if(ImGui::Begin("SSAA preview", &isOpen))
        {
            if(drawGui(m_params))
            {
                reload();
            }

            if(ImGui::SliderInt("num. pixels", &m_numPixels, 1, 16, join(m_numPixels, "x", m_numPixels).c_str()))
            {
                reload();
            }

            drawCanvas();
        }
        ImGui::End();
    }
}

void SsaaGuiPreview::reload()
{
    curandState rand;
    curand_init(1234, 0, 0, &rand);

    SsaaContext ctxt;
    ctxt.setNumRays(m_params.numRays());
    ctxt.rand = &rand;
    ctxt.options = &m_params.options;

    // m_points contains points in screen coords [0;1]^2

    m_points.clear();

    for(int px = 0; px < m_numPixels; px++)
    {
        for(int py = 0; py < m_numPixels; py++)
        {
            for(int r = 0; r < m_params.numRays(); r++)
            {
                ctxt.id = r;
                ssaaApply(m_params.type, ctxt);

                float2 point;
                point.x = unNormalize<float>(ctxt.out_pos.x, px, px + 1);
                point.y = unNormalize<float>(ctxt.out_pos.y, py, py + 1);

                m_points.push_back(point);
            }
        }
    }
}

void SsaaGuiPreview::drawCanvas()
{
    const int numRays = m_params.numRays();

    // Inspiré en grande partie de la partie "Canvas" de la démo ImGui.
    // Note: pour ImGUI, le curseur signifie la position actuel de rendu (et pas la souris de l'utilisateur)

    const int padding = 5; // unit: px

    ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
    canvas_p0.x += padding;
    canvas_p0.y += padding;
    ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
    canvas_sz.x -= padding * 2;
    canvas_sz.y -= padding * 2;

    ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
   
    
    // Draw border and background color
    ImGuiIO& io = ImGui::GetIO();
    ImDrawList& draw_list = *ImGui::GetWindowDrawList();
    draw_list.AddRectFilled(canvas_p0, canvas_p1, IM_COL32(50, 50, 50, 255));
    draw_list.AddRect(canvas_p0, canvas_p1, IM_COL32(255, 255, 255, 255));

    const ImColor point_color(255, 255, 0, 255);
    const ImColor line_color(0, 255, 0);

    float point_radius; // unit: px

    // Pour mieux voir, on donne une taille de point qui dépend du nombre de pixels
    {
        const float px_size = min(canvas_sz.x, canvas_sz.y) / m_numPixels;

        const float ratio = 20.0f; // cb. de points pour faire 1 pixel de long
        point_radius = max(1.0f, px_size / ratio);
    }
    
    // avoid div. by 0
    if(canvas_sz.x != 0 && canvas_sz.y != 0)
    {
        // Permet de donner une taille réelle à notre canvas et d'avancer le curseur ImGui 
        ImGui::InvisibleButton("canvas", canvas_sz, ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
        
        // Draw pixels limits (grid)
        {
            // Vertical lines
            for(int lx = 1; lx < m_numPixels; lx++)
            {
                const float x = static_cast<float>(lx) / m_numPixels;
                ImVec2 p1;
                p1.x = canvas_p0.x + x * canvas_sz.x;
                p1.y = canvas_p0.y;

                ImVec2 p2;
                p2.x = p1.x;
                p2.y = canvas_p1.y;

                draw_list.AddLine(p1, p2, line_color);
            }

            // Horizontal lines
            for(int ly = 1; ly < m_numPixels; ly++)
            {
                const float y = static_cast<float>(ly) / m_numPixels;
                ImVec2 p1;
                p1.x = canvas_p0.x;
                p1.y = canvas_p0.y + y * canvas_sz.y;

                ImVec2 p2;
                p2.x = canvas_p1.x;
                p2.y = p1.y;

                draw_list.AddLine(p1, p2, line_color);
            }
        }
        // Draw all points
        for(const float2& p : m_points)
        {
            ImVec2 center;
            center.x = canvas_p0.x + p.x * canvas_sz.x / m_numPixels;
            center.y = canvas_p0.y + p.y * canvas_sz.y / m_numPixels;

            draw_list.AddCircleFilled(center, point_radius, point_color);
        }

        draw_list.PushClipRect(canvas_p0, canvas_p1, true);
    }
}