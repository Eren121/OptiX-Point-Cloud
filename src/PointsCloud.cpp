#include "PointsCloud.hpp"
#include <iostream>
#include <tinyply.h>
#include <fstream>
#include <optional>

using namespace tinyply;

template<typename T>
std::vector<T> convertPlyDataToVector(PlyData& data)
{
    const size_t numBytes = data.buffer.size_bytes();
    const size_t numData = data.count;
    std::vector<T> ret(numData);

    
    const T *in = reinterpret_cast<T*>(data.buffer.get());
    std::copy(in, in + numData, ret.begin());    

    return ret;
}

void PointsCloud::readPlyFile(const std::string & filepath, const bool preload_into_memory)
{
    // Source:
    // https://github.com/ddiakopoulos/tinyply/blob/master/source/example.cpp

    std::cout << "........................................................................\n";
    std::cout << "Now Reading: " << filepath << std::endl;

    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try
    {
        file_stream.reset(new std::ifstream(filepath, std::ios::binary));
        if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + filepath);

        file_stream->seekg(0, std::ios::end);
        const float size_mb = file_stream->tellg() * float(1e-6);
        file_stream->seekg(0, std::ios::beg);

        PlyFile file;
        file.parse_header(*file_stream);

        std::cout << "\t[ply_header] Type: " << (file.is_binary_file() ? "binary" : "ascii") << std::endl;
        for (const auto & c : file.get_comments()) std::cout << "\t[ply_header] Comment: " << c << std::endl;
        for (const auto & c : file.get_info()) std::cout << "\t[ply_header] Info: " << c << std::endl;

        for (const auto & e : file.get_elements())
        {
            std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
            for (const auto & p : e.properties)
            {
                std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
                if (p.isList) std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
                std::cout << std::endl;
            }
        }

        // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers. 
        // See examples below on how to marry your own application-specific data structures with this one. 
        std::shared_ptr<PlyData> vertices, normals, colors, radius;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties 
        // like vertex position are hard-coded: 
        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }


        try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { colors = file.request_properties_from_element("vertex", { "r", "g", "b", }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
        
        try { radius = file.request_properties_from_element("vertex", { "intensity", }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
        
        file.read(*file_stream);

        if (vertices)   std::cout << "\tRead " << vertices->count  << " total vertices "<< std::endl;
        if (normals)    std::cout << "\tRead " << normals->count   << " total vertex normals " << std::endl;
        if (colors)     std::cout << "\tRead " << colors->count << " total vertex colors " << std::endl;
        if (radius)     std::cout << "\tRead " << radius->count << " radius " << std::endl;

        m_points.resize(vertices->count);

        std::optional<std::vector<float3>> v_vertices;
        std::optional<std::vector<float3>> v_normales;
        std::optional<std::vector<uchar3>> v_colors;
        std::optional<std::vector<float>> v_radius;

        if(vertices) { v_vertices = convertPlyDataToVector<float3>(*vertices); }
        if(normals) { v_normales = convertPlyDataToVector<float3>(*normals); }
        if(colors) { v_colors = convertPlyDataToVector<uchar3>(*colors); }
        if(radius) { v_radius = convertPlyDataToVector<float>(*radius); }
        
        for(size_t i = 0; i < vertices->count; ++i)
        {
            Point p;

            if(v_vertices) {
                p.pos = (*v_vertices)[i];
            }

            if(v_normales) {
                p.nor = (*v_normales)[i];
            }

            // Couleur du point
            // Si le point n'a pas de couleur dans le fichier,
            // on donne une couleur blanc par défaut
            
            if(v_colors) {
                p.col = (*v_colors)[i];
            }
            else {
                p.col = make_uchar3(255, 255, 255);
            }

            if(v_radius) {
                p.r = (*v_radius)[i];
                p.r *= 0.0009f;
                //p.r *= SCALE;
                //p.r = 0.0001f;
            }
            //p.r = 0.0003f * SCALE;

            // On sait que l'indice i existe car on a effectué m_points.resize()
            m_points[i] = p;
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "Caught tinyply exception: " << e.what() << std::endl;
    }
}

void PointsCloud::randomizeColors()
{
    for(Point &point : m_points)
    {
        point.col.x = rand() % 255;
        point.col.y = rand() % 255;
        point.col.z = rand() % 255;
    }
}

PointsCloud::PointsCloud(const char *filename)
{
    readPlyFile(filename);
}