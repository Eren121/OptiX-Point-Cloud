#ifndef POINTCLOUD_HPP
#define POINTCLOUD_HPP

#include <string>
#include <vector>
#include "Point.h"

class PointsCloud
{
public:
    PointsCloud(const char *filename);

    std::vector<Point>& getPoints() { return m_points; }
    const std::vector<Point>& getPoints() const { return m_points; }
    
    Point* data() { return m_points.data(); }
    const Point* data() const { return m_points.data(); }

    size_t size() const { return m_points.size(); }

    /**
     * Assigne à chaque point une couleur aléatoire
     */
    void randomizeColors();

private:
    void readPlyFile(const std::string & filepath, const bool preload_into_memory = true);

private:
    std::vector<Point> m_points;
};

#endif /* POINTCLOUD_HPP */
