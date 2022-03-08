#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

/**
 * Triplet de vecteurs ainsi qu'une position pour exprimer une base en 3D.
 * Les trois vecteurs doivent être orthogonaux.
 */
class Transform
{
public:
    Transform()
        : position(make_float3(0.0f, 0.0f, 0.0f)),
          u(make_float3(1.0f, 0.0f, 0.0f)),
          v(make_float3(0.0f, 1.0f, 0.0f)),
          w(make_float3(0.0f, 0.0f, 1.0f))
    {
    }
    
    Transform(float3 position, float3 u, float3 v, float3 w)
        : position(position),
          u(u),
          v(v),
          w(w)
    {
    }

    float3 position;
    float3 u;
    float3 v;
    float3 w;
};

#endif // TRANSFORM_HPP