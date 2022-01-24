#include <optix.h>
#include "ray.cuh"
#include "Record.hpp"
#include <cuda_runtime.h>
#include "helper_math.h"
#include "Point.h"

texture<uchar4, 2, cudaReadModeElementType> texRef;

extern "C"
{
    __global__ void __raygen__my_program()
    {
        const uint2 idx = make_uint2(optixGetLaunchIndex());
        const uint2 dim = make_uint2(optixGetLaunchDimensions());
        
        const bool perspective = true;

        // Comme en OpenGL, on définit la taille de la fenêtre dans l'intervalle [-1;1]
        // Et on lance un rayon vers l'intérieur de la fenêtre donc en -Z
        float2 gridPos = (make_float2(idx) / make_float2(dim) * 2.0f) - 1.0f;
        float3 rayDirection, rayOrigin;

        if(perspective)
        {
            // Variable utile pour calculer la direction du rayon
            // Ici, on considère que:
            // A la distance znear, le champ de vision voit exactement la taille du viewport
            // définie par (length(params.u), length(params.v))
            const float znear = 1.0f;

            // L'origine du rayon est toujours l'origine de la caméra pour une perspective
            rayOrigin = params.camera.origin;

            
            // Point projecté à une distance znear
            const float3 target =
                gridPos.x * params.camera.u
              + gridPos.y * params.camera.v
              + params.camera.direction * znear;

            rayDirection = normalize(target);
        }
        else
        {
            // Ortographique

            rayDirection = params.camera.direction;

            // Passe de l'intervalle [-1;1] aux coordonnées caméra pour ce pixel
            rayOrigin = params.camera.origin +
                gridPos.x * params.camera.u * SCALE
              + gridPos.y * params.camera.v * SCALE;
            
        }
        // tmin: Distance minimum / maximum d'intersection
        const float tmin = 0.0f, tmax = 1e16f;

        const float rayTime = 0.0f; // Non-utilisé
        const unsigned int rayFlags = OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT;


        // Masque de visibilité
        // Une intersection est trouvée si au moyen un bit en commun est utilisé
        // Comme on utilisé pas de masque, on met tous les bits à 1 pour collisionner contre toutes les géométries
        // OptixVisibilityMask est un int mais seulement 8 bits sont vraiment utilisés
        // Si on met à 0, il ne s'affiche rien car aucun triangle ne collisionne
        const OptixVisibilityMask visibilityMask(255);

        // Non-utilisés
        const unsigned int sbtOffset = 0, sbtStride = 0; // ?

        // Indice du programm "__miss__"
        // On n'en contient qu'un seul, donc l'indice 0.
        const unsigned int missSbtIndex = 0;


        unsigned int payload_0;

        optixTrace(
            params.traversableHandle,
            rayOrigin,
            rayDirection,
            tmin,
            tmax,
            rayTime,
            visibilityMask,
            rayFlags,
            sbtOffset,
            sbtStride,
            missSbtIndex,
            payload_0
        );
        
        uchar3& pixel = *params.at(idx.x, idx.y);
        
        uchar3 rgb = int_as_uchar3(payload_0);
        pixel = rgb;
    }

    __global__ void __anyhit__my_program()
    {
    }

    __global__ void __closesthit__my_program()
    {
        const Point *pointBase = *reinterpret_cast<const Point**>(optixGetSbtDataPointer());
        const size_t primitiveIndex = optixGetPrimitiveIndex();
        const uchar3 rgb = pointBase[primitiveIndex].col;

        optixSetPayload_0(uchar3_as_int(rgb));
    }

    __global__ void __miss__my_program()
    {
        // Exécuté quand le rayon ne trouve pas de collision
        // Envoyer la couleur de fond
        optixSetPayload_0(uchar3_as_int(make_uchar3(0, 0, 0)));
    }

    #define USE_SPHERE 1

    #if USE_SPHERE

    __host__ __device__ inline int
    resolve_2nd_equation(float solutions[2], float a, float b, float c)
    {
        int solutionsCount = 0;
        const float delta = b*b - 4*a*c;

        if(delta < 0.0f)
        {
            // Pas de solution réelle
        }
        else if(delta == 0.0f)
        {
            // 1 solution
            solutionsCount = 1;
            solutions[0] = -b / (2.0f * a);
        }
        else
        {
            // 2 solutions
            solutionsCount = 2;
            solutions[0] = (-b - sqrt(delta)) / (2.0f * a);
            solutions[1] = (-b + sqrt(delta)) / (2.0f * a);
        }

        return solutionsCount;
    }

    /**
    * @brief Programme d'intersection avec des sphères provenant de l'exemple du SDK (cuda/sphere.cu)
    * changé avec les structures de données pour s'adapter pour ce code.
    * 
    */
    __global__ void __intersection__my_program()
    {
        const Point *pointBase = *reinterpret_cast<const Point**>(optixGetSbtDataPointer());
        const size_t primitiveIndex = optixGetPrimitiveIndex();
        const Point& point = pointBase[primitiveIndex];

        const float3 ray_orig = optixGetWorldRayOrigin();
        const float3 ray_dir  = optixGetWorldRayDirection();
        const float  ray_tmin = optixGetRayTmin();
        const float  ray_tmax = optixGetRayTmax();

        const float3 c = point.pos;
        const float3 A = ray_orig;
        const float3 n = ray_dir;
        const float  r = point.r;

        const float eq_a = lengthSquared(n);
        const float eq_b = -2.0f * dot(c - A, n);
        const float eq_c = lengthSquared(c - A) - r*r;

        float solutions[2];

        switch(resolve_2nd_equation(solutions, eq_a, eq_b, eq_c))
        {
            case 0:
                // Pas d'intersection
                break;
            
            case 1:
                {
                    // 1 intersection
                    const float t = solutions[0];
                    if(t > ray_tmin && t < ray_tmax) {
                        optixReportIntersection(t, 0);
                    }
                }
                break;

            case 2:
                {
                    // 2 intersections
                    // On reporte toujours la plus proche d'abord,
                    // Donc pour le t plus petit
                    float t_near, t_far;
                    if(solutions[0] < solutions[1])
                    {
                        t_near = solutions[0];
                        t_far = solutions[1];
                    }
                    else
                    {
                        t_near = solutions[1];
                        t_far = solutions[0];
                    }
                    
                    // Les conditions ne servent à rien,
                    // La doc indique explicitement que optixReportIntersection()
                    // ne fait rien si t n'est pas dans la range

                    //if(t_near > ray_tmin && t_near < ray_tmax)
                    {
                        optixReportIntersection(t_near, 0);
                    }
                    //if(t_far > ray_tmin && t_far < ray_tmax)
                    {
                        //optixReportIntersection(t_far, 0);
                    }
                }
                break;
        }


        /*

        const float3 O      = ray_orig - point.pos;
        const float  l      = 1.0f / length( ray_dir );
        const float3 D      = ray_dir * l;
        const float  radius = point.r;

        float b    = dot( O, D );
        float c    = dot( O, O ) - radius * radius;
        float disc = b * b - c;
        if( disc > 0.0f )
        {
            float sdisc        = sqrtf( disc );
            float root1        = ( -b - sdisc );
            float root11       = 0.0f;
            bool  check_second = true;

            const bool do_refine = fabsf( root1 ) > ( 10.0f * radius );

            if( do_refine )
            {
                // refine root1
                float3 O1 = O + root1 * D;
                b         = dot( O1, D );
                c         = dot( O1, O1 ) - radius * radius;
                disc      = b * b - c;

                if( disc > 0.0f )
                {
                    sdisc  = sqrtf( disc );
                    root11 = ( -b - sdisc );
                }
            }

            float  t;
            float3 normal;
            t = ( root1 + root11 ) * l;
            if( t > ray_tmin && t < ray_tmax )
            {
                normal = ( O + ( root1 + root11 ) * D ) / radius;
                if( optixReportIntersection( t, 0))
                    check_second = false;
            }

            if( check_second )
            {
                float root2 = ( -b + sdisc ) + ( do_refine ? root1 : 0 );
                t           = root2 * l;
                normal      = ( O + root2 * D ) / radius;
                if( t > ray_tmin && t < ray_tmax )
                    optixReportIntersection( t, 0);
            }
        }
        */
    }

    #else

    __global__ void __intersection__my_program()
    {
        // 0: le type de collision défini par l'utilisateur, non-utilisé
        
        const float t = optixGetRayTmin();
        optixReportIntersection(t, 0);
    }

    #endif
}