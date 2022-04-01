#include <optix.h>
#include "ray.cuh"
#include "Record.hpp"
#include <cuda_runtime.h>
#include "helper_math.h"
#include "Point.h"
#include <climits>
#include "Distribution.hpp"

texture<uchar4, 2, cudaReadModeElementType> texRef;

extern "C"
{
    struct Payload
    {
        bool intersected = false;
        size_t primitiveID;
        float3 intersection;
    };

    static __forceinline__ __device__ void* unpackPointer( unsigned int i0, unsigned int i1 )
    {
        const unsigned long long uptr = static_cast<unsigned long long>( i0 ) << 32 | i1;
        void*           ptr = reinterpret_cast<void*>( uptr );
        return ptr;
    }
    
    
    static __forceinline__ __device__ void  packPointer( void* ptr, unsigned int& i0, unsigned int& i1 )
    {
        const unsigned long long uptr = reinterpret_cast<unsigned long long>( ptr );
        i0 = uptr >> 32;
        i1 = uptr & 0x00000000ffffffff;
    }

    __device__ int my_rand(void) // RAND_MAX assumed to be 32767
    {
        static unsigned long int next = 1;

        next = next * 1103515245 + 12345;
        return (unsigned int)(next/65536) % 32768;
    }

    __device__ inline float randf()
    {
        return (float)my_rand() / 32767.0f;
    }

    /**
     * Trace un rayon et retourne sa couleur
     */
    __device__ float3 traceRayAndGetColor(float3 rayOrigin, float3 rayDirection)
    {
        const Point* const pointBase = *reinterpret_cast<const Point**>(optixGetSbtDataPointer());

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
        
        const int maxDepth = 1;

        float3 rayColor = make_float3(0.0f);

        float3 currentRayOrigin = rayOrigin;
        float3 currentRayDirection = rayDirection;

        for(int depth = 0; depth < maxDepth; ++depth)
        {
            Payload payload;
            unsigned int u0;
            unsigned int u1;
            unsigned int u2 = UINT_MAX;
            packPointer(&payload, u0, u1);

            optixTrace(
                params.traversableHandle,
                currentRayOrigin,
                currentRayDirection,
                tmin,
                tmax,
                rayTime,
                visibilityMask,
                rayFlags,
                sbtOffset,
                sbtStride,
                missSbtIndex,
                u0,
                u1,
                u2
            );
            
            if(payload.intersected)
            {
                // Un point d'intersection a été trouvé
                // On rebondit sur la normale pour continuer le rayon
                // si la profondeur maximale n'a pas été atteinte
                // Et on actualise la couleur du rayon en mixant avec la couleur actuelle de rayon avec celle du point

                const Point& pointCollided = pointBase[payload.primitiveID];
                const float3& collisionPos = payload.intersection;
                const float3& pointColor = as_float3(pointCollided.col) / 255.0f;
                const float3 collisionPosNormal = normalize(payload.intersection - pointCollided.pos);

                // Actualise la couleur

                if(depth == 0) {
                    // Première itération, on ne mixe pas avec la couleur d'origine car il y en a pas encore

                    // Application de la lumière (Phong)
                    // Lumière directionnelle
                    const float3 lightColor = make_float3(1.0f, 1.0f, 1.0f);
                    const float3 ambientColor = make_float3(0.2f, 0.2f, 0.2f);
                    const float diffuseIntensity = my::ddot(-params.lightDirection, collisionPosNormal);

                    rayColor = pointColor * ambientColor;

                    // Vérifie si le point est dans l'ombre
                    if(params.shadowRayEnabled)
                    {
                        Payload payloadShadow;
                        unsigned int u0;
                        unsigned int u1;
                        unsigned int u2 = payload.primitiveID; // éviter de re-collisioner avec le même point
                        packPointer(&payloadShadow, u0, u1);
            
                        optixTrace(
                            params.traversableHandle,
                            collisionPos,
                            -params.lightDirection,
                            tmin,
                            tmax,
                            rayTime,
                            visibilityMask,
                            rayFlags,
                            sbtOffset,
                            sbtStride,
                            missSbtIndex,
                            u0,
                            u1,
                            u2
                        );

                        const bool inShadow = payloadShadow.intersected;

                        if(!inShadow) {
                            rayColor += pointColor * lightColor * diffuseIntensity;
                        }
                    }
                    else
                    {
                        rayColor += pointColor * lightColor * diffuseIntensity;
                    }
                }
                else {
                    // Ratio de la couleur courante à garder par rapport
                    // au prochain rayon
                    const float originalColorRatioToKeep = 0.1f;
                    rayColor = my::mix(pointColor, rayColor, originalColorRatioToKeep);
                }

                // Tracing récursif
                // Pour éviter de collider avec le même point car on part de sa surface, on décale légèrement au dessus de sa surface

                const float3 bouncingRayOrigin = pointCollided.pos + collisionPosNormal * pointCollided.r * params.pointRadiusModifier;
                const float3 bouncingRayDirection = reflect(currentRayDirection, collisionPosNormal);

                currentRayOrigin = bouncingRayOrigin;
                currentRayDirection = bouncingRayDirection;
            }
            else
            {
                break;
            }
        }        
        
        const float3 minColor = make_float3(0.0f);
        const float3 maxColor = make_float3(1.0f);
        return clamp(rayColor, minColor, maxColor);
    }

    __global__ void __raygen__my_program()
    {
        const Point* const pointBase = *reinterpret_cast<const Point**>(optixGetSbtDataPointer());

        const uint2 idx = make_uint2(optixGetLaunchIndex());
        const uint2 dim = make_uint2(optixGetLaunchDimensions());
        
        // Un kernel est lancé par pixel
        // gridPos définit la position 2D du pixel cible dans la fenêtre du rayon
        // griPos appartient à l'intervalle (-1;1)^2.
        glm::vec2 gridPos = (Distribution::linspace<glm::vec2>(to_ivec2(idx), to_ivec2(dim)) * 2.0f) - 1.0f;

        // L'origine du rayon est toujours l'origine de la caméra pour une perspective
        const float3 rayOrigin = params.camera.transform.position;
        
        const float halfFovHorizontal = params.camera.horizontalFieldOfView / 2.0f;
        const float halfFovVertical = params.camera.verticalFieldOfView / 2.0f;
        const float3 cameraLook = params.camera.getLook();
        
        const float threadAngleHorizontal = halfFovHorizontal * gridPos.x;
        const float threadAngleVertical = halfFovVertical * gridPos.y;

        // La taille dans le monde d'un pixel sur le plan Near
        const glm::vec2 pixelSizeOnNearPlane = {
            (2.0f * tan(halfFovHorizontal)) / static_cast<float>(dim.x),
            (2.0f * tan(halfFovVertical)) / static_cast<float>(dim.y)
        };
        
        // Lance le nombre désiré de rayons par pixel,
        // Mixe la couleur en faisant la moyenne
        float3 rayColorAverage;

        {
            float3 raysColorsSum = make_float3(0.0f);    
            glm::uvec2 pixelRayIndex;
            for(pixelRayIndex.x = 0; pixelRayIndex.x < params.countRaysPerPixel.x; ++pixelRayIndex.x)
            {
                for(pixelRayIndex.y = 0; pixelRayIndex.y < params.countRaysPerPixel.y; ++pixelRayIndex.y)
                {
                    // Soit le plan Near centré en 0
                    // pixelTarget donne les coordonnées du pixel cible sur ce plan Near
                    glm::vec2 pixelTarget = {
                        tan(threadAngleHorizontal),
                        tan(threadAngleVertical)
                    };

                    // Décalage propre à chaque sous-rayon pour un pixel donné
                    const glm::vec2 pixelRayOffset =
                        pixelSizeOnNearPlane
                        * unNormalize(Distribution::linspace<glm::vec2>(pixelRayIndex, params.countRaysPerPixel), glm::vec2(-0.5f), glm::vec2(0.5f));
                    
                    pixelTarget += pixelRayOffset;

                    // On considère un plan Near à z = 1.0f pour calculer les coordonnées des rayons
                    const float3 rayDirection = normalize(cameraLook
                        + pixelTarget.x * params.camera.getRight()
                        + pixelTarget.y * params.camera.getUp()
                    );

                    raysColorsSum += traceRayAndGetColor(rayOrigin, rayDirection);
                }
            }
            
            rayColorAverage = raysColorsSum / (static_cast<float>(params.countRaysPerPixel.x) * static_cast<float>(params.countRaysPerPixel.y));
        }
        
        uchar3& pixelRef = *params.at(idx.x, idx.y);
        pixelRef = convertFloatToCharColor(rayColorAverage);
    }

    __global__ void __anyhit__my_program()
    {
    }

    /**
     * @brief Récupère la position d'intersection depuis un programme CH.
     */
    __device__ float3 getIntersectionPos()
    {
        float3 intersectionPos;
        intersectionPos.x = int_as_float(optixGetAttribute_1());
        intersectionPos.y = int_as_float(optixGetAttribute_2());
        intersectionPos.z = int_as_float(optixGetAttribute_3());

        return intersectionPos;
    }

    __global__ void __closesthit__my_program()
    {
        const size_t primitiveID = optixGetPrimitiveIndex();
        const float3 intersectionPos = getIntersectionPos();

        const unsigned int u0 = optixGetPayload_0();
        const unsigned int u1 = optixGetPayload_1();
        Payload* const payload = reinterpret_cast<Payload*>(unpackPointer(u0, u1));
        
        payload->intersected = true;
        payload->primitiveID = primitiveID;
        payload->intersection = intersectionPos;
    }

    __global__ void __miss__my_program()
    {
        // Exécuté quand le rayon ne trouve pas de collision
        // Ne rien faire
        // La payload est déjà initialisé dans le programme à "intersected = false"
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

    __device__ void reportSphereIntersection(float t, float3 rayOrigin, float3 rayDirection)
    {
        const int hitKind = 0;
        const float3 intersectPos = rayOrigin + t * rayDirection;
        
        optixReportIntersection(t, hitKind,
            optixGetPrimitiveIndex(),
            float_as_int(intersectPos.x),
            float_as_int(intersectPos.y),
            float_as_int(intersectPos.z));
    }

    /**
    * @brief Programme d'intersection avec des sphères.
    * Position de la collision (x, y, z) stockée dans les attributs 1, 2 et 3
    * en float.
    * Index de la primitive stocké dans l'attribut 0
    */
    __global__ void __intersection__my_program()
    {
        const Point* pointBase = *reinterpret_cast<const Point**>(optixGetSbtDataPointer());
        const size_t primitiveIndex = optixGetPrimitiveIndex();
        const Point& point = pointBase[primitiveIndex];

        if(primitiveIndex == optixGetPayload_2())
        {
            // La payload 2 stocke l'index d'une primitive que l'on souhaite ignorer pour les collisions
            return;
        }
        
        const float3 ray_orig = optixGetWorldRayOrigin();
        const float3 ray_dir  = optixGetWorldRayDirection();
        const float  ray_tmin = optixGetRayTmin();
        const float  ray_tmax = optixGetRayTmax();

        const float3 c = point.pos;
        const float3 A = ray_orig;
        const float3 n = ray_dir;
        const float  r = point.r * params.pointRadiusModifier;

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
                        reportSphereIntersection(t, ray_orig, ray_dir);
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
                    
                    // Pas besoin de vérifier si on est dans la range,
                    // La doc indique explicitement que optixReportIntersection()
                    // ne fait rien si t n'est pas dans la range

                    reportSphereIntersection(t_near, ray_orig, ray_dir);
                    reportSphereIntersection(t_far, ray_orig, ray_dir);
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

    __global__ void __exception__my_program()
    {
        const int code = optixGetExceptionCode();
        //printf("[ExceptionProgram] Error %d\n", code);
            
        if(code == OPTIX_EXCEPTION_CODE_TRAVERSAL_INVALID_TRAVERSABLE)
        {
            OptixTraversableHandle handle = optixGetExceptionInvalidTraversable();
            //printf("[ExceptionProgram] handle = %p, params.handle = %p\n", (void*)handle, (void*)params.traversableHandle);
        }
    }
}