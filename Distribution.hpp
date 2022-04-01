#ifndef DISTRIBUTION_HPP
#define DISTRIBUTION_HPP

#include "common.hpp"

/**
 * Distribution des points dans une grille 2D.
 * Utile pour déterminer les directions de rayons selon la méthode choisie.
 */
class Distribution
{
public:
    /**
     * Répartit les points uniformément sans les bords.
     * On ne touche pas les bords car sinon pour les rayons des pixels adjacents
     * pourraient lancer des rayons vers les mêmes directions.
     * @param id = 0, 1, 2 ... count. id < count.
     * @param count Le nombre maximal de points.
     * @return Le nombre répartit sur (0;1).
     *
     * @remarks Template pour permettre d'être utilisé en dimension 1 (sur un segment) ou 2 (sur le carré unitaire, comme pour un pixel).
     */
    template<typename T = float, typename IndexType = uint>
    my_inline static T linspace(IndexType id, IndexType count)
    {
        return (T(0.5f) + static_cast<T>(id)) / static_cast<T>(count);
    }
};

#endif /* DISTRIBUTION_HPP */
