#pragma once

#include <cstddef>
#include <array>

#include <cuda_runtime.h>

/**
 * Permet d'interpréter un tableau contiguë comme un tableau n-dimensionnel.
 *
 * @remarks
 * La dernière dimension possède toujours les cellules consécutives,
 * bon à savoir si on veut optimiser (data locality).
 */
template<typename T, size_t ndims>
class ArrayView
{
    static_assert(ndims >= 0, "Dimension must be positive");

private:
    template<typename... Dims>
    __device__ __host__ static void check_dimension(Dims...)
    {
        static_assert(sizeof...(Dims) == ndims,
            "Wrong number of arguments (must match the count of dimensions)");
    }

public:
    ArrayView() = default;

    /**
     * @example
     *  Pour déclarer une image RGB, utiliser ArrayView(uchar_data, 1920, 1080, 3)
     */
    template<typename... Sizes>
    __device__ __host__ ArrayView(T* data, Sizes... sizes)
        : m_data(data),
          m_size{static_cast<size_t>(sizes)...}
    {
        check_dimension(sizes...);
        
        m_pitch[ndims - 1] = 1;

        for(int dim = ndims - 2; dim >= 0; dim--)
        {
            // iterate in reverse
            m_pitch[dim] = m_pitch[dim + 1] * m_size[dim + 1];
        }
    }

    __device__ __host__ size_t size(size_t dim) const
    {
        return m_size[dim];
    }

    __device__ __host__ size_t dimensions() const
    {
        return ndims;
    }

    template<typename... Indices>
    __device__ __host__ T& operator()(Indices... indices)
    {
        return m_data[offset(indices...)];
    }

    template<typename... Indices>
    __device__ __host__ const T& operator()(Indices... indices) const
    {
        return m_data[offset(indices...)];
    }

    template<typename... Indices>
    __device__ __host__ constexpr size_t offset(Indices... indices)
    {
        check_dimension(indices...);

        const size_t indices_array[ndims] = {static_cast<size_t>(indices)...};
        size_t bytes = 0;
        for(size_t dim = 0; dim < ndims; dim++)
        {
            bytes += m_pitch[dim] * indices_array[dim];
        }

        return bytes;
    }
    
private:
    T* m_data = nullptr;

    /**
     * pitch[i] contains
     * the count of items to go from index [i] to index [i+1]
     * for the (i+1)-th dimension.
     */
    size_t m_pitch[ndims] = {};

    size_t m_size[ndims] = {};
};

template<typename T> using ArrayView2D = ArrayView<T, 2>;
template<typename T> using ArrayView3D = ArrayView<T, 3>;