#include "managed_device_ptr.h"
#include "check.h"
#include <utility>
#include <cassert>

using std::swap;

managed_device_ptr::managed_device_ptr(const void *data, size_t size)
    : managed_device_ptr(size)
{
    fill(data, size);
}

managed_device_ptr::managed_device_ptr(size_t size)
    : m_size(size)
{
    CUDA_CHECK(cudaMalloc(&to_void_ptr(), size));
}

managed_device_ptr::managed_device_ptr(managed_device_ptr&& rhs)
{

    swap(m_device_ptr, rhs.m_device_ptr);
    swap(m_size, rhs.m_size);
}

managed_device_ptr& managed_device_ptr::operator=(managed_device_ptr&& rhs)
{
    swap(m_device_ptr, rhs.m_device_ptr);
    swap(m_size, rhs.m_size);

    return* this;
}

void*& managed_device_ptr::to_void_ptr()
{
    return reinterpret_cast<void*&>(m_device_ptr);
}

managed_device_ptr::~managed_device_ptr()
{
    CUDA_CHECK(cudaFree(to_void_ptr()));
    m_device_ptr = 0;
    m_size = 0;
}

void* managed_device_ptr::to_void_ptr() const
{
    return reinterpret_cast<void*>(m_device_ptr);
}

void managed_device_ptr::fill(const void *data, size_t size)
{
    assert(size <= this->size());

    CUDA_CHECK(cudaMemcpy(to_void_ptr(), data, size, cudaMemcpyHostToDevice));
}

void managed_device_ptr::download(void *data) const
{
    CUDA_CHECK(cudaMemcpy(data, to_void_ptr(), size(), cudaMemcpyDeviceToHost));
}

managed_device_ptr::operator void*() const
{
    return to_void_ptr();
}

managed_device_ptr::operator const CUdeviceptr&() const
{
    return m_device_ptr;
}

size_t managed_device_ptr::size() const
{
    return m_size;
}