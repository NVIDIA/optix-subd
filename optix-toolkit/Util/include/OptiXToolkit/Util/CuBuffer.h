//
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// 
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once
#include <cassert>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <span>
#include <stdint.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Exception.h"

template <typename T = uint8_t>
struct CuBuffer
{
    typedef T value_type;

    CuBuffer()
        : _data{ nullptr }
        , _size{ 0 }
        , _alloc_size{ 0 }
    {
    }

    CuBuffer( size_t size )
        : _data{ nullptr }
        , _size{ size }
        , _alloc_size{ size }
    {
        CUDA_CHECK( cudaMalloc( &( _data ), _alloc_size * sizeof( T ) ) );
    }

    explicit CuBuffer( size_t size, const T* data )
        : _data{ nullptr }
        , _size{ size }
        , _alloc_size{ size }
    {
        CUDA_CHECK( cudaMalloc( &( _data ), _alloc_size * sizeof( T ) ) );
        CUDA_CHECK( cudaMemcpy( _data, data, _size * sizeof( T ), cudaMemcpyHostToDevice ) );
    }

    CuBuffer( const CuBuffer& ) = delete;

    CuBuffer( CuBuffer&& source ) noexcept { *this = std::move( source ); }

    CuBuffer( const std::vector<T>& vec )
        : CuBuffer( vec.size(), vec.data() )
    {
        ;
    }

    CuBuffer( const std::initializer_list<T>& init )
        : CuBuffer( init.size(), init.begin() )
    {
    }

    template <size_t N>
    CuBuffer( const T ( &array )[N] )
        : CuBuffer( N, array )
    {
    }

    ~CuBuffer() { CUDA_CHECK_NOTHROW( cudaFree( _data ) ); }

    __host__
    CuBuffer& operator=( const CuBuffer& source ) = delete;
    __host__
    CuBuffer& operator=( CuBuffer&& source ) noexcept
    {
        if( this != &source )
        {
            CUDA_CHECK_NOTHROW( cudaFree( _data ) );
            _data        = source._data;
            source._data       = 0;
            _size       = source._size;
            source._size      = 0;
            _alloc_size        = source._alloc_size;
            source._alloc_size = 0;
        }
        return *this;
    }

    __host__
    void reserve( size_t size )
    {
        if( _alloc_size < size )
        {
            CUDA_CHECK_NOTHROW( cudaFree( _data ) );
            _data = nullptr;
            if( size )
                CUDA_CHECK( cudaMalloc( &( _data ), size * sizeof( T ) ) );
            _alloc_size = size;
        }
    }

    __host__ void resize( size_t size )
    {
        if( _alloc_size < size )
        {
            CuBuffer<T> temp_buffer(size);
            CUDA_CHECK( cudaMemcpy( temp_buffer._data, _data, _size * sizeof( T ), cudaMemcpyDeviceToDevice ) );
            std::swap(*this, temp_buffer);
        }
        else
        {
            _size = size;
        }
    }

    __host__ __device__
    size_t empty() const { return _size == 0; }

    __host__ __device__
    size_t size() const { return _size; }

    __host__ __device__
    size_t capacity() const { return _alloc_size; }

    __host__ __device__
    T* data() { return _data; }

    __host__ __device__
    const T* data() const { return _data; }

    __host__ __device__
    CUdeviceptr cu_ptr( size_t index = 0 ) const { return reinterpret_cast<CUdeviceptr>( _data + index ); }

    __host__ __device__
    size_t size_in_bytes() const { return _size * sizeof( T ); }

    std::span<T> span() { return std::span<T>{ _data, _size }; }
    // TODO: std::span<const T>
    const std::span<T> span() const { return std::span<T>{ _data, _size }; }

    constexpr
    std::span<T> subspan(size_t offset, size_t size)
    {
        assert(offset + size <= _size);
        return std::span<T>{_data + offset, size};
    }

    // TODO: std::span<const T>
    constexpr
    const std::span<T> subspan(size_t offset, size_t size ) const
    {
        assert(offset + size <= _size);
        return std::span<T>{_data + offset, size};
    }

    void upload( const T* data )
    {
        CUDA_CHECK( cudaMemcpy( _data, data, _size * sizeof( T ), cudaMemcpyHostToDevice ) );
    }


    void upload( const T* data, size_t size )
    {
        reserve( size );
        OTK_ASSERT( size <= _alloc_size );
        _size = size;
        upload( data );
    }

    void upload( const std::vector<T>& vec ) { upload( vec.data(), vec.size() ); }

    void uploadSub( T const* data, size_t size, size_t offset ) 
    {
        assert( size + offset <= _alloc_size );
        CUDA_CHECK( cudaMemcpy( _data + offset, data, size * sizeof( T ), cudaMemcpyHostToDevice ) );
    }

    void uploadAsync( const T* data, cudaStream_t stream = 0 )
    {
        CUDA_CHECK( cudaMemcpyAsync( _data, data, _size * sizeof( T ), cudaMemcpyHostToDevice, stream ) );
    }

    void uploadAsync( const T* data, size_t size, cudaStream_t stream = 0 )
    {
        reserve( size );
        OTK_ASSERT( size <= _alloc_size );
        _size = size;
        uploadAsync( data, stream );
    }

    void uploadAsync( const std::vector<T>& vec, cudaStream_t stream = 0 ) { uploadAsync( vec.data(), vec.size(), stream ); }

    void download( T* data ) const
    {
        CUDA_CHECK( cudaMemcpy( data, _data, _size * sizeof( T ), cudaMemcpyDeviceToHost ) );
    }

    void download( std::vector<T>& vec ) const
    {
        vec.resize( _size );
        CUDA_CHECK( cudaMemcpy( vec.data(), _data, _size * sizeof( T ), cudaMemcpyDeviceToHost ) );
    }

    void downloadSub( size_t size, size_t offset, T* data ) const
    {
        OTK_ASSERT( size + offset <= _alloc_size );
        CUDA_CHECK( cudaMemcpy( data, _data + offset, size * sizeof( T ), cudaMemcpyDeviceToHost ) );
    }

    void copy( CUdeviceptr data )
    {
        CUDA_CHECK( cudaMemcpy( _data, (void*)data, _size * sizeof( T ), cudaMemcpyDeviceToDevice ) );
    }

    void set( int value = 0 ) { CUDA_CHECK( cudaMemset( _data, value, _size * sizeof( T ) ) ); }

    void setSub( size_t size, size_t offset, int value = 0 )
    {
        assert( size + offset <= _alloc_size );
        CUDA_CHECK( cudaMemset( _data + offset, value, size * sizeof( T ) ) ); 
    }

  private:
    T*     _data = nullptr;
    size_t _size = 0;
    size_t _alloc_size = 0;
};

template <typename T>
std::ostream& operator<<( std::ostream& os, const CuBuffer<T>& buffer )
{
    std::vector<T> h( buffer.count() );
    if( buffer.download( h.data() ) )
        throw std::runtime_error( "cuda error" );

    for( size_t i = 0; i < h.size(); i++ )
    {
        os << "\t" << i << ": " << h[i] << std::endl;
    }

    return os;
}


template <typename T = unsigned char>
class CuBuffer2D
{
  public:
    typedef T value_type;

    // Allocate 2D device buffer with memory aligned rows (pitch).
    //
    // Parameters:
    //   width - width of the 2D buffer being allocated.
    //   height - height of the 2D buffer being allocated.
    //   data - a host buffer being copied into the new 2D buffer
    //   srcPitch - source pitch in bytes, default 0 implies pitch fitting (source data) width, i.e. width * sizeof(value_type).
    //
    CuBuffer2D( size_t width = 0, size_t height = 0, const T* data = nullptr, size_t srcPitch = 0 )
        : m_width( width )
        , m_height( height )
        , m_ptr( nullptr )
    {
        size_t pixels = width * height;
        if( pixels )
            CUDA_CHECK( cudaMallocPitch( &m_ptr, &m_pitch, width * sizeof( T ), height ) );
        if( data )
            CUDA_CHECK( cudaMemcpy2D( m_ptr, m_pitch, data, srcPitch, width * sizeof( T ), height, cudaMemcpyHostToDevice ) );
    }

    CuBuffer2D( const CuBuffer2D& ) = delete;

    CuBuffer2D( CuBuffer2D&& source ) noexcept { swap( source ); }

    CuBuffer2D( size_t width, size_t height, const T* data ) { allocAndUpload( width, height, data ); }

    ~CuBuffer2D() { CUDA_CHECK_NOTHROW( cudaFree( m_ptr ) ); }

    void upload( const T* data ) { upload( data, m_width * sizeof( T ) ); }

    void upload( const T* data, size_t sourcePitchInBytes )
    {
        CUDA_CHECK( cudaMemcpy2D( m_ptr, m_pitch, data, sourcePitchInBytes, m_width * sizeof( T ), m_height, cudaMemcpyHostToDevice ) );
    }

    CUdeviceptr get() const { return reinterpret_cast<CUdeviceptr>( m_ptr ); }

    size_t width() const { return m_width; }
    size_t height() const { return m_height; }
    size_t pitch() const { return m_pitch; }

    CuBuffer2D& operator=( const CuBuffer2D& source ) = delete;
    CuBuffer2D& operator=( CuBuffer2D&& source ) noexcept
    {
        swap( source );
        return *this;
    }

  private:
    void swap( CuBuffer2D& source )
    {
        std::swap( m_pitch, source.m_pitch );
        std::swap( m_width, source.m_width );
        std::swap( m_height, source.m_height );
        std::swap( m_ptr, source.m_ptr );
    }

    size_t m_pitch  = {};
    size_t m_width  = {};
    size_t m_height = {};
    T*     m_ptr    = {};
};
