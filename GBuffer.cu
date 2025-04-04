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


#include "GBuffer.h"
#include "GBuffer.cuh"

#include <glad/glad.h> // Include before gl_interop

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <OptiXToolkit/Gui/GLCheck.h>
#include <OptiXToolkit/ShaderUtil/color.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/Util/Exception.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <texture/textureCuda.h>

template <typename F>
static void map( ReadWriteResourceCuArray<F>& res )
{}

template <typename F>
static void map( ReadWriteResourceInterop<F>& res )
{
    OTK_REQUIRE( !res.m_mapped );

    cudaArray_t prevArray = res.m_array;

    CUDA_CHECK( cudaGraphicsMapResources( 1, &res.m_cudaGraphicsResource, /*stream*/ 0 ) );

    CUDA_CHECK( cudaGraphicsSubResourceGetMappedArray( &res.m_array, res.m_cudaGraphicsResource, 0, 0 ) );

    // Only re-create texture object if array changes
    if( prevArray != res.m_array )
    {
        CUDA_CHECK( createSurfobj( &res.m_surfaceObj, res.m_array ) );
        CUDA_CHECK( createTexobj( &res.m_pointSampleTexObj, res.m_array, nullptr, cudaTextureFilterMode::cudaFilterModePoint,
                                  cudaAddressModeClamp, cudaReadModeElementType, 0.f, /*srgb*/ false, /*normalized*/ false ) );
    }

    res.m_mapped = true;
}


template <typename F>
static void unmap( ReadWriteResourceCuArray<F>& res )
{}

template <typename F>
static void unmap( ReadWriteResourceInterop<F>& res )
{
    OTK_REQUIRE( res.m_mapped );
    CUDA_CHECK( cudaGraphicsUnmapResources( 1, &res.m_cudaGraphicsResource, /*stream*/ 0 ) );
    res.m_mapped = false;
    // Note: keeping cudaArray pointer around but it isn't valid to read from
}


template <typename F>
static void destroy( ReadWriteResourceCuArray<F>& res )
{
    if( res.m_surfaceObj )
    {
        CUDA_CHECK( cudaDestroySurfaceObject( res.m_surfaceObj ) );
        res.m_surfaceObj = 0;
    }
    if( res.m_pointSampleTexObj )
    {
        CUDA_CHECK( cudaDestroyTextureObject( res.m_pointSampleTexObj ) );
        res.m_pointSampleTexObj = 0;
    }
    if( res.m_array )
    {
        CUDA_CHECK( cudaFreeArray( res.m_array ) );
        res.m_array = 0;
    }
    res.m_size = { 0, 0 };
}

template <typename F>
static void destroy( ReadWriteResourceInterop<F>& res )
{
    if( res.m_surfaceObj )
    {
        CUDA_CHECK( cudaDestroySurfaceObject( res.m_surfaceObj ) );
        res.m_surfaceObj = 0;
    }
    if( res.m_pointSampleTexObj )
    {
        CUDA_CHECK( cudaDestroyTextureObject( res.m_pointSampleTexObj ) );
        res.m_pointSampleTexObj = 0;
    }

    if ( res.m_glTexId != 0 ) {
        GL_CHECK( glBindTexture( GL_TEXTURE_2D, 0 ) );
        GL_CHECK( glDeleteTextures( 1, &res.m_glTexId ) );
        res.m_glTexId = 0;
        if ( res.m_cudaGraphicsResource )
        {
            cudaGraphicsUnregisterResource( res.m_cudaGraphicsResource );
            res.m_cudaGraphicsResource = nullptr;
        }
    }

    res.m_array = 0;
    res.m_size = { 0, 0 };
}


template <typename F>
static void create( ReadWriteResourceCuArray<F>& res, uint2 size )
{
    OTK_REQUIRE( res.m_size.x == 0 && res.m_size.y == 0 );
    res.m_size = size;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<F>();
    CUDA_CHECK( cudaMallocArray( &res.m_array, &desc, size.x, size.y, 0 ) );
    CUDA_CHECK( createTexobj( &res.m_pointSampleTexObj, res.m_array, nullptr, cudaFilterModePoint, cudaAddressModeClamp,
                              cudaReadModeElementType, 0.f, /*srgb*/ false, /*normalized*/ false ) );
    CUDA_CHECK( createSurfobj( &res.m_surfaceObj, res.m_array ) );
}

template <typename F>
static void create( ReadWriteResourceInterop<F>& res, uint2 size )
{
    OTK_REQUIRE( res.m_size.x == 0 && res.m_size.y == 0 );
    res.m_size = size;

    // Gen GL buffer but don't map to array until map() is called
    GL_CHECK(glGenTextures(1, &res.m_glTexId));
    GL_CHECK(glBindTexture(GL_TEXTURE_2D, GLuint(res.m_glTexId)));
    if constexpr (std::is_same_v<F, float>)
    {
        GL_CHECK(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, GLsizei(size.x), GLsizei(size.y), 0, GL_RED, GL_FLOAT, nullptr));
    }
    else
    {
        OTK_ASSERT_FAIL_MSG("GL interop missing implementation");
    }
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
    GL_CHECK(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));

    GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0u));

    // Note: read/write access by default; could template this
    CUDA_CHECK(cudaGraphicsGLRegisterImage(&res.m_cudaGraphicsResource, res.m_glTexId, GL_TEXTURE_2D,
                                           cudaGraphicsRegisterFlagsNone));
}


GBuffer::GBuffer( uint2 rendersize, uint2 targetsize )
     : m_rendersize( rendersize ),
     m_targetsize( targetsize )
{
    ::create( m_albedo, rendersize );
    ::create( m_normals, rendersize );
    ::create( m_motionvecs, rendersize );
    ::create( m_color, rendersize );
    ::create( m_depth, rendersize );

    ::create( m_depthHires, targetsize );
    ::create( m_denoised, targetsize );
}

GBuffer::~GBuffer()
{
    std::apply([](auto&&... channel){(::destroy(channel), ...);}, channels());
}

void GBuffer::map()
{
    std::apply([](auto&&... channel){(::map(channel), ...);}, channels());
}

void GBuffer::unmap()
{
    std::apply([](auto&&... channel){(::unmap(channel), ...);}, channels());
}


__forceinline__ __device__ float3 tofloat3( float f )
{
    return make_float3( f, f, f );
}

__forceinline__ __device__ float3 tofloat3( float2 f )
{
    return make_float3( fabsf( f.x ), fabsf( f.y ), 0.f );
}

__forceinline__ __device__ float3 tofloat3( float3 f )
{
    return f;
}

__forceinline__ __device__ float3 tofloat3( float4 f )
{
    return make_float3( f.x, f.y, f.z );
}

struct sRGB_Operator { __device__ float3 operator()( float3 rgb ) { return toSRGB( rgb ); } };

template <typename T, typename... Operators> __global__
void blitKernel( T input, uchar4* output, uint2 outputSize, Operators ... operators )
{
    uint2 idx = {
        .x = blockIdx.x * blockDim.x + threadIdx.x,
        .y = blockIdx.y * blockDim.y + threadIdx.y,
    };

    if( ( idx.x >= outputSize.x ) || ( idx.y >= outputSize.y ) )
        return;

    float3 c = make_float3( .1f, .1f, .15f );

    uint2 inputSize = input.m_size;

    if( idx.x < inputSize.x && idx.y < inputSize.y )
    {
        c = tofloat3( gbuffer::read( input, idx ) );

        // using a fold expression with lambda to unpack the 'operators' variadic
        // arguments (applies each operator in the oreder it was given)
        ([&] {
            c = operators( c );
        } (), ...);
    }

    output[idx.y * outputSize.x + idx.x] =
        make_uchar4( quantizeUnsigned8Bits( c.x ), quantizeUnsigned8Bits( c.y ), quantizeUnsigned8Bits( c.z ), 255u );
}

void GBuffer::blit( Channel channel, uchar4* output, uint2 outputSize, CUstream stream )
{
    auto launchKernel = [&]<typename T, typename... Operators>( T& input, Operators&& ...operators ) {
        
        if( !input.isValid() )
            return;

        const int blockSize1D = 32;
        dim3 numBlocks( m_targetsize.x / blockSize1D + 1, m_targetsize.y / blockSize1D + 1, 1 );
        dim3 numThreadsPerBlock( 32, 32, 1 );

        blitKernel<<<numBlocks, numThreadsPerBlock>>>( input, output, outputSize, operators... );
    };

    switch( channel )
    {
        case Channel::ALBEDO:        launchKernel( m_albedo ); break;
        case Channel::NORMALS:       launchKernel( m_normals ); break;
        case Channel::MOTIONVECS:    launchKernel( m_motionvecs ); break;
        case Channel::DEPTH:         launchKernel( m_depth ); break;
        case Channel::DEPTH_HIRES:   launchKernel( m_depthHires ); break;

        case Channel::DENOISED:      launchKernel( m_denoised, sRGB_Operator{} ); break;
        case Channel::COLOR:         launchKernel( m_color, sRGB_Operator{} ); break;
    }
}

__global__
void depthKernel( RwFloatInterop depthBuf, uint2 pixel, float *result )
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx > 0 ) return;

    result[0] = gbuffer::read( depthBuf, pixel );
}

void GBuffer::pickdepth( uint2 pixel, float *d_out )
{
    ::map( m_depthHires );

    OTK_ASSERT( pixel.x < m_depthHires.m_size.x && pixel.y < m_depthHires.m_size.y );

    dim3 threadsPerBlock( 1, 1, 1 );
    depthKernel<<<1, threadsPerBlock>>>( m_depthHires, pixel, d_out );

    ::unmap( m_depthHires );
}


