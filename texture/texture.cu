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

// clang-format off

#include "./texture.h"
#include "./dds.h"
#include "./exr.h"
#include "./stb.h"
#include "./textureCache.h"
#include "./textureCuda.h"

#include <OptiXToolkit/Util/CuBuffer.h>
#include <OptiXToolkit/Util/Exception.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <cuda_fp16.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
// clang-format on

namespace fs = std::filesystem;

static inline uint8_t numChannels(cudaChannelFormatDesc desc)
{
    return (desc.x > 0) + (desc.y > 0) + (desc.z > 0) + (desc.w > 0);
}

static inline cudaTextureReadMode getCudaTextureReadMode(FormatMapping mapping)
{
    if (mapping.desc.f == cudaChannelFormatKindFloat ||
        mapping.desc.f == cudaChannelFormatKindSignedBlockCompressed6H)
        return cudaTextureReadMode::cudaReadModeElementType;

    return cudaTextureReadMode::cudaReadModeNormalizedFloat;
}

cudaPitchedPtr makeCudaPitchedPtr(const FormatMapping& mapping, void* ptr, uint32_t width, uint32_t height)
{
    assert(width > 0 && height > 0);

    size_t pitch = 0, xsize = width, ysize = height;

    if (mapping.bs > 0)
    {
        // all BC compressed formats use 4x4 pixel blocks
        xsize = std::max<size_t>(1u, (width + 3) / 4);
        ysize = std::max<size_t>(1u, (height + 3) / 4);
        pitch = xsize * mapping.bs;
    }
    else if (mapping.packed)
    {
        xsize = (width + 1) / 2;
        pitch = xsize * mapping.bpp;
    }
    else
        pitch = std::max(size_t(1), (xsize * mapping.bpp + 7) / 8);

    return { .ptr = ptr, .pitch = pitch, .xsize = xsize, .ysize = ysize };
}

// Cuda mipmapping based on Cuda SDK Samples
template <class T>
__global__ void mipmap_kernel( cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, int width, int height )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float px = 1.0 / float( width );
    float py = 1.0 / float( height );


    if( ( x < width ) && ( y < height ) )
    {
        // take the average of 4 samples
        // we are using the normalized access to make sure non-power-of-two
        // textures behave well when downsized.
        T color = ( tex2D<T>( mipInput, ( x + 0 ) * px, ( y + 0 ) * py ) )
                  + ( tex2D<T>( mipInput, ( x + 1 ) * px, ( y + 0 ) * py ) )
                  + ( tex2D<T>( mipInput, ( x + 1 ) * px, ( y + 1 ) * py ) )
                  + ( tex2D<T>( mipInput, ( x + 0 ) * px, ( y + 1 ) * py ) );
        color /= 4.0f;
        surf2Dwrite( color, mipOutput, x * sizeof( T ), y );
    }
}

void TexMipmap::operator()( cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, size_t numChannels, size_t width, size_t height )
{
    // run mipmap kernel
    dim3 blockSize( 16, 16, 1 );
    dim3 gridSize( ( width + blockSize.x - 1 ) / blockSize.x, ( height + blockSize.y - 1 ) / blockSize.y, 1 );
    if( numChannels == 4 )
        mipmap_kernel<float4><<<gridSize, blockSize>>>( mipOutput, mipInput, width, height );
    else if( numChannels == 2 )
        mipmap_kernel<float2><<<gridSize, blockSize>>>( mipOutput, mipInput, width, height );
    else
        mipmap_kernel<float><<<gridSize, blockSize>>>( mipOutput, mipInput, width, height );
    CUDA_SYNC_CHECK();
}

template <class T>
__global__ void flipVerticallyKernel( T* data, size_t w, size_t h )
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h / 2) return;
    const size_t index = y * w + x;
    std::swap( data[index], data[index + ( h - 2 * y - 1 ) * w] );
}

void flipVertically( Texture& tex )
{
    OTK_REQUIRE( tex.m_pixelFormatMapping.desc.f == cudaChannelFormatKindFloat );

    uint32_t width  = tex.m_width;
    uint32_t height = tex.m_height;

    uint8_t nchannels = numChannels( tex.m_pixelFormatMapping.desc );
    OTK_REQUIRE( nchannels == 4 || nchannels == 2 || nchannels == 1 );

    float* data = reinterpret_cast<float*>( tex.m_data.get());

    dim3 blockSize( 16, 16, 1 );
    dim3 gridSize( ( width + blockSize.x - 1 ) / blockSize.x, ( height / 2 + blockSize.y - 1 ) / blockSize.y, 1 );
    size_t numPixels = width * height;
    if( nchannels == 4 )
    {
        CuBuffer<float4> d( numPixels, (float4*)data );
        flipVerticallyKernel<float4><<<gridSize, blockSize>>>( d.data(), width, height );
        d.download( (float4*)data );
    }
    else if( nchannels == 2 )
    {
        CuBuffer<float2> d( numPixels, (float2*)data );
        flipVerticallyKernel<float2><<<gridSize, blockSize>>>( d.data(), width, height );
        d.download( (float2*)data );
    }
    else
    {
        CuBuffer<float> d( numPixels, (float*)data );
        flipVerticallyKernel<float><<<gridSize, blockSize>>>( d.data(), width, height );
        d.download( (float*)data );
    }
    CUDA_SYNC_CHECK();
}

template <int numChannels = 4>
__global__ void convertTexToSingleChannelKernel( const float* mcdata, float* scdata, size_t w, size_t h )
{
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= w || y >= h )
        return;
    const size_t index = y * w + x;
    scdata[index] = *( mcdata + numChannels * index );
}

template <int numChannels = 4>
void convertTexToSingleChannel( const float* multiChannel, float* singleChannel, size_t width, size_t height )
{
    dim3   blockSize( 16, 16, 1 );
    dim3   gridSize( ( width + blockSize.x - 1 ) / blockSize.x, ( height + blockSize.y - 1 ) / blockSize.y, 1 );
    size_t numPixels = width * height;

    CuBuffer<float> scdata( numPixels );
    CuBuffer<float> mcdata( numChannels * numPixels, multiChannel );
    convertTexToSingleChannelKernel<numChannels><<<gridSize, blockSize>>>( mcdata.data(), scdata.data(), width, height );
    scdata.download( singleChannel );
    CUDA_SYNC_CHECK();
}

void convertToSingleChannel( Texture& tex )
{
    OTK_REQUIRE( tex.m_pixelFormat == DXGI_FORMAT_R32G32B32A32_FLOAT
        || tex.m_pixelFormat == DXGI_FORMAT_R32G32_FLOAT );

    uint8_t nchannels = numChannels( tex.m_pixelFormatMapping.desc );

    const float* multiChannel = reinterpret_cast<float const*>( tex.m_data.get() );

    std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>( tex.m_width * tex.m_height * sizeof( float ) );

    float* singleChannel = reinterpret_cast<float*>( data.get() );

    if( nchannels == 2 )
        convertTexToSingleChannel<2>( multiChannel, singleChannel, tex.m_width, tex.m_height );
    else if( nchannels == 4 )
        convertTexToSingleChannel<4>( multiChannel, singleChannel, tex.m_width, tex.m_height );

    tex.m_pixelFormat = DXGI_FORMAT_R32_FLOAT;

    tex.m_pixelFormatMapping = *getPixelFormatMapping( tex.m_pixelFormat );

    tex.m_mipmaps[0] = makeCudaPitchedPtr( tex.m_pixelFormatMapping, data.get(), tex.m_width, tex.m_height );

    tex.m_data = std::move( data );
}

cudaError_t createTexobj( cudaTextureObject_t*         tex_object,
                          const cudaArray_t            cuda_array,
                          const cudaMipmappedArray_t   cuda_mipmap_array,
                          const cudaTextureFilterMode  filter_mode,
                          const cudaTextureAddressMode address_mode,
                          const cudaTextureReadMode    tex_read_mode,
                          const float                  maxMipLevel,
                          const bool                   srgb,
                          const bool                   normalized )
{
    OTK_REQUIRE( ( cuda_array == nullptr ) ^ ( cuda_mipmap_array == nullptr ) );

    cudaResourceDesc resDescr;
    memset( &resDescr, 0, sizeof( cudaResourceDesc ) );

    cudaTextureDesc texDescr;
    memset( &texDescr, 0, sizeof( cudaTextureDesc ) );

    if( cuda_array )
    {
        resDescr.resType         = cudaResourceTypeArray;
        resDescr.res.array.array = cuda_array;

        texDescr.borderColor[0] = 0.0f;
        texDescr.borderColor[1] = 0.0f;
        texDescr.borderColor[2] = 0.0f;
        texDescr.borderColor[3] = 0.0f;
        texDescr.maxAnisotropy  = 1;
        // Not using mipmaps.
        texDescr.mipmapFilterMode    = cudaFilterModePoint;
        texDescr.mipmapLevelBias     = 0.0f;
        texDescr.minMipmapLevelClamp = 0.0f;
        texDescr.maxMipmapLevelClamp = 0.0f;
    }
    else if( cuda_mipmap_array )
    {
        resDescr.resType           = cudaResourceTypeMipmappedArray;
        resDescr.res.mipmap.mipmap = cuda_mipmap_array;

        texDescr.mipmapFilterMode    = filter_mode;
        texDescr.maxMipmapLevelClamp = maxMipLevel;
        texDescr.maxAnisotropy       = 64;
    }

    texDescr.addressMode[0]      = address_mode;
    texDescr.addressMode[1]      = address_mode;
    texDescr.addressMode[2]      = address_mode;
    texDescr.filterMode          = filter_mode;
    texDescr.readMode            = tex_read_mode;
    texDescr.sRGB                = srgb;
    texDescr.normalizedCoords    = normalized;

    return cudaCreateTextureObject( tex_object, &resDescr, &texDescr, nullptr );
}

cudaError_t createSurfobj( cudaSurfaceObject_t* surface_object, cudaArray_t cuda_array )
{
    cudaResourceDesc surfRes;
    memset( &surfRes, 0, sizeof( cudaResourceDesc ) );
    surfRes.resType         = cudaResourceTypeArray;
    surfRes.res.array.array = cuda_array;
    return cudaCreateSurfaceObject( surface_object, &surfRes );
}

static void uploadCudaTexture( const Texture& tex, TextureCuda& ctex )
{
    uint32_t width  = tex.m_width;
    uint32_t height = tex.m_height;

    cudaArray_t cuArray{ 0 };
    CUDA_CHECK( cudaMallocArray( &cuArray, &tex.m_pixelFormatMapping.desc, width, height ) );

    if( !tex.m_mipmaps.empty() )
    {
        cudaMemcpy3DParms copyParams;
        memset( &copyParams, 0, sizeof( cudaMemcpy3DParms ) );
        copyParams.srcPtr = tex.m_mipmaps[0];
        copyParams.extent = make_cudaExtent( width, height, 1 );
        copyParams.kind = cudaMemcpyHostToDevice;
        copyParams.dstArray = cuArray;
        CUDA_CHECK( cudaMemcpy3D( &copyParams) );
    }

    cudaMipmappedArray_t cuMipmappedArray = nullptr;
    constexpr bool normalizedUv = true;
    cudaTextureReadMode readMode = getCudaTextureReadMode( tex.m_pixelFormatMapping );

    cudaTextureObject_t texref;
    CUDA_CHECK( createTexobj( &texref, cuArray, cuMipmappedArray, cudaTextureFilterMode::cudaFilterModeLinear,
                              tex.m_addressMode, readMode, /*maxMipLevel*/ 0, tex.isSRGB(), normalizedUv ) );

    ctex.initialize( width, height, texref, cuArray, nullptr );

    ctex.upload();
}

static void uploadMipmaps( const Texture& tex, cudaMipmappedArray_t& cuMipmappedArray )
{
    uint32_t width  = tex.m_width;
    uint32_t height = tex.m_height;
    uint32_t nlevels = uint32_t( std::log2( std::max( width, height ) ) + 1 );

    for (int level = 0; level < nlevels; level++)
    {
        cudaArray_t levelData = 0;
        CUDA_CHECK( cudaGetMipmappedArrayLevel( &levelData, cuMipmappedArray, level ) );

        cudaMemcpy3DParms copyParams;
        memset( &copyParams, 0, sizeof( cudaMemcpy3DParms ) );
        copyParams.srcPtr = tex.m_mipmaps[level];
        copyParams.extent = make_cudaExtent( width, height, 1 );
        copyParams.kind = cudaMemcpyHostToDevice;
        copyParams.dstArray = levelData;
        CUDA_CHECK( cudaMemcpy3D( &copyParams ) );

        width = std::max( 1u, width / 2u );
        height = std::max( 1u, height / 2u );
    }
}

template <typename MIPMAP_FUNCTION>
inline void generateMipmaps( const uint32_t                    texWidth,
                             const uint32_t                    texHeight,
                             const std::unique_ptr<uint8_t[]>& texData,
                             const FormatMapping               pixelFmtMapping,
                             const cudaTextureAddressMode      addressMode,
                             cudaMipmappedArray_t&             cuMipmappedArray )
{
    uint32_t    width      = texWidth;
    uint32_t    height     = texHeight;
    uint8_t     nchannels  = numChannels( pixelFmtMapping.desc );
    cudaArray_t levelFirst = 0;
    CUDA_CHECK( cudaGetMipmappedArrayLevel( &levelFirst, cuMipmappedArray, 0 ) );

    if( texData )
    {
        cudaMemcpy3DParms copyParams;
        memset( &copyParams, 0, sizeof( cudaMemcpy3DParms ) );
        copyParams.srcPtr   = makeCudaPitchedPtr( pixelFmtMapping, texData.get(), width, height );
        copyParams.extent   = make_cudaExtent( width, height, 1 );
        copyParams.kind     = cudaMemcpyHostToDevice;
        copyParams.dstArray = levelFirst;
        CUDA_CHECK( cudaMemcpy3D( &copyParams ) );
    }

    unsigned level = 0;
    while( width != 1 || height != 1 )
    {
        width  = std::max( 1u, width / 2 );
        height = std::max( 1u, height / 2 );

        cudaArray_t levelFrom;
        CUDA_CHECK( cudaGetMipmappedArrayLevel( &levelFrom, cuMipmappedArray, level ) );
        cudaArray_t levelTo;
        CUDA_CHECK( cudaGetMipmappedArrayLevel( &levelTo, cuMipmappedArray, level + 1 ) );
        cudaExtent levelToSize;
        CUDA_CHECK( cudaArrayGetInfo( NULL, &levelToSize, NULL, levelTo ) );
        // generate texture object for reading
        cudaTextureObject_t texInput;
        cudaResourceDesc    texRes;
        memset( &texRes, 0, sizeof( cudaResourceDesc ) );
        texRes.resType         = cudaResourceTypeArray;
        texRes.res.array.array = levelFrom;
        cudaTextureDesc texDescr;
        memset( &texDescr, 0, sizeof( cudaTextureDesc ) );

        texDescr.normalizedCoords = true;
        texDescr.filterMode       = cudaTextureFilterMode::cudaFilterModeLinear;
        texDescr.addressMode[0]   = addressMode;
        texDescr.addressMode[1]   = addressMode;
        texDescr.addressMode[2]   = addressMode;
        texDescr.readMode         = cudaReadModeElementType;
        CUDA_CHECK( cudaCreateTextureObject( &texInput, &texRes, &texDescr, NULL ) );
        // generate surface object for writing
        cudaSurfaceObject_t surfOutput;
        cudaResourceDesc    surfRes;
        memset( &surfRes, 0, sizeof( cudaResourceDesc ) );
        surfRes.resType         = cudaResourceTypeArray;
        surfRes.res.array.array = levelTo;
        CUDA_CHECK( cudaCreateSurfaceObject( &surfOutput, &surfRes ) );

        MIPMAP_FUNCTION()( surfOutput, texInput, nchannels, width, height );

        CUDA_CHECK( cudaDeviceSynchronize() );
        CUDA_CHECK( cudaGetLastError() );
        CUDA_CHECK( cudaDestroySurfaceObject( surfOutput ) );
        CUDA_CHECK( cudaDestroyTextureObject( texInput ) );
        level++;
    }
}

template <typename MIPMAP_FUNCTION>
static void generateMipmaps( const Texture& tex, cudaMipmappedArray_t& cuMipmappedArray )
{
    generateMipmaps<MIPMAP_FUNCTION>( tex.m_width, tex.m_height, tex.m_data, tex.m_pixelFormatMapping,
                                      tex.m_addressMode, cuMipmappedArray );
}

static void uploadCudaTextureMipmapped( const Texture& tex, TextureCuda& ctex )
{
    cudaMipmappedArray_t  cuMipmappedArray = nullptr;

    float levelsf = std::log2( std::max( tex.m_width, tex.m_height ) );
    uint32_t levels = uint32_t( levelsf + 1 );

    CUDA_CHECK( cudaMallocMipmappedArray( &cuMipmappedArray,
        &tex.m_pixelFormatMapping.desc, make_cudaExtent( tex.m_width, tex.m_height, 0 ), levels ) );

    if( tex.m_mipmaps.size() > 1 )
        uploadMipmaps( tex, cuMipmappedArray );
    else
        generateMipmaps<TexMipmap>( tex, cuMipmappedArray );


    cudaArray_t         cuArray      = nullptr;
    constexpr bool      normalizedUv = true;
    cudaTextureReadMode readMode     = getCudaTextureReadMode( tex.m_pixelFormatMapping );

    cudaTextureObject_t texref;
    CUDA_CHECK( createTexobj( &texref, cuArray, cuMipmappedArray, cudaTextureFilterMode::cudaFilterModeLinear,
                              tex.m_addressMode, readMode, levelsf, tex.isSRGB(), normalizedUv ) );

    ctex.initialize( tex.m_width, tex.m_height, texref, cuArray, cuMipmappedArray );

    ctex.upload();
}

void Texture::read( const fs::path& filepath, const TextureOptions& options )
{
    const bool vflip = options.vflip;

    std::string fp = filepath.lexically_normal().generic_string();

    if( fp.empty() )
    {
        std::string cwd = fs::current_path().lexically_normal().generic_string();
        throw std::runtime_error("Failed to open " + fp + "\n CWD " + cwd + "\n");
    }

    if( filepath.extension() == ".exr" )
        readEXRFile(*this, fp, options.vflip, options.convertToSingleChannel);
    else if( filepath.extension() == ".dds" )
        readDDSFile( *this, fp );
    else
        readTexFile( *this, fp, vflip );
}

void Texture::upload( TextureCuda& tex, bool mipmap )
{
    bool hasMips = mipmap && m_mipmaps.size() > 1;

    if( hasMips )
        uploadCudaTextureMipmapped( *this, tex );
    else
        uploadCudaTexture( *this, tex );
}

TextureCuda::TextureCuda( TextureCuda&& c ) noexcept
{
    if( this != &c )
    {
        m_dim            = c.m_dim;
        m_texref         = c.m_texref;
        m_array          = c.m_array;
        m_mipmappedArray = c.m_mipmappedArray;
        m_dtex           = c.m_dtex;

        c.m_dim            = {0,0};
        c.m_texref         = 0;
        c.m_array          = 0;
        c.m_mipmappedArray = 0;
        c.m_dtex           = 0;
    }
}

TextureCuda::~TextureCuda() noexcept
{
    if( m_texref != 0 )
    {
        CUDA_CHECK_NOTHROW( cudaDestroyTextureObject( m_texref ) );
        m_texref = 0;
    }

    if( m_array )
    {
        CUDA_CHECK_NOTHROW( cudaFreeArray( m_array ) );
        m_array = 0;
    }

    if( m_mipmappedArray )
    {
        CUDA_CHECK_NOTHROW( cudaFreeMipmappedArray( m_mipmappedArray ) );
        m_mipmappedArray = 0;
    }

    if( m_dtex )
    {
        CUDA_CHECK_NOTHROW( cudaFree( (void*)m_dtex ) );
        m_dtex = 0;
    }
}

void TextureCuda::upload() const
{
    if( !m_dtex )
        CUDA_CHECK( cudaMalloc( (void**)&m_dtex, sizeof( TextureCuda ) ) );
    CUDA_CHECK( cudaMemcpy( (void**)m_dtex, this, sizeof( TextureCuda ), cudaMemcpyHostToDevice ) );
}
