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

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/Util/Exception.h>

#include <cassert>
#include <memory>

class TextureCuda
{
public:
    
    void initialize( uint32_t width, uint32_t height, cudaTextureObject_t texref, cudaArray_t cuArray, cudaMipmappedArray_t cuMipmappedArray )
    {
        OTK_ASSERT( !m_array && !m_mipmappedArray && !m_texref );
        m_dim = make_float2( float(width), float(height) );
        m_texref         = texref;
        m_array          = cuArray;
        m_mipmappedArray = cuMipmappedArray;
    }

    TextureCuda( const TextureCuda& c ) = delete;
    TextureCuda& operator=( const TextureCuda& c ) = delete;
    TextureCuda& operator=( TextureCuda&& c ) = delete;
    TextureCuda( TextureCuda&& c ) noexcept;

    TextureCuda() = default;
    ~TextureCuda() noexcept;

    const TextureCuda* getDevicePtr() const { return m_dtex; }

    void upload() const;

    __device__ __host__ cudaTextureObject_t getTexref() const { return m_texref; }
    __device__ __host__ float2 getDim() const { return m_dim; }


private:
    float2 m_dim = {0.f, 0.f};
    cudaArray_t m_array = nullptr;
    cudaMipmappedArray_t m_mipmappedArray = nullptr;
    cudaTextureObject_t m_texref{0};

    const TextureCuda* m_dtex = nullptr;
};

cudaError_t createSurfobj( cudaSurfaceObject_t* surface_object, cudaArray_t cuda_array );
cudaError_t createTexobj( cudaTextureObject_t*         tex_object,
                                 const cudaArray_t            cuda_array,
                                 const cudaMipmappedArray_t   cuda_mipmap_array,
                                 const cudaTextureFilterMode  filter_mode,
                                 const cudaTextureAddressMode address_mode,
                                 const cudaTextureReadMode    tex_read_mode,
                                 const float                  maxMipLevel,
                                 const bool                   srgb,
                                 const bool                   normalized );

// Full descriptor of Cuda pixel formats
struct FormatMapping
{
    cudaChannelFormatDesc desc = { .x = 0, .y = 0, .z = 0, .w = 0 };
    uint8_t bpp        = 0; // bits per pixel
    uint8_t bs         = 0; // block size (in bits)
    uint8_t packed : 1 = false;
    uint8_t sRGB   : 1 = false;
};
                                
template <typename MIPMAP_FUNCTION>
void generateMipmaps( const uint32_t                    texWidth,
                             const uint32_t                    texHeight,
                             const std::unique_ptr<uint8_t[]>& texData,
                             const FormatMapping               pixelFmtMapping,
                             const cudaTextureAddressMode      addressMode,
                             cudaMipmappedArray_t&             cuMipmappedArray );

struct TexMipmap
{
    void operator()( cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, size_t numChannels, size_t width, size_t height );
};
