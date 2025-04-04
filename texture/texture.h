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

#include "./textureCuda.h"
#include "./dxgi_formats.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

struct TextureOptions;


FormatMapping const* getPixelFormatMapping(DXGI_FORMAT format);

// Internal helper to hoist texture data from disk to device ;
// once a TextureCuda has been created from a Texture instance,
// that Texture instance can be safely deleted.

struct Texture
{
    using PixelFormat = DXGI_FORMAT;

    enum class AlphaMode : uint8_t
    {
        Unknown = 0,
        Straight = 1,
        Premultiplied = 2,
        Opaque = 3,
        Custom = 4,
    };

    enum class ResourceDimension : uint8_t {
        Unknown = 0,
        Buffer,
        Texture1D,
        Texture2D,
        Texture3D
    };

    std::unique_ptr<uint8_t[]>  m_data;
    std::vector<cudaPitchedPtr> m_mipmaps;  
                               
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_depth = 0;
    uint32_t m_size = 0;

    PixelFormat       m_pixelFormat = DXGI_FORMAT_UNKNOWN;
    FormatMapping     m_pixelFormatMapping = {};
    ResourceDimension m_dimension = ResourceDimension::Unknown;
    AlphaMode         m_alphaMode = AlphaMode::Unknown;                         
                              
    cudaTextureAddressMode m_addressMode = cudaTextureAddressMode::cudaAddressModeWrap;
                                                             
    Texture() = default;
    Texture( const Texture& c ) = delete;
    Texture& operator=( const Texture& c ) = delete;
    Texture& operator=( Texture&& c ) = delete;

    ~Texture() = default;

    bool isSRGB() const;

    void read( const std::filesystem::path& filePath, const TextureOptions& options );

    void upload( TextureCuda& tex, bool mipmap );
    
};

inline bool Texture::isSRGB() const
{
    return m_pixelFormatMapping.sRGB;
}
