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

#include "./texture.h"

#include "./dxgi_formats.h"

#include <exception>
#include <filesystem>
#include <fstream>

cudaPitchedPtr makeCudaPitchedPtr( const FormatMapping& mapping, void* ptr, uint32_t width, uint32_t height);

static std::pair<std::unique_ptr<uint8_t []>, size_t> readFile(const std::string& filepath)
{
    std::ifstream filestream( filepath, std::ios::binary | std::ios::in );
    if( !filestream.is_open() )
        return {};

    // Read the file into a vector.
    filestream.seekg( 0, std::ios::end );
    size_t size = filestream.tellg();

    auto data = std::make_unique<uint8_t[]>( size );
    
    filestream.seekg( 0 );
    filestream.read( (char*)data.get(), size );
    filestream.close();
    
    return { std::move(data), size };
}

inline constexpr uint32_t MAKEFOURCC(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3)
{
    return c0 | (c1 << 8) | (c2 << 16) | (c3 << 24);
}

enum class DdsMagicNumber : uint32_t {
    DDS  = MAKEFOURCC('D', 'D', 'S', ' '),
    DXT1 = MAKEFOURCC('D', 'X', 'T', '1'), // BC1_UNORM
    DXT2 = MAKEFOURCC('D', 'X', 'T', '2'), // BC2_UNORM
    DXT3 = MAKEFOURCC('D', 'X', 'T', '3'), // BC2_UNORM
    DXT4 = MAKEFOURCC('D', 'X', 'T', '4'), // BC3_UNORM
    DXT5 = MAKEFOURCC('D', 'X', 'T', '5'), // BC3_UNORM
    ATI1 = MAKEFOURCC('A', 'T', 'I', '1'), // BC4_UNORM
    BC4U = MAKEFOURCC('B', 'C', '4', 'U'), // BC4_UNORM
    BC4S = MAKEFOURCC('B', 'C', '4', 'S'), // BC4_SNORM
    ATI2 = MAKEFOURCC('A', 'T', 'I', '2'), // BC5_UNORM
    BC5U = MAKEFOURCC('B', 'C', '5', 'U'), // BC5_UNORM
    BC5S = MAKEFOURCC('B', 'C', '5', 'S'), // BC5_SNORM
    RGBG = MAKEFOURCC('R', 'G', 'B', 'G'), // R8G8_B8G8_UNORM
    GRBG = MAKEFOURCC('G', 'R', 'B', 'G'), // G8R8_G8B8_UNORM
    YUY2 = MAKEFOURCC('Y', 'U', 'Y', '2'), // YUY2
    UYVY = MAKEFOURCC('U', 'Y', 'V', 'Y'),
    DX10 = MAKEFOURCC('D', 'X', '1', '0'), // Any DXGI format
};

enum class PixelFormatFlags : uint32_t {
    AlphaPixels = 0x1,
    Alpha = 0x2,
    FourCC = 0x4,
    PAL8 = 0x20,
    RGB = 0x40,
    RGBA = RGB | AlphaPixels,
    YUV = 0x200,
    Luminance = 0x20000,
    LuminanceA = Luminance | AlphaPixels,
    BumpDuDv = 0x80000
};

inline PixelFormatFlags operator&(PixelFormatFlags a, PixelFormatFlags b) {
    return PixelFormatFlags( uint32_t(a) | uint32_t(b) );
}

struct FilePixelFormat {
    uint32_t size;
    PixelFormatFlags flags;
    uint32_t fourCC;
    uint32_t bitCount;
    uint32_t rBitMask;
    uint32_t gBitMask;
    uint32_t bBitMask;
    uint32_t aBitMask;
};

enum class HeaderFlags : uint32_t {
    Caps = 0x1,
    Height = 0x2,
    Width = 0x4,
    Pitch = 0x8,
    PixelFormat = 0x1000,
    Texture = Caps | Height | Width | PixelFormat,
    Mipmap = 0x20000,
    Volume = 0x800000,
    LinearSize = 0x00080000,
};

// Subset here matches D3D10_RESOURCE_MISC_FLAG and D3D11_RESOURCE_MISC_FLAG
enum DDS_RESOURCE_MISC_FLAG
{
    DDS_RESOURCE_MISC_TEXTURECUBE = 0x4L,
};

enum DDS_MISC_FLAGS2
{
    DDS_MISC_FLAGS2_ALPHA_MODE_MASK = 0x7L,
};

enum Caps2Flags : uint32_t {
    Cubemap = 0x200,
};

struct DDSHeader {
    uint32_t size;
    HeaderFlags flags;
    uint32_t height;
    uint32_t width;
    uint32_t pitch;
    uint32_t depth;
    uint32_t mipmapCount;
    uint32_t reserved[11];
    FilePixelFormat pixelFormat;
    uint32_t caps1;
    uint32_t caps2;
    uint32_t caps3;
    uint32_t caps4;
    uint32_t reserved2;
};

static_assert( sizeof( DDSHeader ) == 124 );

/** An additional header for DX10 */
struct Dx10Header {
    DXGI_FORMAT dxgiFormat;
    Texture::ResourceDimension resourceDimension;
    uint32_t miscFlags;
    uint32_t arraySize;
    uint32_t miscFlags2;
};

static Texture::AlphaMode getAlphaMode( const DDSHeader& header )
{
    using enum Texture::AlphaMode;

    if( uint32_t( header.pixelFormat.flags ) & uint32_t( PixelFormatFlags::FourCC ) )
    {
        if( MAKEFOURCC( 'D', 'X', '1', '0' ) == uint32_t( header.pixelFormat.flags ) )
        {
            auto d3d10ext = reinterpret_cast<const Dx10Header*>(
                reinterpret_cast<const uint8_t*>(&header) + sizeof(Dx10Header));

            auto mode = Texture::AlphaMode(d3d10ext->miscFlags2 & DDS_MISC_FLAGS2_ALPHA_MODE_MASK);

            switch( mode )
            {
                case Straight:
                case Premultiplied:
                case Opaque:
                case Custom:
                    return mode;
            }
        }
        else if( (MAKEFOURCC('D', 'X', 'T', '2') == header.pixelFormat.fourCC)
            || (MAKEFOURCC('D', 'X', 'T', '4') == header.pixelFormat.fourCC) )
            return Premultiplied;
    }
    return Unknown;
}

template <typename T>
static inline constexpr bool hasBit( T value, T bit )
{
    return ( value & bit ) == bit;
}

static DXGI_FORMAT convertPixelFormat(const FilePixelFormat& ddpf, bool forceSRGB)
{
    auto ISBITMASK = [&ddpf](uint32_t r, uint32_t g, uint32_t b, uint32_t a) -> bool {
        return ddpf.rBitMask == r && ddpf.gBitMask == g && ddpf.bBitMask == b && ddpf.aBitMask == a;
    };

    if (uint32_t(ddpf.flags) & uint32_t(PixelFormatFlags::RGB))
    {
        // Note that sRGB formats are written using the "DX10" extended header

        switch (ddpf.bitCount)
        {
        case 32:
            if (ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))
                return forceSRGB ? DXGI_FORMAT_R8G8B8A8_UNORM : DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;

            if (ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000))
                return forceSRGB ? DXGI_FORMAT_B8G8R8A8_UNORM : DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;

            if (ISBITMASK(0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000))
                return forceSRGB ? DXGI_FORMAT_B8G8R8X8_UNORM : DXGI_FORMAT_B8G8R8X8_UNORM_SRGB;

            // No DXGI format maps to ISBITMASK(0x000000ff,0x0000ff00,0x00ff0000,0x00000000) aka D3DFMT_X8B8G8R8

            // Note that many common DDS reader/writers (including D3DX) swap the
            // the RED/BLUE masks for 10:10:10:2 formats. We assume
            // below that the 'backwards' header mask is being used since it is most
            // likely written by D3DX. The more robust solution is to use the 'DX10'
            // header extension and specify the DXGI_FORMAT_R10G10B10A2_UNORM format directly

            // For 'correct' writers, this should be 0x000003ff,0x000ffc00,0x3ff00000 for RGB data
            if (ISBITMASK(0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000))
                return DXGI_FORMAT_R10G10B10A2_UNORM;

            // No DXGI format maps to ISBITMASK(0x000003ff,0x000ffc00,0x3ff00000,0xc0000000) aka D3DFMT_A2R10G10B10
            if (ISBITMASK(0x0000ffff, 0xffff0000, 0x00000000, 0x00000000))
                return DXGI_FORMAT_R16G16_UNORM;

            if (ISBITMASK(0xffffffff, 0x00000000, 0x00000000, 0x00000000))
                // Only 32-bit color channel format in D3D9 was R32F
                return DXGI_FORMAT_R32_FLOAT;
            break;

        case 24:
            // No 24bpp DXGI formats aka D3DFMT_R8G8B8
            break;

        case 16:
            if (ISBITMASK(0x7c00, 0x03e0, 0x001f, 0x8000))
                return DXGI_FORMAT_B5G5R5A1_UNORM;

            if (ISBITMASK(0xf800, 0x07e0, 0x001f, 0x0000))
                return DXGI_FORMAT_B5G6R5_UNORM;

            // No DXGI format maps to ISBITMASK(0x7c00,0x03e0,0x001f,0x0000) aka D3DFMT_X1R5G5B5
            if (ISBITMASK(0x0f00, 0x00f0, 0x000f, 0xf000))
                return DXGI_FORMAT_B4G4R4A4_UNORM;

            // No DXGI format maps to ISBITMASK(0x0f00,0x00f0,0x000f,0x0000) aka D3DFMT_X4R4G4B4

            // No 3:3:2, 3:3:2:8, or paletted DXGI formats aka D3DFMT_A8R3G3B2, D3DFMT_R3G3B2, D3DFMT_P8, D3DFMT_A8P8, etc.
            break;
        }
    }
    else if (uint32_t(ddpf.flags) & uint32_t(PixelFormatFlags::Luminance))
    {
        if (8 == ddpf.bitCount)
        {
            if (ISBITMASK(0x000000ff, 0x00000000, 0x00000000, 0x00000000))
                return DXGI_FORMAT_R8_UNORM;

            // No DXGI format maps to ISBITMASK(0x0f,0x00,0x00,0xf0) aka D3DFMT_A4L4

            if (ISBITMASK(0x000000ff, 0x00000000, 0x00000000, 0x0000ff00))
                return DXGI_FORMAT_R8G8_UNORM;
        }

        if (16 == ddpf.bitCount)
        {
            if (ISBITMASK(0x0000ffff, 0x00000000, 0x00000000, 0x00000000))
                return DXGI_FORMAT_R16_UNORM;

            if (ISBITMASK(0x000000ff, 0x00000000, 0x00000000, 0x0000ff00))
                return DXGI_FORMAT_R8G8_UNORM;
        }
    }
    else if (uint32_t(ddpf.flags) & uint32_t(PixelFormatFlags::Alpha))
    {
        if (8 == ddpf.bitCount)
            return DXGI_FORMAT_R8_UNORM; // we don't support A8 in NVRHI
    }
    else if (uint32_t(ddpf.flags) & uint32_t(PixelFormatFlags::BumpDuDv))
    {
        if (16 == ddpf.bitCount)
        {
            if (ISBITMASK(0x00ff, 0xff00, 0x0000, 0x0000))
                return DXGI_FORMAT_R8G8_SNORM;
        }

        if (32 == ddpf.bitCount)
        {
            if (ISBITMASK(0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))
                return DXGI_FORMAT_R8G8B8A8_SNORM;
            if (ISBITMASK(0x0000ffff, 0xffff0000, 0x00000000, 0x00000000))
                return DXGI_FORMAT_R16G16_SNORM;

            // No DXGI format maps to ISBITMASK(0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000) aka D3DFMT_A2W10V10U10
        }
    }
    else if (uint32_t(ddpf.flags) & uint32_t(PixelFormatFlags::FourCC))
    {
        if (MAKEFOURCC('D', 'X', 'T', '1') == ddpf.fourCC)
            return forceSRGB ? DXGI_FORMAT_BC1_UNORM_SRGB : DXGI_FORMAT_BC1_UNORM;

        if (MAKEFOURCC('D', 'X', 'T', '3') == ddpf.fourCC)
            return forceSRGB ? DXGI_FORMAT_BC2_UNORM_SRGB : DXGI_FORMAT_BC2_UNORM;

        if (MAKEFOURCC('D', 'X', 'T', '5') == ddpf.fourCC)
            return forceSRGB ? DXGI_FORMAT_BC3_UNORM_SRGB : DXGI_FORMAT_BC3_UNORM;

        // While pre-multiplied alpha isn't directly supported by the DXGI formats,
        // they are basically the same as these BC formats so they can be mapped
        if (MAKEFOURCC('D', 'X', 'T', '2') == ddpf.fourCC)
            return DXGI_FORMAT_BC2_UNORM;

        if (MAKEFOURCC('D', 'X', 'T', '4') == ddpf.fourCC)
            return DXGI_FORMAT_BC3_UNORM;

        if (MAKEFOURCC('A', 'T', 'I', '1') == ddpf.fourCC)
            return DXGI_FORMAT_BC4_UNORM;

        if (MAKEFOURCC('B', 'C', '4', 'U') == ddpf.fourCC)
            return DXGI_FORMAT_BC4_UNORM;

        if (MAKEFOURCC('B', 'C', '4', 'S') == ddpf.fourCC)
            return DXGI_FORMAT_BC4_SNORM;

        if (MAKEFOURCC('A', 'T', 'I', '2') == ddpf.fourCC)
            return DXGI_FORMAT_BC5_UNORM;

        if (MAKEFOURCC('B', 'C', '5', 'U') == ddpf.fourCC)
            return DXGI_FORMAT_BC5_UNORM;

        if (MAKEFOURCC('B', 'C', '5', 'S') == ddpf.fourCC)
            return DXGI_FORMAT_BC5_SNORM;


        // Check for D3DFORMAT enums being set here
        switch (ddpf.fourCC)
        {
        case 36: // D3DFMT_A16B16G16R16
            return DXGI_FORMAT_R16G16B16A16_UNORM;

        case 110: // D3DFMT_Q16W16V16U16
            return DXGI_FORMAT_R16G16B16A16_SNORM;

        case 111: // D3DFMT_R16F
            return DXGI_FORMAT_R16_FLOAT;

        case 112: // D3DFMT_G16R16F
            return DXGI_FORMAT_R16G16_FLOAT;

        case 113: // D3DFMT_A16B16G16R16F
            return DXGI_FORMAT_R16G16B16A16_FLOAT;

        case 114: // D3DFMT_R32F
            return DXGI_FORMAT_R32_FLOAT;

        case 115: // D3DFMT_G32R32F
            return DXGI_FORMAT_R32G32_FLOAT;

        case 116: // D3DFMT_A32B32G32R32F
            return DXGI_FORMAT_R32G32B32A32_FLOAT;
        }
    }
    return DXGI_FORMAT_UNKNOWN;
}

void readDDSFile( Texture& tex, const std::string& filepath )
{
    auto [data, size] = readFile(filepath);

    if( !data || size == 0 )
        throw std::runtime_error( "error reading '" + filepath + "'" );

    uint8_t* ptr = data.get();

    DdsMagicNumber ddsMagic = *reinterpret_cast<DdsMagicNumber*>( ptr );
    ptr += sizeof( uint32_t );

    if( ddsMagic != DdsMagicNumber::DDS )
        throw std::runtime_error( "invalid DDS MAGIC '" + filepath + "'" );

    if( ( ptr + sizeof( DDSHeader ) ) >= ( data.get() + size ) )
        throw std::runtime_error( "invalid DDS header size '" + filepath + "'" );

    const auto* header = reinterpret_cast<const DDSHeader*>( ptr );
    ptr += sizeof( DDSHeader );

    if( hasBit( header->pixelFormat.flags, PixelFormatFlags::FourCC ) &&
        hasBit( header->pixelFormat.fourCC, uint32_t( DdsMagicNumber::DX10 ) ) )
    {
        if( ( ptr + sizeof( Dx10Header ) ) >= ( data.get() + size ) )
            throw std::runtime_error("invalid DX10 DDS header size '" + filepath + "'");

        const Dx10Header* d3d10Header = reinterpret_cast<const Dx10Header*>( ptr );
        ptr += sizeof( Dx10Header );

        if( d3d10Header->arraySize > 1 )
            throw std::runtime_error("DDS arrays not supported '" + filepath + "'");

        tex.m_pixelFormat = d3d10Header->dxgiFormat;
        tex.m_dimension   = d3d10Header->resourceDimension;
    }
    else
    {
        tex.m_pixelFormat = convertPixelFormat( header->pixelFormat, false );

        using enum Texture::ResourceDimension;

        if( uint32_t(header->flags) & uint32_t(HeaderFlags::Volume) || header->caps2 & Caps2Flags::Cubemap )
            tex.m_dimension = Texture3D;
        else
            tex.m_dimension = header->height > 1 ? Texture2D : Texture1D;
    }

    if( tex.m_pixelFormat == DXGI_FORMAT_UNKNOWN )
        throw std::runtime_error("unknown pixel format '" + filepath + "'");

    const FormatMapping* mapping = getPixelFormatMapping( tex.m_pixelFormat );
    if( !mapping )
        throw std::runtime_error("unsupported pixel format '" + filepath + "'");

    uint32_t nmipmaps = std::max( header->mipmapCount, 1u );   
    uint32_t width    = header->width;
    uint32_t height   = header->height;

    tex.m_mipmaps.resize(nmipmaps);

    for( uint32_t i = 0; i < nmipmaps && width != 0; ++i )
    {
        cudaPitchedPtr& mip = tex.m_mipmaps[i];
        
        mip = makeCudaPitchedPtr( *mapping, ptr, width, height );

        ptr += mip.pitch * mip.ysize;

        width  = std::max( width  / 2u, 1u );
        height = std::max( height / 2u, 1u );
    }

    tex.m_width     = header->width;
    tex.m_height    = header->height;
    tex.m_pixelFormatMapping = *mapping;
    tex.m_alphaMode = getAlphaMode(*header);

    tex.m_size      = size;
    tex.m_data      = std::move(data);
}

