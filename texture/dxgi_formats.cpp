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

#include "./dxgi_formats.h"
#include "./texture.h"

#include <cuda_runtime.h>

#include <map>

static std::map<DXGI_FORMAT, FormatMapping> mappings = {

    { DXGI_FORMAT_R32G32B32A32_TYPELESS, { .desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindNone), .bpp = 128 } },
    { DXGI_FORMAT_R32G32B32A32_TYPELESS, { .desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindNone), .bpp=128 } },
    { DXGI_FORMAT_R32G32B32A32_FLOAT,    { .desc = cudaCreateChannelDesc<float4>(), .bpp=128 } },
    { DXGI_FORMAT_R32G32B32A32_UINT,     { .desc = cudaCreateChannelDesc<uint4>(), .bpp=128 } },
    { DXGI_FORMAT_R32G32B32A32_SINT,     { .desc = cudaCreateChannelDesc<int4>(), .bpp=128 } },

    { DXGI_FORMAT_R32G32B32_TYPELESS,    { .desc = cudaCreateChannelDesc(32, 32, 32, 0, cudaChannelFormatKindNone), .bpp=96 } },
    { DXGI_FORMAT_R32G32B32_FLOAT,       { .desc = cudaCreateChannelDesc<float3>(), .bpp=96 } },
    { DXGI_FORMAT_R32G32B32_UINT,        { .desc = cudaCreateChannelDesc<uint3>(), .bpp=96 } },
    { DXGI_FORMAT_R32G32B32_SINT,        { .desc = cudaCreateChannelDesc<int3>(), .bpp=96 } },

    { DXGI_FORMAT_R16G16B16A16_TYPELESS, { .desc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindNone), .bpp=64 } },
    { DXGI_FORMAT_R16G16B16A16_FLOAT,    { .desc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat), .bpp=64 } },
    { DXGI_FORMAT_R16G16B16A16_UNORM,    { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized16X4>(), .bpp=64 } },
    { DXGI_FORMAT_R16G16B16A16_UINT,     { .desc = cudaCreateChannelDesc<ushort4>(), .bpp=64 } },
    { DXGI_FORMAT_R16G16B16A16_SNORM,    { .desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized16X4>(), .bpp=64 } },
    { DXGI_FORMAT_R16G16B16A16_SINT,     { .desc = cudaCreateChannelDesc<ushort4>(), .bpp=64 } },

    { DXGI_FORMAT_R32G32_TYPELESS,       { .desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindNone), .bpp=64 } },
    { DXGI_FORMAT_R32G32_FLOAT,          { .desc = cudaCreateChannelDesc<float2>(), .bpp=64 } },
    { DXGI_FORMAT_R32G32_UINT,           { .desc = cudaCreateChannelDesc<uint2>(), .bpp=64 } },
    { DXGI_FORMAT_R32G32_SINT,           { .desc = cudaCreateChannelDesc<uint2>(), .bpp=64 } },

    { DXGI_FORMAT_R8G8B8A8_TYPELESS,     { .desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindNone), .bpp=32 } },
    { DXGI_FORMAT_R8G8B8A8_UNORM,        { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>(), .bpp=32 } },
    { DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,   { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>(), .bpp=32, .bs=0, .sRGB=true } },
    { DXGI_FORMAT_R8G8B8A8_UINT,         { .desc = cudaCreateChannelDesc<uchar4>(), .bpp=32 } },
    { DXGI_FORMAT_R8G8B8A8_SNORM,        { .desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized8X4>(), .bpp=32 } },
    { DXGI_FORMAT_R8G8B8A8_SINT,         { .desc = cudaCreateChannelDesc<char4>(), .bpp=32 } },
    { DXGI_FORMAT_R16G16_TYPELESS,       { .desc = cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindNone), .bpp=32 } },
    { DXGI_FORMAT_R16G16_FLOAT,          { .desc = cudaCreateChannelDesc<float2>(), .bpp=32 } },
    { DXGI_FORMAT_R16G16_UNORM,          { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized16X2>(), .bpp=32 } },
    { DXGI_FORMAT_R16G16_UINT,           { .desc = cudaCreateChannelDesc<uint2>(), .bpp=32 } },
    { DXGI_FORMAT_R16G16_SNORM,          { .desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized16X2>(), .bpp=32 } },
    { DXGI_FORMAT_R16G16_SINT,           { .desc = cudaCreateChannelDesc<int2>(), .bpp = 32 } },
    { DXGI_FORMAT_R32_TYPELESS,          { .desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindNone), .bpp=32, } },
    { DXGI_FORMAT_D32_FLOAT,             { .desc = cudaCreateChannelDesc<float>(), .bpp=32, } },
    { DXGI_FORMAT_R32_FLOAT,             { .desc = cudaCreateChannelDesc<float>(), .bpp=32, } },
    { DXGI_FORMAT_R32_UINT,              { .desc = cudaCreateChannelDesc<unsigned int>(), .bpp=32, } },
    { DXGI_FORMAT_R32_SINT,              { .desc = cudaCreateChannelDesc<int>(), .bpp=32, } },

    { DXGI_FORMAT_R8G8_TYPELESS,         { .desc = cudaCreateChannelDesc(8, 8, 0, 0, cudaChannelFormatKindNone), .bpp=16, } },
    { DXGI_FORMAT_R8G8_UNORM,            { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X2>(), .bpp=16, } },
    { DXGI_FORMAT_R8G8_UINT,             { .desc = cudaCreateChannelDesc<uchar2>(), .bpp=16, } },
    { DXGI_FORMAT_R8G8_SNORM,            { .desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized8X2>(), .bpp=16, } },
    { DXGI_FORMAT_R8G8_SINT,             { .desc = cudaCreateChannelDesc<char2>(), .bpp=16, } },
    { DXGI_FORMAT_R16_TYPELESS,          { .desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindNone), .bpp=16, } },
    { DXGI_FORMAT_R16_FLOAT,             { .desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat), .bpp=16, } },
    { DXGI_FORMAT_D16_UNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized16X1>(), .bpp=16, } },
    { DXGI_FORMAT_R16_UNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized16X1>(), .bpp=16, } },
    { DXGI_FORMAT_R16_UINT,              { .desc = cudaCreateChannelDesc<unsigned short>(), .bpp=16, } },
    { DXGI_FORMAT_R16_SNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized16X1>(), .bpp=16, } },
    { DXGI_FORMAT_R16_SINT,              { .desc = cudaCreateChannelDesc<short>(), .bpp=16, } },

    { DXGI_FORMAT_R8_TYPELESS,           { .desc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindNone), .bpp=8, } },
    { DXGI_FORMAT_R8_UNORM,              { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X1>(), .bpp=8, } },
    { DXGI_FORMAT_R8_UINT,               { .desc = cudaCreateChannelDesc<unsigned char>(), .bpp=8, } },
    { DXGI_FORMAT_R8_SNORM,              { .desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedNormalized8X1>(), .bpp=8, } },
    { DXGI_FORMAT_R8_SINT,               { .desc = cudaCreateChannelDesc<char>(), .bpp=8, } },
    { DXGI_FORMAT_A8_UNORM,              { .desc = cudaCreateChannelDesc(0, 0, 0, 8, cudaChannelFormatKindUnsignedNormalized8X1), .bpp=8, } },
    { DXGI_FORMAT_R8G8_B8G8_UNORM,       { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>(), .bpp=32, .packed = true } }, // pixels are re-used
    { DXGI_FORMAT_G8R8_G8B8_UNORM,       { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>(), .bpp=32, .packed = true } },

    { DXGI_FORMAT_BC1_UNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1>(),     .bpp=1,  .bs=8  } },
    { DXGI_FORMAT_BC1_UNORM_SRGB,        { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed1SRGB>(), .bpp=1,  .bs=8, .sRGB = true } },
    { DXGI_FORMAT_BC2_UNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed2>(),     .bpp=8,  .bs=16 } },
    { DXGI_FORMAT_BC2_UNORM_SRGB,        { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed2SRGB>(), .bpp=8,  .bs=16, .sRGB = true } },
    { DXGI_FORMAT_BC3_UNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3>(),     .bpp=8,  .bs=16 } },
    { DXGI_FORMAT_BC3_UNORM_SRGB,        { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed3SRGB>(), .bpp=8,  .bs=16, .sRGB = true } },
    { DXGI_FORMAT_BC4_UNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed4>(),     .bpp=4,  .bs=8  } },
    { DXGI_FORMAT_BC4_SNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed4>(),       .bpp=4,  .bs=8  } },
    { DXGI_FORMAT_BC5_UNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed5>(),     .bpp=8,  .bs=16 } },
    { DXGI_FORMAT_BC5_SNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed5>(),       .bpp=8,  .bs=16 } },

    { DXGI_FORMAT_B8G8R8A8_UNORM,        { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>(),        .bpp=32, } },
    { DXGI_FORMAT_B8G8R8X8_UNORM,        { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>(),        .bpp=32, } },
    { DXGI_FORMAT_B8G8R8A8_TYPELESS,     { .desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindNone),               .bpp=32, } },
    { DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,   { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>(),        .bpp=32, .sRGB = true } },
    { DXGI_FORMAT_B8G8R8X8_TYPELESS,     { .desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindNone),               .bpp=32, } },
    { DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,   { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedNormalized8X4>(),        .bpp=32, .sRGB = true } },

    { DXGI_FORMAT_BC6H_UF16,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed6H>(),    .bpp=16, .bs=16, } },
    { DXGI_FORMAT_BC6H_SF16,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindSignedBlockCompressed6H>(),      .bpp=16, .bs=16, } },
    { DXGI_FORMAT_BC7_UNORM,             { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7>(),     .bpp=8,  .bs=16, } },
    { DXGI_FORMAT_BC7_UNORM_SRGB,        { .desc = cudaCreateChannelDesc<cudaChannelFormatKindUnsignedBlockCompressed7SRGB>(), .bpp=8,  .bs=16, .sRGB = true } },
};

FormatMapping const* getPixelFormatMapping( DXGI_FORMAT format )
{
    if( auto it = mappings.find(format); it != mappings.end() )
        return &it->second;   
    return nullptr;
}


