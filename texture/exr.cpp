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

#define TINYEXR_USE_MINIZ 1
#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_THREAD 1
#include "tinyexr/tinyexr.h"

cudaPitchedPtr makeCudaPitchedPtr(const FormatMapping& mapping, void* ptr, uint32_t width, uint32_t height);

void flipVertically( Texture& tex );
void convertToSingleChannel( Texture& tex );

void readEXRFile( Texture& tex, const std::string& filepath, bool vflip, bool singleChannel )
{
    float* data = nullptr;
    int    w = 0;
    int    h = 0;
    int    nchannels = 4;

    const char* err = nullptr;
    if( LoadEXR( &data, &w, &h, filepath.c_str(), &err ) < 0 )
    {
        std::stringstream ss;
        ss << "Failed to load EXR " << filepath << "\n" << err << "\n";
        throw std::runtime_error( ss.str() );
    }

    tex.m_pixelFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;

    tex.m_width = w;
    tex.m_height = h;

    assert( nchannels == 4 || nchannels == 1 );
    tex.m_pixelFormat = nchannels == 4 ?
        DXGI_FORMAT_R32G32B32A32_FLOAT : DXGI_FORMAT_R32_FLOAT;

    FormatMapping const* mapping = getPixelFormatMapping( tex.m_pixelFormat );
    if( mapping )
        tex.m_pixelFormatMapping = *mapping;
    else
        throw std::runtime_error("unsupported pixel format '" + filepath + "'");

    tex.m_mipmaps.resize(1);
    tex.m_mipmaps[0] = makeCudaPitchedPtr( *mapping, data, w, h );

    tex.m_data = std::unique_ptr<uint8_t[]>(reinterpret_cast<uint8_t*>(data));
    
    if( singleChannel )
        convertToSingleChannel( tex );

    if( vflip )
        flipVertically( tex );
}