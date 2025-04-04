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

// clang-format off

#include "./materialCuda.h"

#include <scene/shapeUtils.h>
#include <OptiXToolkit/Util/CuBuffer.h>

#include <cstdint>
#include <vector>
#include <filesystem>

// clang-format on

struct MaterialCuda;
struct Shape;
struct TextureOptions;
class TextureCache;

// note: the MaterialCache is currently not thread-safe !

class MaterialCache
{
  public:
    enum TextureType
    {
        ALBEDO = 0,
        ROUGHNESS,
        SPECULAR,
        METALNESS,
        DISPLACEMENT,
        ENVMAP,
        COUNT
    };

    MaterialCache( );
    ~MaterialCache();

    void clear();

    size_t size() const
    {
      return d_materials.size();
    }

    std::vector<uint16_t> cacheMaterials( Shape const& shape, uint32_t nsurfaces = 0 );

    std::vector<uint16_t> cacheMaterials( std::vector<std::unique_ptr<Shape::material>> const& mtls,
                                          std::vector<unsigned short> const&                   mtlbind,
                                          std::filesystem::path const&                         basepath );

    // waits for pending textures to finish loading and uploads the finalized
    // materials to device
    void initDeviceData();

    // pointer to device data
    std::span<MaterialCuda const> getDeviceData() const;

    TextureCache& getTextureCache() { return *m_textureCache; }    
    TextureCache const& getTextureCache() const { return *m_textureCache; }

    // returns preferred presets to load asset textures (obsolete soon)
    static TextureOptions getTextureOptions( TextureType t );

  private:
    MaterialCache( MaterialCache& )                  = delete;
    MaterialCache& operator=( MaterialCache const& ) = delete;

    std::vector<MaterialCuda> m_materials;

    CuBuffer<MaterialCuda> d_materials;

    std::unique_ptr<TextureCache> m_textureCache;
};

inline std::span<MaterialCuda const> MaterialCache::getDeviceData() const
{
    return d_materials.span();
}
