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

#include <cuda.h>
#include <cuda_runtime.h>

#include "./materialCache.h"
#include "./materialCuda.h"

#include <subdivision/SubdivisionSurface.h>
#include <scene/shapeUtils.h>
#include <texture/textureCache.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <numeric>

namespace fs = std::filesystem;

// clang-format on

float3& operator << (float3& a, auto const& b) 
{ 
    static_assert(sizeof(b) == sizeof(float3));
    std::memcpy(&a, b, sizeof(float3));
    return a; 
}

struct Context
{
    TextureCache& texCache;

    Shape const& shape;
    
    fs::path const& basepath;

    uint32_t nmaterials = 0;
    uint32_t nsurfaces = 0;
};

TextureOptions MaterialCache::getTextureOptions( MaterialCache::TextureType t )
{
    TextureOptions options;
    options.convertToSingleChannel = ( t == MaterialCache::TextureType::DISPLACEMENT || t == MaterialCache::TextureType::ROUGHNESS );
    options.vflip                  = ( t != MaterialCache::TextureType::ENVMAP );
    options.mipmap                 = ( t == MaterialCache::TextureType::DISPLACEMENT );
    return options;
}

static void translateMaterial( Context const& ctx, Shape::material const& mtl, MaterialCuda& m )
{
    m.albedo << mtl.kd;
    m.specular << mtl.ks;
    m.roughness                 = mtl.Pr;
    m.metalness                 = mtl.Pm;
    m.displacementSampler.scale = mtl.bm;
    m.displacementSampler.bias  = mtl.bb;
    m.udim                      = mtl.udim;
    m.udimMax                   = mtl.udimMax;

    using enum MaterialCache::TextureType;

    auto addTexture = [&ctx]( const std::string& texpath, MaterialCache::TextureType texType ) -> const TextureCuda* {
        
        if( texpath.empty() )
            return nullptr;
        
        if( fs::path fp = ( ctx.basepath / texpath ).lexically_normal(); fs::is_regular_file( fp ) )
        {
            TextureOptions opts = MaterialCache::getTextureOptions( texType );

            auto tex = ctx.texCache.addTextureAsync( fp.generic_string(), opts );

            // note: the only way to remove a texture from the cache is a clear,
            // which will wait until all loading operations are completed, so this
            // unprotected pointer should be actually safe.
            return tex.get();
        }
        return nullptr;
    };

    m.albedoMap = addTexture( mtl.map_kd, ALBEDO );

    // Disable textures that are not used in this sample
#if 0
    {
        m.albedoMap               = addTexture( mtl.map_kd, ALBEDO );
        m.specularMap             = addTexture( mtl.map_ks, SPECULAR );
        m.roughnessMap            = addTexture( mtl.map_pr, ROUGHNESS );
        m.metalnessMap            = addTexture( mtl.map_pm, METALNESS );
    }
#endif

    m.displacementSampler.tex = addTexture( mtl.map_bump, DISPLACEMENT );
    
    m.udim = mtl.udim;
}

static std::vector<uint16_t> quadrangulateBindings(
    Shape const& shape, uint32_t nsurfaces, uint16_t idOffset )
{
    assert( shape.scheme == Scheme::kCatmark );

    if( shape.nvertsPerFace.empty() || shape.mtlbind.empty() || !nsurfaces )
        return {};

    std::vector<uint16_t> bindings( nsurfaces );

    for( uint32_t face = 0, vcount = 0; face < (uint32_t)shape.nvertsPerFace.size(); ++face )
    {
        int nverts = shape.nvertsPerFace[face];
        
        uint32_t materialID = shape.mtlbind[face];

        if( nverts == 4 )
            bindings[vcount++] = static_cast<uint16_t>( materialID + idOffset );
        else
        {
            for( int vert = 0; vert < nverts; ++vert )
                bindings[vcount + vert] = static_cast<uint16_t>( materialID + idOffset );
            vcount += nverts;
        }        
    }
    return bindings;
}

static std::vector<uint16_t> translateShapeBindings( Context const& ctx, uint16_t idOffset )
{   
    if( ctx.shape.scheme == Scheme::kCatmark )
        return quadrangulateBindings( ctx.shape, ctx.nsurfaces, idOffset );

    std::vector<uint16_t> bindings( ctx.shape.mtlbind.size() );

    for (uint32_t i = 0; i < (uint32_t)ctx.shape.mtlbind.size(); ++i)
        bindings[i] = uint16_t( ctx.shape.mtlbind[i] ) + idOffset;

    return bindings;
}

static std::vector<uint16_t> cacheMaterials( std::vector<MaterialCuda>&                           materials,
                                             Context&                                             ctx,
                                             std::vector<std::unique_ptr<Shape::material>> const& mtls )
{
    // append new materials & bindings

    uint16_t idOffset = (uint16_t)materials.size();

    materials.resize( materials.size() + mtls.size() );

    assert( materials.size() < std::numeric_limits<uint16_t>::max() );

    for( size_t i = 0; i < mtls.size(); ++i )
        translateMaterial( ctx, *mtls[i], materials[idOffset + i] );

    return translateShapeBindings( ctx, idOffset );
}

MaterialCache::MaterialCache( )
    : m_textureCache( std::make_unique<TextureCache>() )
{

}

MaterialCache::~MaterialCache() = default;

std::vector<uint16_t> MaterialCache::cacheMaterials( Shape const& shape, uint32_t nsurfaces )
{
    if( nsurfaces == 0 || shape.mtls.empty() || shape.mtlbind.empty() )
        return {};

    Context ctx = {
        .texCache = getTextureCache(),
        .shape = shape,
        .basepath = shape.filepath.parent_path(),
        .nmaterials = (uint32_t)shape.mtls.size(),
        .nsurfaces = nsurfaces,
    };

    return ::cacheMaterials( m_materials, ctx, shape.mtls );
}

std::vector<uint16_t> MaterialCache::cacheMaterials( std::vector<std::unique_ptr<Shape::material>> const& mtls,
                                                     std::vector<unsigned short> const&                   mtlbind,
                                                     fs::path const&                                      basepath )
{
    Shape dummyShape{};
    dummyShape.scheme  = kBilinear;  // HACK: don't call quadrangulateBindings
    dummyShape.mtlbind = mtlbind;
    Context ctx        = {
               .texCache   = getTextureCache(),
               .shape      = dummyShape,
               .basepath   = basepath,
               .nmaterials = (uint32_t)mtls.size(),
               .nsurfaces  = 0,
    };
    return ::cacheMaterials( m_materials, ctx, mtls );
}

void MaterialCache::initDeviceData( )
{
    // wait for the textures to be uploaded to device & populate with the device pointers
    if( m_textureCache )
        m_textureCache->wait();

    uint32_t materialID = 0;
    for( auto& material : m_materials )
    {
        {
            if( material.albedoMap )
                material.albedoMap = material.albedoMap->getDevicePtr();
            if( material.specularMap )
                material.specularMap = material.specularMap->getDevicePtr();
            if( material.roughnessMap )
                material.roughnessMap = material.roughnessMap->getDevicePtr();
            if( material.metalnessMap )
                material.metalnessMap = material.metalnessMap->getDevicePtr();
        }

        if( material.displacementSampler.tex )
            material.displacementSampler.tex = material.displacementSampler.tex->getDevicePtr();
    }

    if( !m_materials.empty() )
    {
        d_materials.uploadAsync( m_materials );

        OTK_ASSERT( d_materials.cu_ptr() );
    }
}

void MaterialCache::clear()
{
    m_materials.clear();

    d_materials = {};

    if( m_textureCache )
        m_textureCache->clear();
}
