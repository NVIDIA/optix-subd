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

#include "./textureCache.h"
#include "./texture.h"

#include <OptiXToolkit/Util/Exception.h>


#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>

// clang-format on

namespace fs = std::filesystem;


fs::path resolveFilePath( fs::path const& filePath, fs::path const& relativePath = {} )
{
    if( filePath.empty() )
        return {};

    auto checkPath = []( fs::path const& fp ) -> fs::path {
        if( fs::is_regular_file( fp ) )
        {
            fs::path dds_fp( fp );
            dds_fp.replace_extension( ".dds" );
            if( fs::exists( dds_fp ) )
                return dds_fp;
            return fp;
        }
        return {};
    };

    if( auto fp = checkPath( filePath ); !fp.empty() )
        return fp;

    if( relativePath.empty() )
        return {};

    if( auto fp = checkPath( relativePath / filePath ); !fp.empty() )
        return fp;

    return {};
}

void TextureCache::clear()
{
    std::lock_guard<std::mutex> lock( m_texturesMutex );

    wait();

    m_textures.clear();
}

void TextureCache::wait()
{
    std::lock_guard<std::mutex> lock( m_loadTasksMutex );

    for (auto& task : m_loadTasks)
        task.wait();

    m_loadTasks.clear();
}

std::shared_ptr<const TextureCuda> TextureCache::addTexture(
    const std::string& filepath, const TextureOptions& options )
{
    return addTexture( filepath, options, false );
}

std::shared_ptr<const TextureCuda> TextureCache::addTextureAsync(
    const std::string& filepath, const TextureOptions& options )
{
    return addTexture( filepath, options, true );
}

std::shared_ptr<const TextureCuda> TextureCache::addTexture( 
    const std::string& filepath, const TextureOptions& options, bool async )
{
    std::string fp = resolveFilePath( filepath, "../../scenes/textures" ).generic_string();

    if( fp.empty() )
        return nullptr;

    std::shared_ptr<TextureCuda> ctex;
    {
        // critical section : get or create texture for filepath
        std::lock_guard<std::mutex> lock( m_texturesMutex );

        if( auto it = m_textures.find( filepath ); it != m_textures.end() )
            return it->second;
        else
        {
            ctex = std::make_shared<TextureCuda>();

            m_textures.insert( { filepath, ctex } );
        }
    }

    auto load = [this]( std::string&& filepath, std::shared_ptr<TextureCuda> ctex, TextureOptions options ) {
        Texture tex;
        tex.read( filepath, options );
        m_memoryUse += tex.m_size;
        tex.upload( *ctex, options.mipmap );
        std::fprintf( stdout, "loaded %s\n", filepath.c_str() );
    };

    if( async )
    {
        std::lock_guard<std::mutex> lock( m_loadTasksMutex );

        auto future = std::async( std::launch::async, load, std::move( fp ), ctex, options );

        m_loadTasks.push_back( std::move( future ) );
    }
    else
        load( std::move( fp ), ctex, options );

    return ctex;
}


std::shared_ptr<const TextureCuda> TextureCache::addTextureFromArgs( const std::string& args, const TextureOptions& options )
{
    auto split = [](const std::string& s, const std::string& delimiter) {

        size_t pos_start = 0, pos_end, delim_len = delimiter.length();
        std::string token;
        std::vector<std::string> res;

        while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos)
        {
            token = s.substr(pos_start, pos_end - pos_start);
            pos_start = pos_end + delim_len;
            res.push_back(token);
        }

        res.push_back(s.substr(pos_start));
        return res;
    };

    std::string filepath;

    cudaTextureAddressMode addressMode = cudaTextureAddressMode::cudaAddressModeWrap;
    bool buildMipLevels = options.mipmap;

    std::vector<std::string> c = split( args, " " );
    for( int i = 0; i < c.size(); ++i )
    {
        const std::vector<std::string> o = split( c[i], "=" );
        if( o.size() < 2 )
        {
            std::stringstream ss;
            ss << "invalid texture option " << c[i] << "\n";
            throw std::runtime_error( ss.str() );
        }

        if( o[0] == "tex" )
        {
            filepath = o[1];
        }
        else if( o[0] == "address" )
        {
            if( o[1] == "w" )
                addressMode = cudaTextureAddressMode::cudaAddressModeWrap;
            else if( o[1] == "c" )
                addressMode = cudaTextureAddressMode::cudaAddressModeClamp;
            else if( o[1] == "r" )
                addressMode = cudaTextureAddressMode::cudaAddressModeMirror;
            else
            {
                std::stringstream ss;
                ss << o[1] << " is an invalid addressMode. \n";
                ss << "available addressModes: <w|c|m> \n";
                throw std::runtime_error( ss.str() );
            }
        }
        else if( o[0] == "mipmap" )
        {
            if( o[1] == "false" )
            {
                buildMipLevels = false;
            }
            else if ( o[1] == "true" )
            {
                buildMipLevels = true;
            }
        }
        else
        {
            std::stringstream ss;
            ss << o[0] << " is not a valid texture option"
               << "\n";
            throw std::runtime_error( ss.str() );
        }
    }

    std::string fp = resolveFilePath( filepath, "../../scenes/textures"  ).generic_string();

    return addTexture( fp, options );
}

char const* TextureCache::findTexturePath( const TextureCuda* textureCuda ) const
{
    for( auto const& it : m_textures )
        if( it.second.get() == textureCuda )
            return it.first.c_str();
    return nullptr;
}
