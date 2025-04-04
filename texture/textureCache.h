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
#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_map>
#include "textureCuda.h"
// clang-format on

struct Texture;

struct TextureOptions
{
    bool mipmap                 = false;
    bool vflip                  = false;
    bool convertToSingleChannel = false;
};

class TextureCache
{
  public:
    TextureCache() = default;
    ~TextureCache() = default;

    // loads a texture synchronously - the TextureCuda instanced returned has
    // a valid device pointer and is ready to use on device.
    std::shared_ptr<const TextureCuda> addTexture(
        const std::string& filepath, const TextureOptions& options );

    // synchronous texture load from parsed arguments
    std::shared_ptr<const TextureCuda> addTextureFromArgs( 
        const std::string& args, const TextureOptions& options );

    // note: the TextureCuda instance that is returned is valid, but there is no
    // guarantee that is ready to use on device. Call wait() to guarantee all
    // textures have been fully initilized on device.
    std::shared_ptr<const TextureCuda> addTextureAsync(
        const std::string& filepath, const TextureOptions& options );

    // blocks until all pending async texture tasks are completed
    void wait();

    void clear();

    // iterators for range-based cache traversal
    // note: texture accessors are not thread-safe ; any cache operation
    // (concurrent or not) will invalidate these iterators. Tread carefully.
    typedef std::unordered_map<std::string, std::shared_ptr<TextureCuda>>::const_iterator const_iterator_type;

    const_iterator_type begin() const { return m_textures.begin(); }

    const_iterator_type end() const { return m_textures.end(); }

    // searches the texture cache for the TextureCuda (host memory pointer)
    // note: O(n) search time !
    char const* findTexturePath( const TextureCuda* textureCuda ) const;

    size_t memoryUse() const { return m_memoryUse.load(); }

  private:
    TextureCache( const TextureCache& c )            = delete;
    TextureCache& operator=( const TextureCache& c ) = delete;

    std::atomic<size_t> m_memoryUse = 0;

    std::shared_ptr<const TextureCuda> addTexture(
        const std::string& filepath, const TextureOptions& options, bool async );

    std::mutex m_texturesMutex;
    std::unordered_map<std::string, std::shared_ptr<TextureCuda>> m_textures;

    std::mutex m_loadTasksMutex;
    std::vector<std::future<void>> m_loadTasks;
};
