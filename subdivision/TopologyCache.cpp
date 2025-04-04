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

#include "./TopologyCache.h"
#include "./TopologyMap.h"

#include <opensubdiv/tmr/topologyMap.h>

// clang-format on

using namespace OpenSubdiv;

union Key {
    Tmr::TopologyMap::Traits traits;
    uint8_t value;
};

TopologyCache::TopologyCache(TopologyCache::Options const& opts) : options(opts)
{

}

TopologyMap& TopologyCache::get( uint8_t traits )
{
    std::lock_guard lock(m_mtx);

    Key key{ .value = traits, };

    if (auto it = m_topologyMaps.find( key.value ); it != m_topologyMaps.end())
        return *it->second;

    auto a_topology_map = std::make_unique<Tmr::TopologyMap>(
        key.traits, Tmr::TopologyMap::Options( uint8_t( m_topologyMaps.size() ) ) );

    auto [it, done] = m_topologyMaps.emplace( 
        key.value, std::make_unique<TopologyMap>( std::move( a_topology_map) ) );

    return *it->second;
}

bool TopologyCache::empty() const
{
    std::lock_guard lock(m_mtx);
    return m_topologyMaps.empty();
}

size_t TopologyCache::size() const
{
    std::lock_guard lock(m_mtx);
    return m_topologyMaps.size();
}

void TopologyCache::clear()
{
    std::lock_guard lock(m_mtx);
    return m_topologyMaps.clear();
}

// note: all hashing must be completed before hoisting maps into device memory !
std::vector<std::unique_ptr<TopologyMap const>> TopologyCache::initDeviceData( bool keepHostData )
{
    std::lock_guard lock(m_mtx);

    std::vector<std::unique_ptr<TopologyMap const>> topologyMaps;

    topologyMaps.reserve(m_topologyMaps.size());

    for( auto& it : m_topologyMaps )
    {
        it.second->initDeviceData( keepHostData );

        topologyMaps.emplace_back( std::move( it.second ) );
    }
    return topologyMaps;
}

