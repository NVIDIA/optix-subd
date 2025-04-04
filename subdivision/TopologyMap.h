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

#include <OptiXToolkit/Util/CuBuffer.h>

#include <opensubdiv/version.h>

#include <cstdint>
#include <span>
// clang-format on

struct SubdivisionPlanCUDA;
namespace OpenSubdiv::OPENSUBDIV_VERSION::Tmr 
{
    class TopologyMap;
}
namespace stats
{
    struct TopologyMapStats;    
}

struct TopologyMap
{
    CuBuffer<uint32_t> d_subpatch_trees_arrays;
    CuBuffer<int>      d_patch_point_indices_arrays;
    CuBuffer<float>    d_stencil_matrix_arrays;

    CuBuffer<SubdivisionPlanCUDA> d_plans;

    std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap> a_topology_map;

    TopologyMap( const TopologyMap& other ) = delete;
    TopologyMap( TopologyMap&& other ) = delete;

    TopologyMap( std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap> atopologyMap );

    void initDeviceData( bool keepHostData = true );

    // direct de-serialization (ex. when loading from file) ; cannot hash topology
    struct TopologyMapDesc
    {
        std::span<SubdivisionPlanCUDA> plans;
        std::span<uint32_t>            subpatch_trees;
        std::span<int>                 patchpoint_indices;
        std::span<float>               stencil_matrices;
    };
    TopologyMap( const TopologyMapDesc& desc );

    // statistics

    struct SubdivisionPlanStats
    {
        uint32_t plansCount    = 0;
        size_t   plansByteSize = 0;

        uint32_t regularFacePlansCount = 0;  // plans created without quadrangulation

        uint32_t maxFaceSize    = 0;
        uint32_t sharpnessCount = 0;
        float    sharpnessMax   = 0.f;

        uint32_t              stencilCountMin = ~uint32_t( 0 );
        uint32_t              stencilCountMax = 0;
        float                 stencilCountAvg = 0.f;
        std::vector<uint32_t> stencilCountHistogram;
    };

    static stats::TopologyMapStats computeStatistics( 
        const OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap& topologyMap, int histogramSize = 50 );
};

