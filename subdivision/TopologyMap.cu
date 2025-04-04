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

#include "TopologyMap.h"
#include "SubdivisionPlanCUDA.h"

#include "../statistics.h"

#include <opensubdiv/tmr/neighborhood.h>
#include <opensubdiv/tmr/topologyMap.h>

#include <vector>


using namespace OpenSubdiv;

// clang-format on

template <typename T>
struct segmented_vector
{
    std::vector<T>        elements;      // element vector
    std::vector<uint32_t> offsets{ 0 };  // offset vector first element 0
    std::vector<uint32_t> sizes;         // segment sizes

    template <typename U>  // a container
    void append( const U& segment )
    {
        sizes.push_back( segment.size() );
        offsets.push_back( offsets.back() + sizes.back() );
        elements.insert( elements.end(), segment.begin(), segment.end() );
    }

    void append( const T* a_elements, uint32_t n_elements )
    {
        sizes.push_back( n_elements );
        offsets.push_back( offsets.back() + sizes.back() );
        elements.insert( elements.end(), &a_elements[0], &a_elements[n_elements] );
    }

    void reserve( size_t n )
    {
        offsets.reserve(n);
        sizes.reserve(n);
    }
};

TopologyMap::TopologyMap( 
    std::unique_ptr<OpenSubdiv::OPENSUBDIV_VERSION::Tmr::TopologyMap> atopologyMap )
    : a_topology_map( std::move( atopologyMap ) )
{

}

void TopologyMap::initDeviceData( bool keepHostData )
{
    assert(a_topology_map);

    Sdc::SchemeType schemeType = a_topology_map->GetTraits().getSchemeType();
    Tmr::EndCapType endcapType = a_topology_map->GetTraits().getEndCapType();

    const int n_plans = a_topology_map->GetNumSubdivisionPlans();

    OTK_ASSERT( n_plans > 0 );

    std::vector<SubdivisionPlanCUDA> a_plans( n_plans );

    segmented_vector<uint32_t> a_trees;
    a_trees.reserve( n_plans );

    segmented_vector<int> a_patch_point_index_arrays;
    a_patch_point_index_arrays.reserve( n_plans );

    segmented_vector<float> a_stencil_matrix_arrays;
    a_stencil_matrix_arrays.reserve( n_plans );

    for( auto i_plan = 0; i_plan < n_plans; ++i_plan )
    {
        SubdivisionPlanCUDA& cuda_plan = a_plans[i_plan];

        cuda_plan._scheme = schemeType;
        cuda_plan._endCap = endcapType;

        if( const auto* plan = a_topology_map->GetSubdivisionPlan( i_plan ) )
        {
            const auto& tree_desc = plan->GetTreeDescriptor();
            
            cuda_plan._numControlPoints   = tree_desc.GetNumControlPoints();
            cuda_plan._coarseFaceQuadrant = tree_desc.GetSubfaceIndex();
            cuda_plan._coarseFaceSize     = tree_desc.GetFaceSize();

            a_patch_point_index_arrays.append( plan->GetPatchPoints() );
            a_trees.append( plan->GetPatchTreeData() );
            a_stencil_matrix_arrays.append( plan->GetStencilMatrix() );
        }
        else
        {
            static std::vector<int> empty{};
            a_patch_point_index_arrays.append( empty );
            a_trees.append( empty );
            a_stencil_matrix_arrays.append( reinterpret_cast<std::vector<float>&>( empty ) );
        }
    }

    d_subpatch_trees_arrays.uploadAsync( a_trees.elements );
    d_patch_point_indices_arrays.uploadAsync( a_patch_point_index_arrays.elements );
    d_stencil_matrix_arrays.uploadAsync( a_stencil_matrix_arrays.elements );

    for( auto i_plan = 0; i_plan < n_plans; ++i_plan )
    {
        a_plans[i_plan]._tree = d_subpatch_trees_arrays.subspan( a_trees.offsets[i_plan], a_trees.sizes[i_plan] );

        a_plans[i_plan]._patchPoints = d_patch_point_indices_arrays.subspan( a_patch_point_index_arrays.offsets[i_plan],
                                                                             a_patch_point_index_arrays.sizes[i_plan] );

        a_plans[i_plan]._stencilMatrix =
            d_stencil_matrix_arrays.subspan( a_stencil_matrix_arrays.offsets[i_plan], a_stencil_matrix_arrays.sizes[i_plan] );
    }

    d_plans.uploadAsync( a_plans );    

    if ( !keepHostData )
        a_topology_map.reset();
}

TopologyMap::TopologyMap( const TopologyMap::TopologyMapDesc& desc )
{
    d_plans.uploadAsync( desc.plans.data(), desc.plans.size() );

    d_subpatch_trees_arrays.uploadAsync( desc.subpatch_trees.data(), desc.subpatch_trees.size() );
    d_patch_point_indices_arrays.uploadAsync( desc.patchpoint_indices.data(), desc.patchpoint_indices.size() );
    d_stencil_matrix_arrays.uploadAsync( desc.stencil_matrices.data(), desc.stencil_matrices.size() );
}

stats::TopologyMapStats TopologyMap::computeStatistics(
    const Tmr::TopologyMap& topologyMap, int histogramSize)
{
    using namespace OpenSubdiv;

    stats::TopologyMapStats stats;

    {
        auto hashStats = topologyMap.ComputeHashTableStatistics();

        stats.pslMean = hashStats.pslMean;
        stats.hashCount = hashStats.hashCount;
        stats.addressCount = hashStats.addressCount;
        stats.loadFactor = hashStats.loadFactor;
    }

    int nplans = topologyMap.GetNumSubdivisionPlans();

    if( nplans == 0 )
        return stats;

    size_t stencilSum = 0;

    for( int planIndex = 0; planIndex < nplans; ++planIndex )
    {
        Tmr::SubdivisionPlan const* plan = topologyMap.GetSubdivisionPlan( planIndex );

        if( planIndex == 0 && !plan )
        {
            assert( Tmr::TopologyMap::kRegularPlanAtIndexZero );
            continue;
        }

        if( plan->IsRegularFace() )
            ++stats.regularFacePlansCount;

        stats.maxFaceSize = std::max( (uint32_t)plan->GetFaceSize(), stats.maxFaceSize );

        if( plan->GetNumNeighborhoods() )
        {

            Tmr::Neighborhood const& n = plan->GetNeighborhood( 0 );

            Tmr::ConstFloatArray corners = n.GetCornerSharpness();
            Tmr::ConstFloatArray creases = n.GetCreaseSharpness();

            if( bool hasSharpness = !( corners.empty() && creases.empty() ) )
            {
                ++stats.sharpnessCount;
                for( int i = 0; i < corners.size(); ++i )
                    stats.sharpnessMax = std::max( stats.sharpnessMax, corners[i] );
                for( int i = 0; i < creases.size(); ++i )
                    stats.sharpnessMax = std::max( stats.sharpnessMax, creases[i] );
            }
        }

        size_t nstencils = plan->GetNumStencils();

        stencilSum += nstencils;

        stats.stencilCountMin = std::min( stats.stencilCountMin, (uint32_t)nstencils );
        stats.stencilCountMax = std::max( stats.stencilCountMax, (uint32_t)nstencils );

        stats.plansByteSize += plan->GetByteSize( true );
    }

    stats.plansCount = nplans - ( topologyMap.GetSubdivisionPlan( 0 ) == nullptr );

    stats.stencilCountAvg = float( stencilSum ) / float( stats.plansCount );

    // fill stencil counts histogram
    if( stats.stencilCountMin == stats.stencilCountMax )
    {
        // all the plans have the same number of stencils (ex. single cube)
        stats.stencilCountHistogram.push_back( stats.plansCount );
    }
    else
    {
        stats.stencilCountHistogram.resize( histogramSize );

        float delta = float( stats.stencilCountMax - stats.stencilCountMin ) / histogramSize;

        for( uint32_t planIndex = 0; planIndex < nplans; ++planIndex )
        {
            Tmr::SubdivisionPlan const* plan = topologyMap.GetSubdivisionPlan( planIndex );

            if( planIndex == 0 && !plan )
                continue;

            size_t nstencils = plan->GetNumStencils();

            uint32_t i = (uint32_t)std::floor( float( nstencils - stats.stencilCountMin ) / delta );

            ++stats.stencilCountHistogram[std::min( uint32_t( histogramSize - 1 ), i )];
        }
    }
    return stats;
}

