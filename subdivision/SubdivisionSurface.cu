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
#include "./SubdivisionSurface.h"

#include "./SubdivisionSurfaceCUDA.h"
#include "./SubdivisionPlanCUDA.h"
#include "./TopologyCache.h"
#include "./TopologyMap.h"

#include "./farUtils.h"

#include "./statistics.h"
#include "../utils.cuh"

#include <opensubdiv/tmr/surfaceTableFactory.h>
#include <opensubdiv/tmr/subdivisionPlanBuilder.h>
#include <opensubdiv/tmr/subdivisionPlan.h>

#include <scene/shapeUtils.h>

#include <OptiXToolkit/Util/Exception.h>

#include <span>
#include <numeric>
#include <thrust/extrema.h>
#include <thrust/transform_reduce.h>

#include <cub/cub.cuh>
// clang-format on

using namespace OpenSubdiv;

using TexcoordDeviceData = SubdivisionSurface::SurfaceTableDeviceData<Tmr::LinearSurfaceDescriptor, TexCoord>;

template <typename DescriptorT, typename PatchPointT>
void initSubdLinearDeviceData( const Tmr::LinearSurfaceTable& surface_table,
                               TexcoordDeviceData&            deviceData )
{
    OTK_ASSERT_MSG( deviceData.control_point_indices.size() == 0, "TMR device data should only be initialized once" );

    deviceData.surface_descriptors.uploadAsync( surface_table.descriptors );

    deviceData.control_point_indices.uploadAsync( surface_table.controlPointIndices );

    // Support (patch) points
    const uint32_t n_surfaces = surface_table.GetNumSurfaces();
    std::vector<uint32_t> patch_points_offsets( n_surfaces + 1, 0 );
    for( auto i = 0; i < n_surfaces; i++ )
    {
        Tmr::LinearSurfaceDescriptor desc = surface_table.GetDescriptor( i );
        if( !desc.HasLimit() )
        {
            patch_points_offsets[i + 1] = patch_points_offsets[i];
            continue;
        }

        uint32_t numPatchPoints = ( desc.GetQuadSubfaceIndex() == Tmr::LOCAL_INDEX_INVALID ) ? 0 : desc.GetFaceSize() + 1;
        patch_points_offsets[i + 1] = patch_points_offsets[i] + numPatchPoints;
    }

    deviceData.patch_points_offsets.uploadAsync( patch_points_offsets );
    deviceData.patch_points.resize( patch_points_offsets.back() );
}

static void gatherStatistics( Shape const& shape,
                              Far::TopologyRefiner const& refiner, 
                              Tmr::TopologyMap const& topologyMap, 
                              Tmr::SurfaceTable const& surfTable )
{
    int nsurfaces = surfTable.GetNumSurfaces();
    
    auto& evalStats = stats::evaluatorSamplers;

    evalStats.indexBufferSize = refiner.GetLevel( 0 ).GetNumFaceVertices() * sizeof( uint32_t );
    evalStats.vertCountBufferSize = refiner.GetLevel( 0 ).GetNumFaces() * sizeof( uint32_t );
   
    evalStats.topologyMapStats = TopologyMap::computeStatistics( topologyMap );

    static constexpr int const histogramSize = 50;

    stats::SurfaceTableStats surfStats;
    
    surfStats.name = shape.filepath.filename().generic_string();

    surfStats.maxValence = refiner.getLevel(0).getMaxValence();

    surfStats.byteSize = surfTable.GetByteSize();

    std::vector<uint8_t> topologyQuality( nsurfaces, 0 );

    size_t stencilSum = 0;

    for( int surfIndex = 0; surfIndex < nsurfaces; ++surfIndex )
    {
        Tmr::SurfaceDescriptor const& desc = surfTable.GetDescriptor( surfIndex );

        if ( !desc.HasLimit() ) {
            ++surfStats.holesCount;
            continue;
        }

        auto const plan = topologyMap.GetSubdivisionPlan( desc.GetSubdivisionPlanIndex() );

        uint8_t& quality = topologyQuality[ surfIndex ];

        // check face size (regular / non-quad)
        if( !plan->IsRegularFace() )
            ++surfStats.irregularFaceCount;

        uint32_t faceSize = plan->GetFaceSize();

        if( faceSize > 5 )
            quality = std::max( quality,  (uint8_t)0xff );

        surfStats.maxFaceSize = std::max( faceSize, surfStats.maxFaceSize );

        // check vertex valences
        const Tmr::Index* controlPoints = surfTable.GetControlPointIndices( surfIndex );

        uint32_t maxVertexValence = 0;
        for( uint8_t i = 0; i < faceSize; ++i )
        {
            auto edges = refiner.GetLevel(0).GetVertexEdges( controlPoints[i] );
            maxVertexValence = std::max( maxVertexValence, uint32_t( edges.size() ) );
        }
        if( maxVertexValence > 8 )
            quality = std::max( quality,  (uint8_t)0xff );

        // check sharpness
        bool hasSharpness = false;

        if( plan->GetNumNeighborhoods() ) 
        {
            Tmr::Neighborhood const& n = plan->GetNeighborhood(0);

            Tmr::ConstFloatArray corners = n.GetCornerSharpness();
            Tmr::ConstFloatArray creases = n.GetCreaseSharpness();

            if( hasSharpness = !( corners.empty() && creases.empty() ) ) {

                auto processSharpness = [&surfStats, &quality]( Tmr::ConstFloatArray values ) {
                    for (int i = 0; i < values.size(); ++i)
                    {
                        if( values[i] >= 10.f )
                            ++surfStats.infSharpCreases;
                        else
                        {
                            surfStats.sharpnessMax = std::max( surfStats.sharpnessMax, values[i] );
                            
                            if( values[i] > 8.f )
                                quality = std::max( quality,  (uint8_t)0xff );
                            else if( values[i] > 4.f)
                                quality = std::max( quality, uint8_t( ( values[i] / Sdc::Crease::SHARPNESS_INFINITE ) * 255.f ) );
                        }
                    }
                };
                processSharpness( creases );
                processSharpness( corners );
            }
        }

        // check stencil matrix
        size_t nstencils = plan->GetNumStencils();

        if( nstencils == 0 )
        {
            if( plan->GetNumControlPoints() == 16 )
                ++surfStats.bsplineSurfaceCount;
            else
                ++surfStats.regularSurfaceCount;
        }
        else 
        {
            if( hasSharpness )
                ++surfStats.sharpSurfaceCount;
            else
                ++surfStats.isolationSurfaceCount;
        }

        stencilSum += nstencils;

        surfStats.stencilCountMin = std::min(surfStats.stencilCountMin, (uint32_t)nstencils);
        surfStats.stencilCountMax = std::max(surfStats.stencilCountMax, (uint32_t)nstencils);
    }

    assert( ( surfStats.holesCount +
              surfStats.bsplineSurfaceCount +
              surfStats.regularSurfaceCount +
              surfStats.isolationSurfaceCount +
              surfStats.sharpSurfaceCount ) == nsurfaces );

    surfStats.stencilCountAvg = float( stencilSum ) / float( nsurfaces );

    surfStats.stencilCountHistogram.resize( histogramSize );

    surfStats.surfaceCount = nsurfaces;


    if( !surfStats.isCatmarkTopology() )
    {
        // if we suspect this was not a sub-d model (likely a triangular mesh), run a second
        // pass of the surfaces to tag all the irregular faces (non-quads) as poor quality
        int const regularFaceSize =
            Sdc::SchemeTypeTraits::GetRegularFaceSize( refiner.GetSchemeType() );

        const Vtr::internal::Level& level = refiner.getLevel( 0 );
        for( int faceIndex = 0, surfaceIndex = 0; faceIndex < level.getNumFaces(); ++faceIndex )
        {
            if( level.isFaceHole( faceIndex ) )
                continue;
            if( int nverts =  level.getFaceVertices( faceIndex ).size(); nverts == regularFaceSize )
                ++surfaceIndex;
            else
            {
                for( int vert = 0; vert < nverts; ++vert, ++surfaceIndex )
                    topologyQuality[surfaceIndex] = 0xff;
            }
        }
    }

    // fill stencil counts histogram
    if( surfStats.stencilCountMin == surfStats.stencilCountMax ) 
    {
        // all the surfaces have the same number of stencils
        surfStats.stencilCountHistogram.push_back( nsurfaces );
    }
    else 
    {
        surfStats.stencilCountHistogram.resize( histogramSize );

        float delta = float(surfStats.stencilCountMax - surfStats.stencilCountMin) / histogramSize;

        for (int surfIndex = 0; surfIndex < nsurfaces; ++surfIndex) {

            Tmr::SurfaceDescriptor const& desc = surfTable.GetDescriptor(surfIndex);

            if (!desc.HasLimit())
                continue;

            auto const plan = topologyMap.GetSubdivisionPlan(desc.GetSubdivisionPlanIndex());

            uint32_t nstencils = (uint32_t)plan->GetNumStencils();

            uint32_t i = (uint32_t)std::floor(float(nstencils - surfStats.stencilCountMin) / delta);

            ++surfStats.stencilCountHistogram[std::min( uint32_t( histogramSize - 1 ), i )];
        }
    }  

    surfStats.buildTopologyRecommendations();

    surfStats.topologyQuality.upload( topologyQuality );
    evalStats.surfaceCountTotal += surfStats.surfaceCount;
    evalStats.surfaceTablesByteSizeTotal += surfStats.byteSize;
    evalStats.hasBadTopology |= ( ! surfStats.topologyRecommendations.empty() );

    evalStats.surfaceTableStats.emplace_back( std::move( surfStats ) );
}


// -----------------------------------------------------------------------------
// SubdivisionSurface
// -----------------------------------------------------------------------------
SubdivisionSurface::SubdivisionSurface( TopologyCache& topologyCache, std::unique_ptr<Shape> shape )
{
    // create Far mesh (control cage topology)

    Sdc::SchemeType schemeType    = GetSdcType( *shape );
    Sdc::Options    schemeOptions = GetSdcOptions( *shape );
    Tmr::EndCapType endCaps       = Tmr::EndCapType::ENDCAP_BSPLINE_BASIS;

    {
        // note: for now the topology cache only supports a single map
        // for a given set of traits ; eventually Tmr::SurfaceTableFactory
        // may support directly topology caches, allowing a given
        // Tmr::SurfaceTable to reference multiple topology maps at run-time.
        Tmr::TopologyMap::Traits traits;
        traits.SetCompatible( schemeType, schemeOptions, endCaps );

        m_topology_map = &topologyCache.get( traits.value() );
        OTK_REQUIRE( m_topology_map->a_topology_map );
    }

    Tmr::TopologyMap& topologyMap = *m_topology_map->a_topology_map;

    std::unique_ptr<Far::TopologyRefiner> refiner;
    
    refiner.reset(Far::TopologyRefinerFactory<Shape>::Create(
        *shape, Far::TopologyRefinerFactory<Shape>::Options(schemeType, schemeOptions)));

    Tmr::SurfaceTableFactory tableFactory;

    Tmr::SurfaceTableFactory::Options options;
    options.planBuilderOptions.endCapType                       = endCaps;
    options.planBuilderOptions.isolationLevel                   = topologyCache.options.isoLevelSharp;
    options.planBuilderOptions.isolationLevelSecondary          = topologyCache.options.isoLevelSmooth;
    options.planBuilderOptions.useSingleCreasePatch             = true;
    options.planBuilderOptions.useInfSharpPatch                 = true;
    options.planBuilderOptions.useTerminalNode                  = topologyCache.options.useTerminalNodes;
    options.planBuilderOptions.useDynamicIsolation              = true;
    options.planBuilderOptions.orderStencilMatrixByLevel        = true;
    options.planBuilderOptions.generateLegacySharpCornerPatches = false;

    m_surface_table = tableFactory.Create( *refiner, topologyMap, options );

    gatherStatistics( *shape, *refiner, topologyMap, *m_surface_table );

    d_positions.uploadAsync( shape->verts.data(), shape->verts.size() );

    initDeviceData( m_surface_table );

    // setup for texcoords
    OTK_ASSERT( shape->hasUV() );
    Tmr::LinearSurfaceTableFactory tableFactoryFvar;
    constexpr int const fvarChannel = 0;
    m_texcoord_surface_table = tableFactoryFvar.Create( *refiner, fvarChannel, m_surface_table.get());

    initSubdLinearDeviceData<Tmr::SurfaceDescriptor, TexCoord>( *m_texcoord_surface_table, m_texcoordDeviceData );

    d_texcoords.uploadAsync(reinterpret_cast<const TexCoord*>(shape->uvs.data()), shape->uvs.size());

    m_aabb = shape->aabb;

    // note : keyframe data is uploaded to device directly by sequence loader
    // to prevent costly mem copies
    m_shape = std::move(shape);
}

std::vector<uint32_t> SubdivisionSurface::getControlCageEdges() const
{
    Sdc::SchemeType schemeType    = GetSdcType( *m_shape );
    Sdc::Options    schemeOptions = GetSdcOptions( *m_shape );

    std::unique_ptr<Far::TopologyRefiner> refiner;

    refiner.reset( Far::TopologyRefinerFactory<Shape>::Create(
        *m_shape, Far::TopologyRefinerFactory<Shape>::Options( schemeType, schemeOptions ) ) );

    auto level = refiner->GetLevel( 0 );

    int numEdges = level.GetNumEdges();
    std::vector<uint32_t> edgeIndices;
    edgeIndices.reserve( 2*numEdges );

    for (int i = 0; i < numEdges; ++i) {
        OpenSubdiv::Far::ConstIndexArray verts = level.GetEdgeVertices(i);
        edgeIndices.push_back(verts[0]);
        edgeIndices.push_back(verts[1]);
    }

    return edgeIndices;
}

uint32_t SubdivisionSurface::numVertices() const
{
    return d_positions.size();
}

uint32_t SubdivisionSurface::surfaceCount() const
{
    if( m_vertexDeviceData.surface_descriptors.size() > 0 )
        return uint32_t(m_vertexDeviceData.surface_descriptors.size());

    if( m_surface_table )
        return m_surface_table->GetNumSurfaces();
    
    return 0;
}


void SubdivisionSurface::initDeviceData( const std::unique_ptr<const OpenSubdiv::Tmr::SurfaceTable>& surface_table )
{
    const uint32_t        n_surfaces = surface_table->GetNumSurfaces();
    std::vector<uint32_t> patch_points_offsets( n_surfaces + 1, 0 );
    OTK_ASSERT( n_surfaces > 0 );  // need at least one surface for this code to work
    for( auto i_surface = 0; i_surface < n_surfaces; ++i_surface )
    {
        const Tmr::SurfaceDescriptor surface = surface_table->descriptors[i_surface];
        if( !surface.HasLimit() )
        {
            patch_points_offsets[i_surface + 1] = patch_points_offsets[i_surface];
            continue;
        }

        // plan is never going to be null here
        const auto* plan = surface_table->topologyMap.GetSubdivisionPlan( surface.GetSubdivisionPlanIndex() );

        patch_points_offsets[i_surface + 1] = patch_points_offsets[i_surface] + plan->GetNumPatchPoints();
    }

    m_vertexDeviceData.surface_descriptors.uploadAsync( surface_table->descriptors );
    m_vertexDeviceData.control_point_indices.uploadAsync( surface_table->controlPointIndices );

    m_vertexDeviceData.patch_points.resize( patch_points_offsets.back() );
    m_vertexDeviceData.patch_points_offsets.uploadAsync( patch_points_offsets );
}

bool SubdivisionSurface::hasAnimation() const
{
    return !d_positionKeyframes.empty();
}

uint32_t SubdivisionSurface::numKeyframes() const
{
    return (uint32_t)d_positionKeyframes.size();
}

static inline otk::Aabb lerpAabb(const otk::Aabb& a, const otk::Aabb& b, float t)
{
    otk::Aabb result;
    result.m_min = lerp(a.m_min, b.m_min, t);
    result.m_max = lerp(a.m_max, b.m_max, t);
    return result;
}

__global__ void lerpKeyframesKernel(const float3* kf0, const float3* kf1, float3* dst, unsigned int numVertices, float t)
{
    const int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index >= numVertices) return;
    const float3 v0 = kf0[thread_index];
    const float3 v1 = kf1[thread_index];

    dst[thread_index] = lerp(v0, v1, t);
}

void lerpVertices(const float3* kf0, const float3* kf1, float3* dst, unsigned int numVertices, float t)
{
    constexpr int blockSize = 32;
    const int     numBlocks = (numVertices + blockSize - 1) / blockSize;
    lerpKeyframesKernel<<<numBlocks, blockSize>>>(kf0, kf1, dst, numVertices, t);
    CUDA_SYNC_CHECK();
}

void SubdivisionSurface::animate( float t, float frameRate )
{
    if( !hasAnimation() )
        return;

    // Cache positions for computing motion vecs
    d_positionsCached.resize(d_positions.size());
    cuMemcpy( d_positionsCached.cu_ptr(), d_positions.cu_ptr(), d_positions.size_in_bytes() );

    uint32_t nframes = d_positionKeyframes.size();

    float frameTime = m_frameOffset + t * frameRate;
    float frame = std::truncf(frameTime);

    // animation implicitly loops if frameTime >= numKeyframes
    int f0 = static_cast<int>(frame) % nframes;
    int f1 = (f0 + 1) % nframes;
    OTK_ASSERT(f1 < nframes);

    const float dt = frameTime - frame;

    const float3* kf0         = reinterpret_cast<float3*>( d_positionKeyframes[f0].data() );
    const float3* kf1         = reinterpret_cast<float3*>( d_positionKeyframes[f1].data() );
    float3*       dst         = reinterpret_cast<float3*>( d_positions.data() );

    lerpVertices( kf0, kf1, dst, numVertices(), dt );

    m_aabb = lerpAabb( m_aabbKeyframes[f0], m_aabbKeyframes[f1], t );
}

void SubdivisionSurface::clearMotionCache()
{
    d_positionsCached.resize( 0 );
}

