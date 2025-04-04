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

#include "clusterAccelBuilder.h"

#include "cluster.h"
#include "tessellator.h"

#include "../statistics.h"

#include <material/materialCache.h>
#include <scene/scene.h>
#include <scene/vertex.h>
#include <subdivision/SubdivisionSurface.h>

#include <map>
#include <numeric>
#include <span>

#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform_scan.h>
#include <thrust/transform.h>
#include "utils.cuh"
// clang-format on


// packages args for fillClusters function
struct FillClusterInput
{
    std::span<Instance const>                               instances;
    const std::vector<std::unique_ptr<SubdivisionSurface>>& subds;
    const std::vector<size_t>&                              clusterOffsets;
    const CuBuffer<Cluster>&                                clusters;
    const CuBuffer<GridSampler>&                            samplers;
    uint32_t                                                maxSbtIndexValue = 0;

    auto clusterOffset( uint32_t iInstance ) const { return static_cast<uint32_t>( clusterOffsets[iInstance] ); }
    auto clusterCount( uint32_t iInstance ) const
    {
        return static_cast<uint32_t>( clusterOffsets[iInstance + 1] - clusterOffsets[iInstance] );
    }
    std::span<const Cluster> instanceClusters( uint32_t iInstance ) const
    {
        return clusters.subspan( clusterOffset(iInstance), clusterCount(iInstance) );
    }
    const auto& subdSurface( uint32_t iInstance ) const { return *subds[instances[iInstance].meshID]; }
};


ClusterAccelBuilder::ClusterAccelBuilder( OptixDeviceContext context, CUstream stream, const TessellatorConfig& tessConfig )
    : m_context( context )
    , m_stream( stream )
    , m_tessellatorConfig( tessConfig )
    , d_tessellationCounters( std::vector<TessellationCounters>( 1, { 0 } ) )
{
    // Fixed-size allocation for cluster description buffer
    constexpr uint32_t MAX_CLUSTER_COUNT = 1 << 22;
    d_clusters.reserve( MAX_CLUSTER_COUNT );
}

void buildIndirectArgsDeviceBuffer( const CuBuffer<size_t>&                  d_clusterOffsets,
                                    const CUdeviceptr*                       d_clasPtrs,
                                    OptixClusterAccelBuildInputClustersArgs* d_indirectArgs )
{
    // Transform cluster offsets into indirect args
    thrust::transform( thrust::device, d_clusterOffsets.data() + 1, d_clusterOffsets.data() + d_clusterOffsets.size(),
                       d_clusterOffsets.data(), d_indirectArgs,
                       [=] __device__( size_t nextoffset, size_t offset ) -> OptixClusterAccelBuildInputClustersArgs {
                           return OptixClusterAccelBuildInputClustersArgs{ .clusterHandlesCount = (uint32_t)( nextoffset - offset ),
                                                                           .clusterHandlesBufferStrideInBytes = 0,
                                                                           .clusterHandlesBuffer = CUdeviceptr( d_clasPtrs + offset ) };
                       } );
}


void ClusterAccelBuilder::initClusterTemplates( uint32_t maxSbtIndexValue )
{
    // Only build templates once
    if ( m_templateBuffers.d_destData.size() > 0 && m_templateBuffers.maxSbtIndexValue == maxSbtIndexValue ) return;
    
    m_templateBuffers.maxSbtIndexValue = maxSbtIndexValue;

    OptixAccelBufferSizes templateBufferSizes{};
    const uint32_t        maxClusterEdgeSegments = m_tessellatorConfig.maxClusterEdgeSegments;
    const uint32_t        maxNumTemplates        = maxClusterEdgeSegments * maxClusterEdgeSegments;
    const uint32_t        clusterMaxTriangles    = maxClusterEdgeSegments * maxClusterEdgeSegments * 2;
    const uint32_t        clusterMaxVertices     = ( maxClusterEdgeSegments + 1 ) * ( maxClusterEdgeSegments + 1 );

    m_templateBuffers.d_destAddressData.resize( maxNumTemplates );
    m_templateBuffers.d_destSizeData.resize( maxNumTemplates );


    // Compute memory usage for the template build

    OptixClusterAccelBuildInput createTemplateBuildInputs = {};

    createTemplateBuildInputs =
        OptixClusterAccelBuildInput{ .type = OptixClusterAccelBuildType::OPTIX_CLUSTER_ACCEL_BUILD_TYPE_TEMPLATES_FROM_GRIDS,
                                     .grids = { .flags = OptixClusterAccelBuildFlags::OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE,
                                                .maxArgCount  = maxNumTemplates,
                                                .vertexFormat = OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3,
                                                .maxSbtIndexValue = maxSbtIndexValue,
                                                .maxWidth  = maxClusterEdgeSegments,
                                                .maxHeight = maxClusterEdgeSegments } };

    OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( m_context, OptixClusterAccelBuildMode::OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
                                                      &createTemplateBuildInputs, &templateBufferSizes ) );

    m_templateBuffers.d_destData.resize( templateBufferSizes.outputSizeInBytes );

    d_scratch.resize( templateBufferSizes.tempSizeInBytes );


    // Build one template per possible cluster grid size (1x1, 1x2, ... )

    CuBuffer<OptixClusterAccelBuildInputGridsArgs> d_createTemplateArgData( maxNumTemplates );
    {
        const auto qbits = m_tessellatorConfig.quantNBits;

        thrust::counting_iterator<uint32_t> first( 0 );
        thrust::transform( thrust::device, first, first + maxNumTemplates, d_createTemplateArgData.data(),
                           [qbits, maxClusterEdgeSegments] __device__( uint32_t t ) -> OptixClusterAccelBuildInputGridsArgs {

                               uint32_t w = (t / maxClusterEdgeSegments)+1;
                               uint32_t h = (t % maxClusterEdgeSegments)+1;

                               OptixClusterAccelBuildInputGridsArgs args = {};
                               args.basePrimitiveInfo.primitiveFlags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
                               args.positionTruncateBitCount = qbits;
                               args.dimensions[0] = (uint8_t)w;
                               args.dimensions[1] = (uint8_t)h;

                               return args;
                           } );
    }

    OptixClusterAccelBuildModeDesc createTemplateBuildDesc = OptixClusterAccelBuildModeDesc{
        .mode         = OptixClusterAccelBuildMode::OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
        .implicitDest = { .outputBuffer               = m_templateBuffers.d_destData.cu_ptr(),
                          .outputBufferSizeInBytes    = m_templateBuffers.d_destData.size_in_bytes(),
                          .tempBuffer                 = d_scratch.cu_ptr(),
                          .tempBufferSizeInBytes      = d_scratch.size_in_bytes(),
                          .outputHandlesBuffer        = m_templateBuffers.d_destAddressData.cu_ptr(),
                          .outputHandlesStrideInBytes = sizeof( CUdeviceptr ),
                          .outputSizesBuffer          = m_templateBuffers.d_destSizeData.cu_ptr(),
                          .outputSizesStrideInBytes   = sizeof( uint32_t ) } };

    OPTIX_CHECK( optixClusterAccelBuild( m_context, m_stream, &createTemplateBuildDesc,
                                         &createTemplateBuildInputs, d_createTemplateArgData.cu_ptr(), 0 /*argsCount*/,
                                         static_cast<uint32_t>( sizeof( OptixClusterAccelBuildInputGridsArgs ) ) ) );


    // Call GET_SIZES to get a tight estimate of future instantiated CLAS size for each template

    CuBuffer<OptixClusterAccelBuildInputTemplatesArgs> d_getSizesArgData( maxNumTemplates );
    {
        const CUdeviceptr* templateAddresses = m_templateBuffers.d_destAddressData.data();
        thrust::transform( thrust::device, templateAddresses, templateAddresses + maxNumTemplates, d_getSizesArgData.data(),
                           [] __device__( CUdeviceptr templateAddress ) -> OptixClusterAccelBuildInputTemplatesArgs {
                               OptixClusterAccelBuildInputTemplatesArgs args = {};
                               args.clusterTemplate = templateAddress;
                               return args;
                           } );
    }

    OptixClusterAccelBuildInput getSizesBuildInputs = {
        .type      = OptixClusterAccelBuildType::OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES,
        .triangles = { .flags                        = OptixClusterAccelBuildFlags::OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE,
                       .maxArgCount                  = maxNumTemplates,
                       .vertexFormat                 = OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3,
                       .maxSbtIndexValue             = maxSbtIndexValue,
                       .maxUniqueSbtIndexCountPerArg = 1,
                       .maxTriangleCountPerArg       = clusterMaxTriangles,
                       .maxVertexCountPerArg         = clusterMaxVertices,
                       .maxTotalTriangleCount        = 0,
                       .maxTotalVertexCount          = 0,
                       .minPositionTruncateBitCount  = 0 } };

    // Note: New scratch mem size for the GET_SIZES call below
    OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( m_context, OptixClusterAccelBuildMode::OPTIX_CLUSTER_ACCEL_BUILD_MODE_GET_SIZES,
                                                      &getSizesBuildInputs, &templateBufferSizes ) );
    d_scratch.resize( templateBufferSizes.tempSizeInBytes );

    OptixClusterAccelBuildModeDesc getSizesBuildDesc = {
        .mode    = OptixClusterAccelBuildMode::OPTIX_CLUSTER_ACCEL_BUILD_MODE_GET_SIZES,
        .getSize = { .outputSizesBuffer        = m_templateBuffers.d_destSizeData.cu_ptr(),
                     .outputSizesStrideInBytes = sizeof( uint32_t ),
                     .tempBuffer               = d_scratch.cu_ptr(),
                     .tempBufferSizeInBytes    = d_scratch.size_in_bytes() } };

    OPTIX_CHECK( optixClusterAccelBuild( m_context, m_stream, &getSizesBuildDesc, &getSizesBuildInputs,
                                         d_getSizesArgData.cu_ptr(), 0, 0 ) );
}


__global__ void clasSizesKernel( size_t          numClusters,
                                const Cluster*  clusters,
                                const uint32_t* templateSizes,
                                uint32_t        maxClusterEdgeSegments,
                                unsigned long long int* clasOffset,
                                CUdeviceptr*    clasPtrs,
                                uint32_t*       clasSizes)
{
    const uint32_t clusterId = blockIdx.x * blockDim.y + threadIdx.y;
    if( clusterId >= numClusters )
        return;

    const uchar2 es = clusters[clusterId].size;

    assert( es.x <= maxClusterEdgeSegments && es.y <= maxClusterEdgeSegments );

    const uint32_t templateIndex = ( es.x - 1 ) * maxClusterEdgeSegments + ( es.y - 1 );

    clasPtrs[clusterId] = atomicAdd( (unsigned long long int*)clasOffset, templateSizes[templateIndex] );
    clasSizes[clusterId] = templateSizes[templateIndex];
}

size_t computeClasSizes( size_t          numClusters,
                         const Cluster*  d_clusters,
                         const uint32_t* d_templateSizes,
                         uint32_t        maxClusterEdgeSegments,
                         CUdeviceptr     scratch,
                         CUdeviceptr*    d_clasPtrs,
                         uint32_t*       d_clasSizes )
{
    unsigned long long int* d_clasOffset = reinterpret_cast<unsigned long long int*>( scratch );
    cudaMemset( d_clasOffset, 0, sizeof( unsigned long long int ) );

    clasSizesKernel<<<numClusters, 1>>>( numClusters, d_clusters, d_templateSizes,
                                        maxClusterEdgeSegments, d_clasOffset,
                                        d_clasPtrs, d_clasSizes );
    CUDA_SYNC_CHECK();
    unsigned long long int h_outputSizeInBytes = 0u;
    cudaMemcpy( &h_outputSizeInBytes, d_clasOffset, sizeof( unsigned long long int ), cudaMemcpyDeviceToHost );
    CUDA_SYNC_CHECK();

    return h_outputSizeInBytes;
}


void ClusterAccelBuilder::buildClases( ClusterAccels& accel, uint32_t maxSbtIndexValue )
{
    auto& clas_build_timer = stats::clusterAccelSamplers.buildClasTime;
    clas_build_timer.start();

    const uint32_t clusterCount           = static_cast<uint32_t>( d_clusters.size() );
    const uint32_t maxClusterEdgeSegments = m_tessellatorConfig.maxClusterEdgeSegments;

    const uint32_t clusterMaxTriangles = maxClusterEdgeSegments * maxClusterEdgeSegments * 2;
    const uint32_t clusterMaxVertices  = ( maxClusterEdgeSegments + 1 ) * ( maxClusterEdgeSegments + 1 );

    OptixClusterAccelBuildInput clasBuildInputs = {
        .type      = OptixClusterAccelBuildType::OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TEMPLATES,
        .triangles = { .flags                        = OptixClusterAccelBuildFlags::OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE,
                       .maxArgCount                  = clusterCount,
                       .vertexFormat                 = OptixVertexFormat::OPTIX_VERTEX_FORMAT_FLOAT3,
                       .maxSbtIndexValue             = maxSbtIndexValue,
                       .maxUniqueSbtIndexCountPerArg = 1,
                       .maxTriangleCountPerArg       = clusterMaxTriangles,
                       .maxVertexCountPerArg         = clusterMaxVertices,
                       .maxTotalTriangleCount        = clusterMaxTriangles * clusterCount,
                       .maxTotalVertexCount          = clusterMaxVertices * clusterCount,
                       .minPositionTruncateBitCount  = 0 } };

    // Resize scratch buf and build CLASes
        
    OptixAccelBufferSizes bufferSizes = {};
    OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( m_context, OptixClusterAccelBuildMode::OPTIX_CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS,
                                                      &clasBuildInputs, &bufferSizes ) );
    d_scratch.resize( bufferSizes.tempSizeInBytes );

    OptixClusterAccelBuildModeDesc clasBuildDesc = {
        .mode         = OptixClusterAccelBuildMode::OPTIX_CLUSTER_ACCEL_BUILD_MODE_EXPLICIT_DESTINATIONS,
        .explicitDest = { .tempBuffer                 = d_scratch.cu_ptr(),
                          .tempBufferSizeInBytes      = d_scratch.size_in_bytes(),
                          .destAddressesBuffer        = accel.d_clasPtrsBuffer.cu_ptr(),
                          .destAddressesStrideInBytes = sizeof( CUdeviceptr ),
                          .outputHandlesBuffer        = accel.d_clasPtrsBuffer.cu_ptr(),
                          .outputHandlesStrideInBytes = sizeof( CUdeviceptr ),
                          .outputSizesBuffer          = accel.d_clasSizesBuffer.cu_ptr(),
                          .outputSizesStrideInBytes   = sizeof( uint32_t ) } };


    OPTIX_CHECK( optixClusterAccelBuild( m_context, m_stream, &clasBuildDesc, &clasBuildInputs,
                                         d_clasIndirectArgData.cu_ptr(), 0, 0 ) );

    clas_build_timer.stop();
}

void copyGasHandlesToInstanceBuffer( std::span<OptixInstance const> inputInstances, const CUdeviceptr* gasHandles, OptixInstance* outputInstances )
{
    unsigned int n = inputInstances.size();
    thrust::for_each( thrust::device, thrust::make_zip_iterator( thrust::make_tuple( inputInstances.data(), gasHandles, outputInstances) ),
                      thrust::make_zip_iterator( thrust::make_tuple( inputInstances.data() + n, gasHandles + n, outputInstances + n ) ),
                      [] __device__( thrust::tuple<OptixInstance const&, CUdeviceptr, OptixInstance&> t ) {
                          thrust::get<2>( t ) = thrust::get<0>( t );
                          thrust::get<2>( t ).traversableHandle = (OptixTraversableHandle)thrust::get<1>( t );
                      } );
}

inline size_t alignOffset( size_t& cur, size_t align, size_t size )
{
    cur = ((cur + align - 1) / align ) * align;
    size_t ret = cur;
    cur += size;
    return ret;
}

void ClusterAccelBuilder::buildAccel( const Scene& scene, ClusterAccels& accel, ClusterStatistics& stats )
{
    const auto& subdMeshes = scene.getSubdMeshes();
    const auto& instances = scene.getSubdMeshInstances();

    if( subdMeshes.empty() || instances.empty() )
        return;

    const auto totalSubdPatches = scene.totalSubdPatchCount();
    d_gridSamplers.resize( totalSubdPatches );
    
    d_tessellationCounters.set(0);

    d_clusterOffsets.resize( instances.size() + 1 );
    d_clusterOffsets.set(0);

    // We don't know how many clusters will be generated, so expand buffer to max capacity
    d_clusters.resize( d_clusters.capacity() );

    uint32_t patchOffset{0};
    for( size_t i = 0; i < instances.size(); ++i )
    {
        const auto& inst         = instances[i];
        const auto& subd         = *subdMeshes[inst.meshID];
        const auto& localToWorld = inst.localToWorld.matrix3x4();

        uint32_t patchCount{subd.surfaceCount()};
        std::span<GridSampler> samplers = { d_gridSamplers.data() + patchOffset, patchCount };
        tess::computeClusterTiling( m_tessellatorConfig, subd, localToWorld, samplers, d_clusters.span(),
                                    d_tessellationCounters.data() );
        patchOffset += patchCount;
        // Save cluster offset for this instance
        thrust::transform( thrust::device, d_tessellationCounters.data(), d_tessellationCounters.data() + 1,
                           d_clusterOffsets.data() + i + 1,
                           [] __device__( const TessellationCounters& c ) -> size_t { return c.clusters; } );
        CUDA_SYNC_CHECK();
    }

    // Read back cluster offsets per instance, needed for some launches below
    d_clusterOffsets.download( h_clusterOffsets );

    // Read back tessellation counters, needed for resizing cluster buffer
    TessellationCounters counters;
    d_tessellationCounters.download( &counters );

    OTK_REQUIRE( d_clusters.capacity() >= counters.clusters );

    // shrink buffer to fit.  Further tessellation steps assume this size is exact.
    d_clusters.resize( counters.clusters );

    // initialize templates if needed
    const uint32_t maxSbtIndexValue = uint32_t( scene.getMaterialCache().size() );
    initClusterTemplates( maxSbtIndexValue );

    accel.d_clusterShadingData.resize( counters.clusters );
    accel.d_clusterVertexPositions.resize( counters.vertices);
    accel.d_packedClusterVertexNormals.resize( m_tessellatorConfig.enableVertexNormals ? counters.vertices : 0 );

    // Allocate CLASes
    {
        d_scratch.resize( sizeof( unsigned long long int ) );

        accel.d_clasPtrsBuffer.resize( d_clusters.size() );
        accel.d_clasSizesBuffer.resize( d_clusters.size() );

        size_t clasSizeInBytes =
            computeClasSizes( d_clusters.size(), d_clusters.data(), m_templateBuffers.d_destSizeData.data(),
                              m_tessellatorConfig.maxClusterEdgeSegments, d_scratch.cu_ptr(),
                              accel.d_clasPtrsBuffer.data(), accel.d_clasSizesBuffer.data() );

        CUDA_SYNC_CHECK();

        accel.d_clasBuffer.resize( clasSizeInBytes );
        const CUdeviceptr offset = accel.d_clasBuffer.cu_ptr();

        thrust::transform( thrust::device, accel.d_clasPtrsBuffer.data(),
                           accel.d_clasPtrsBuffer.data() + accel.d_clasPtrsBuffer.size(), accel.d_clasPtrsBuffer.data(),
                           [offset] __device__( CUdeviceptr ptr ) -> CUdeviceptr { return offset + ptr; } );
        CUDA_SYNC_CHECK();

    }

    // Fill clusters by evaluating subd vertices
    fillInstanceClusters( { instances, subdMeshes, h_clusterOffsets, d_clusters, d_gridSamplers, maxSbtIndexValue }, accel );

    // Build CLASes for all instances at once
    buildClases( accel, maxSbtIndexValue );


    // Allocate GAS buffers

    OptixAccelBufferSizes gasBufferSizes{};

    size_t maxClustersPerInstance = 0;
    for (size_t i = 0; i < instances.size(); ++i)
        maxClustersPerInstance = std::max( maxClustersPerInstance, h_clusterOffsets[i+1] - h_clusterOffsets[i] );

    const uint32_t maxClusterCount = (uint32_t) d_clusters.size();
    const uint32_t maxClusterCountPerBlas = (uint32_t)maxClustersPerInstance;
    
    size_t indirectArgsBufferOffset = 0;
    {
        OptixClusterAccelBuildInput blasBuildInputs = { .type = OptixClusterAccelBuildType::OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS,
                                                        .clusters = { .flags = OptixClusterAccelBuildFlags::OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE,
                                                                      .maxArgCount = (uint32_t)instances.size(),
                                                                      .maxTotalClusterCount = maxClusterCount,
                                                                      .maxClusterCountPerArg = maxClusterCountPerBlas } };

        OPTIX_CHECK( optixClusterAccelComputeMemoryUsage( m_context, OptixClusterAccelBuildMode::OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
                                                          &blasBuildInputs, &gasBufferSizes ) );
        
        // Expand temp size to include room for indirect args buffer
        size_t totalSize = 0;
        alignOffset( totalSize, 128, gasBufferSizes.tempSizeInBytes );
        indirectArgsBufferOffset = alignOffset( totalSize, 16, instances.size() * sizeof( OptixClusterAccelBuildInputClustersArgs ) );
        gasBufferSizes.tempSizeInBytes = totalSize;
    }

    accel.d_gasBuffer.resize( gasBufferSizes.outputSizeInBytes );

    // Build GASes
    {
        auto& sw = stats::clusterAccelSamplers.buildGasTime;

        d_scratch.resize( gasBufferSizes.tempSizeInBytes );

        accel.d_gasPtrsBuffer.resize( instances.size() );
        accel.d_gasSizesBuffer.resize( instances.size() );

        {
            sw.start();

            OptixClusterAccelBuildInputClustersArgs* d_indirectArgs =
                (OptixClusterAccelBuildInputClustersArgs*)d_scratch.cu_ptr( indirectArgsBufferOffset );

            buildIndirectArgsDeviceBuffer( d_clusterOffsets, accel.d_clasPtrsBuffer.data(),
                                           (OptixClusterAccelBuildInputClustersArgs*)d_indirectArgs );
            OptixClusterAccelBuildInput blasBuildInputs = {
                .type     = OptixClusterAccelBuildType::OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS,
                .clusters = { .flags                 = OptixClusterAccelBuildFlags::OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE,
                              .maxArgCount           = (uint32_t)instances.size(),
                              .maxTotalClusterCount  = maxClusterCount,
                              .maxClusterCountPerArg = maxClusterCountPerBlas } };

            OptixClusterAccelBuildModeDesc blasBuildDesc = {
                .mode         = OptixClusterAccelBuildMode::OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS,
                .implicitDest = { .outputBuffer               = accel.d_gasBuffer.cu_ptr(),
                                  .outputBufferSizeInBytes    = accel.d_gasBuffer.size_in_bytes(),
                                  .tempBuffer                 = d_scratch.cu_ptr(),
                                  .tempBufferSizeInBytes      = d_scratch.size_in_bytes(),
                                  .outputHandlesBuffer        = accel.d_gasPtrsBuffer.cu_ptr(),
                                  .outputHandlesStrideInBytes = sizeof( CUdeviceptr ),
                                  .outputSizesBuffer          = accel.d_gasSizesBuffer.cu_ptr(),
                                  .outputSizesStrideInBytes   = sizeof( uint32_t ) } };

            OPTIX_CHECK( optixClusterAccelBuild(
                m_context, m_stream, &blasBuildDesc, &blasBuildInputs, (CUdeviceptr)d_indirectArgs, 0 /*argsCount*/,
                static_cast<uint32_t>( sizeof( OptixClusterAccelBuildInputClustersArgs ) ) ) );
            sw.stop();

            stats.m_clas_size = accel.d_clasBuffer.size();
            stats.m_num_triangles = counters.triangles;
            stats.m_cluster_data_size = accel.d_clusterShadingData.size_in_bytes()
                                        + accel.d_clasPtrsBuffer.size_in_bytes() + accel.d_clasSizesBuffer.size_in_bytes();
        }

        stats.m_gas_temp_size         = gasBufferSizes.tempSizeInBytes;
        stats.m_gas_size  = gasBufferSizes.outputSizeInBytes;
        stats.m_vertex_buffer_size    = accel.d_clusterVertexPositions.size_in_bytes();
        stats.m_normal_buffer_size    = accel.d_packedClusterVertexNormals.size_in_bytes();
        stats.m_num_clusters          = maxClusterCount;
    }

    std::span<OptixInstance const> d_optixInstances = scene.getOptixInstancesDevice();
    d_instances.resize( instances.size() );
    copyGasHandlesToInstanceBuffer( d_optixInstances, accel.d_gasPtrsBuffer.data(), d_instances.data() );
    buildInstanceAccel( accel, d_instances );

}


void ClusterAccelBuilder::fillInstanceClusters( const FillClusterInput& in, ClusterAccels& accels )
{
    auto& fine_tessellation_timer = stats::clusterAccelSamplers.clusterFillTime;
    fine_tessellation_timer.start();

    const uint32_t totalClusterCount = static_cast<uint32_t>( d_clusters.size() );
    d_clasIndirectArgData.resize( totalClusterCount );

    uint32_t patchOffset{ 0 };
    for( size_t i = 0; i < in.instances.size(); ++i )
    {
        const auto&              subd     = in.subdSurface( i );
        std::span<const Cluster> clusters = in.instanceClusters( i );

        const uint32_t         patchCount = subd.surfaceCount();
        std::span<GridSampler> samplers   = in.samplers.subspan( patchOffset, patchCount );
        patchOffset += patchCount;

        uint32_t                clusterOffset = in.clusterOffsets[i];
        uint32_t                clusterCount  = in.clusterCount( i );
        tess::FillClusterOutput out           = {
                      .vertexPositions     = accels.d_clusterVertexPositions.span(),
                      .packedVertexNormals = accels.d_packedClusterVertexNormals.span(),
                      .clusterShadingData  = accels.d_clusterShadingData.subspan( clusterOffset, clusterCount ),
                      .templateAddresses   = m_templateBuffers.d_destAddressData.span(),
                      .indirectArgData     = d_clasIndirectArgData.subspan( clusterOffset, clusterCount ),
        };
        tess::fillClusters( m_tessellatorConfig, subd, samplers, clusters, in.clusterOffsets[i], out );
    }

    fine_tessellation_timer.stop();
}


void ClusterAccelBuilder::buildInstanceAccel( ClusterAccels& result, const CuBuffer<OptixInstance>& instances )
{

    // Top-level/instance acceleration structure input
    OptixBuildInputInstanceArray instanceArray{ .instances    = instances.cu_ptr(),
                                                .numInstances = static_cast<unsigned int>( instances.size() ) };
    OptixBuildInput              buildInput{ .type = OPTIX_BUILD_INPUT_TYPE_INSTANCES, .instanceArray = instanceArray };
    OptixAccelBuildOptions options{ .buildFlags = OPTIX_BUILD_FLAG_NONE, .operation = OPTIX_BUILD_OPERATION_BUILD };

    // Calculate the sizes required for the acceleration structure buffers
    OptixAccelBufferSizes bufferSizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( m_context, &options, &buildInput, 1, &bufferSizes ) );

    // Update the output allocation, if needed
    result.d_iasBuffer.resize( bufferSizes.outputSizeInBytes );

    d_scratch.resize( bufferSizes.tempSizeInBytes );

    // Build the top-level acceleration structure
    OPTIX_CHECK( optixAccelBuild( m_context, m_stream, &options, &buildInput, 1,
                                  d_scratch.cu_ptr(), bufferSizes.tempSizeInBytes,
                                  result.d_iasBuffer.cu_ptr(), result.d_iasBuffer.size(), &result.iasHandle, nullptr, 0 ) );
}



