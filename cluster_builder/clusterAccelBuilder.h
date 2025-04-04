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

#include "cluster.h"
#include "tessellatorConfig.h"

#include <OptiXToolkit/ShaderUtil/Aabb.h>
#include <OptiXToolkit/Util/CuBuffer.h>

#include <scene/sceneTypes.h>

#include <memory>
#include <vector>
#include <span>

// clang-format on

class Scene;
class SubdivisionSurface;

struct ClusterShadingData;
struct FillClusterInput;
struct Params;
struct TessellationCounters;

struct ClusterAccels
{
    CuBuffer<unsigned char> d_gasBuffer;
    CuBuffer<unsigned char> d_clasBuffer;
    CuBuffer<unsigned char> d_iasBuffer;

    CuBuffer<CUdeviceptr>  d_clasPtrsBuffer;  // address of each CLAS in the CLAS buffer
    CuBuffer<uint32_t>     d_clasSizesBuffer;
    CuBuffer<CUdeviceptr>  d_gasPtrsBuffer;   // address of each GAS
    CuBuffer<uint32_t>     d_gasSizesBuffer;

    OptixTraversableHandle  iasHandle = 0;

    // -------------------------------------------------------------------------
    // Per-cluster data passed to the OptiX launch
    //
    CuBuffer<ClusterShadingData> d_clusterShadingData;

    // -------------------------------------------------------------------------
    // Vertex Position buffer that we stage into before creating CLASes
    //
    CuBuffer<float3> d_clusterVertexPositions;

    // -------------------------------------------------------------------------
    // Vertex Normal buffer (optional)
    //
    CuBuffer<uint32_t> d_packedClusterVertexNormals;
};

struct ClusterStatistics
{
    size_t   m_clas_size          = 0;
    uint32_t m_gas_size           = 0;
    uint32_t m_gas_temp_size      = 0;
    uint32_t m_num_clusters       = 0;
    uint32_t m_num_triangles      = 0;
    size_t   m_vertex_buffer_size = 0;
    size_t   m_normal_buffer_size = 0;
    size_t   m_cluster_data_size  = 0;

    void print() const
    {
        printf( "CLAS buffer size %zu bytes\n", m_clas_size );
        printf( "GAS buffer size %u bytes\n", m_gas_size );
        printf( "GAS temp buffer size %u bytes\n", m_gas_temp_size );
        printf( "num clusters %u \n", m_num_clusters );
        printf( "num triangles %u \n", m_num_triangles );
        printf( "vertex buffer size %zu bytes\n", m_vertex_buffer_size );
        printf( "normal buffer size %zu bytes\n", m_normal_buffer_size );
        printf( "cluster shading data size %zu bytes\n", m_cluster_data_size );
    }
};

class ClusterAccelBuilder
{
  public:
    ClusterAccelBuilder( OptixDeviceContext context, CUstream stream, const TessellatorConfig& tessConfig );

    void buildAccel( const Scene& scene, ClusterAccels& accels, ClusterStatistics& stats );

    void setTessellatorConfig( const TessellatorConfig& config ) { m_tessellatorConfig = config; }

    void initClusterTemplates( uint32_t maxSbtIndexValue );

  protected:

    void fillInstanceClusters( const FillClusterInput& in, ClusterAccels& accels );

    void buildClases( ClusterAccels& accels, uint32_t maxSbtIndexValue );

    void buildInstanceAccel( ClusterAccels& accels, const CuBuffer<OptixInstance>& instances );

    OptixDeviceContext m_context = 0;
    CUstream           m_stream  = 0;

    TessellatorConfig              m_tessellatorConfig;
    CuBuffer<TessellationCounters> d_tessellationCounters;

    CuBuffer<Cluster>              d_clusters;

    CuBuffer<GridSampler>          d_gridSamplers;
    CuBuffer<size_t>               d_clusterOffsets;
    std::vector<size_t>            h_clusterOffsets;

    CuBuffer<unsigned char>        d_scratch;  // generic temp buffer


    struct TemplateBuffers
    {
        uint32_t                maxSbtIndexValue = 0;
        CuBuffer<unsigned char> d_destData;
        CuBuffer<CUdeviceptr>   d_destAddressData;
        CuBuffer<uint32_t>      d_destSizeData;
    };
    TemplateBuffers m_templateBuffers; // Buffers used to create templates.  They are created once but need to be persistent throughout the app's run time.

    // indirect arg data used in instantiate templates. This is a temp buffer but it's declared a member variable to avoid repeated allocations.
    CuBuffer<OptixClusterAccelBuildInputTemplatesArgs> d_clasIndirectArgData;

    // Temp input to IAS build
    CuBuffer<OptixInstance>    d_instances;
};

