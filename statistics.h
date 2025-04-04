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

#pragma once

#include "profiler/stopwatch.h"
#include <OptiXToolkit/Util/CuBuffer.h>

#include <profiler/profiler.h>

#include <string>
#include <map>
#include <mutex>

namespace otk {
    class ImGuiRenderer;
}

namespace stats {

using CPUTimer = Profiler::CPUTimer;
using GPUTimer = Profiler::GPUTimer;

//
// SubD stats
//

struct TopologyMapStats {

    // hashmap
    float  pslMean = 0.f;
    size_t hashCount = 0;
    size_t addressCount = 0;
    float  loadFactor = 0.f;

    // plans
    size_t   plansCount = 0;
    size_t   plansByteSize = 0;

    uint32_t regularFacePlansCount = 0;

    uint32_t maxFaceSize = 0;
    uint32_t sharpnessCount = 0;
    float    sharpnessMax;

    // patch points
    uint32_t stencilCountMin = 0;
    uint32_t stencilCountMax = 0;
    float stencilCountAvg = 0;
    std::vector<uint32_t> stencilCountHistogram;
};

struct SurfaceTableStats {

    std::string name;

    size_t byteSize = 0;        
    size_t surfaceCount = 0;

    uint32_t irregularFaceCount = 0;
    uint32_t maxValence = 0;
    uint32_t maxFaceSize = 0;
                                        // | boundaries | stencils  | creases |
    uint32_t holesCount = 0;            // |            |           |         |
    uint32_t bsplineSurfaceCount = 0;   // |            |           |         |
    uint32_t regularSurfaceCount = 0;   // |     x      |           |         |
    uint32_t isolationSurfaceCount = 0; // |     X      |     X     |         |
    uint32_t sharpSurfaceCount = 0;     // |     X      |     X     |    X    |

    float sharpnessMax = 0.f;
    uint32_t infSharpCreases = 0;

    uint32_t stencilCountMin = ~uint32_t(0);
    uint32_t stencilCountMax = 0;
    float stencilCountAvg = 0;
    std::vector<uint32_t> stencilCountHistogram;

    // buffer storing per-surface topology 'quality' factor
    CuBuffer<uint8_t> topologyQuality;

    std::vector<std::string> topologyRecommendations;
    
    bool isCatmarkTopology( float* ratio = nullptr ) const {
        // guess if the user passed a triangles mesh (ie. not a subd model)
        float _ratio =  float( irregularFaceCount ) / float( surfaceCount );
        if( ratio )
            *ratio = _ratio;
        return _ratio < .25f;
    }

    void buildTopologyRecommendations();

    void buildRecommendationsUI( otk::ImGuiRenderer& renderer ) const;

    void buildUI( otk::ImGuiRenderer& renderer, uint32_t imguiID ) const;
};


//
// General stats
//

struct FrameSamplers
{
    std::string name = "Frame";

    Sampler<float, Profiler::BENCH_FRAME_COUNT> cpuFrameTime = {.name = "CPU/frame (ms)"};

    GPUTimer& gpuFrameTime    = Profiler::initTimer<GPUTimer>( "GPU/frame (ms)" );
    GPUTimer& gpuRenderTime   = Profiler::initTimer<GPUTimer>( "GPU/trace (ms)" );
    GPUTimer& gpuBlitTime     = Profiler::initTimer<GPUTimer>( "GPU/blit (ms)" );

    GPUTimer& motionVecTime = Profiler::initTimer<GPUTimer>("GPU/motionVecPass (ms)");

    void buildUI( otk::ImGuiRenderer& renderer ) const;
};
extern FrameSamplers frameSamplers;

struct ClusterAccelSamplers
{
    std::string name = "AccelBuilder";

    GPUTimer& buildClasTime  = Profiler::initTimer<GPUTimer>( "GPU/CLAS build (ms)" );
    GPUTimer& buildGasTime  = Profiler::initTimer<GPUTimer>( "GPU/GAS build (ms)" );
    GPUTimer& clusterTilingTime = Profiler::initTimer<GPUTimer>( "GPU/Cluster Tiling (ms)" );
    GPUTimer& clusterFillTime = Profiler::initTimer<GPUTimer>( "GPU/Cluster Fill (ms)" );

    Sampler<uint32_t> numClusters = { .name = "Cluster count", };
    Sampler<uint32_t> numTriangles = { .name = "Triangle count", };

    Sampler<uint32_t> gasSize = { .name = "GAS size", };

    void buildUI( otk::ImGuiRenderer& renderer ) const;
};
extern ClusterAccelSamplers clusterAccelSamplers;

struct EvaluatorSamplers
{
    std::string name = "Evaluator";

    size_t indexBufferSize = 0;
    size_t vertCountBufferSize = 0;

    TopologyMapStats topologyMapStats;

    bool hasBadTopology = false;
    
    size_t surfaceCountTotal = 0;
    size_t surfaceTablesByteSizeTotal = 0;

    std::vector<SurfaceTableStats> surfaceTableStats;

    void buildUI( otk::ImGuiRenderer& renderer ) const;
};
extern EvaluatorSamplers evaluatorSamplers;

struct MemUsageSamplers
{
    std::string name = "Memory";

    size_t bcSize = 0;

    Sampler<size_t> gasSize = { .name = "GAS size", };
    Sampler<size_t> gasTempSize = { .name = "GAS Temp Buffer size", };
    Sampler<size_t> clasSize = { .name = "CLAS size", };
    Sampler<size_t> vertexBufferSize = { .name = "Vertex Buffer size", };
    Sampler<size_t> normalBufferSize = { .name = "Normal Buffer size", };
    Sampler<size_t> clusterShadingDataSize = { .name = "Cluster Data Buffer size", };

    void buildUI( otk::ImGuiRenderer& renderer ) const;
};
extern MemUsageSamplers memUsageSamplers;


}  // end namespace stats
