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
#include "shadingTypes.h"
#include "pipeline.h"
#include "scene/sceneTypes.h"

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Util/CuBuffer.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>

// clang-format on

struct ClusterAccels;
struct TessellatorConfig;

namespace otk {
    class Camera;
    template <typename PIXEL_FORMAT> class CUDAOutputBuffer;
}

class OptixRenderer
{
public: 

    struct Options {

        otk::CUDAOutputBufferType output_buffer_type = otk::CUDAOutputBufferType::GL_INTEROP;
        uint2 output_target_resolution = { 1024, 1024 };

        bool enable_instancing = false;

        bool print_sbt = false;

        int log_level = 2;

        std::function<void(uint32_t, char const*, char const*)> log_callback =
            [](unsigned int level, const char* tag, const char* message) {
            std::fprintf(stderr, "[%2d][%12s]:%s\n", level, tag, message);
        };
    };

    OptixRenderer( Options const& opts );

    ~OptixRenderer();

    void launchSubframe( CUstream stream = nullptr );
    void resetSubframes();
    void denoise();

    // Returns world space point under a pixel, with depth in .w
    std::optional<float4> pick( uint2 pick_pos, bool yflip = true );


public:
    
    void setColorMode (ColorMode colorMode );
    ColorMode getColorMode() const { return m_params.bound.colorMode; }
    
    void setAOSamples(int n);
    int getAOSamples() const { return m_params.aoSamples; }

    void setMissColor(float3 missColor);
    float3 getMissColor() const { return m_params.missColor; }

    void setWireframe(bool wireframe);
    bool getWireframe() const { return m_params.bound.enableWireframe; }

    void setSurfaceWireframe(bool wireframe);
    bool getSurfaceWireframe() const { return m_params.bound.enableSurfaceWireframe; }

    void setDisplayChannel( GBuffer::Channel channel );
    GBuffer::Channel getDisplayChannel() const { return m_displayChannel; }

    void setMaterials( std::span<MaterialCuda const> materials );

    void setGeometry( const ClusterAccels& accels, const TessellatorConfig& config );

    void setRenderCamera(otk::Camera& camera);
    void setTessellationCamera(const otk::Camera& camera);

    unsigned int getFrameIndex() const { return m_params.frame_index; }

    void saveScreenshot(std::string const& filepath = {});

    void setPipelineNeedsUpdate() { m_pipelinesNeedsUpdate = true; }

public:

    OptixDeviceContext getContext() { return m_context; }

    typedef otk::CUDAOutputBuffer<uchar4> OutputBuffer;

    OutputBuffer& getOutputBuffer();

    CuBuffer<HitResult>& getHitBuffer() { return m_hitBuffer; }

    GBuffer& getGBuffer();

    const Params& getParams() const { return m_params; }

    void blitFramebuffer( CUstream stream );

public:


    void resizeOutputBuffers( uint2 targetsize, CUstream stream = 0 );


  private:

    void createContext(int logLevel);   

    void buildOrUpdatePipelines();

    static void logCallback(uint32_t level, char const* tag, char const* msg, void* data);


private:
    
    Options m_options;
    
    Params  m_params;
    CUdeviceptr m_d_params = 0;

    OptixDeviceContext m_context = nullptr;

    Pipeline m_pipeline;
    bool m_pipelinesNeedsUpdate = true;

    std::span<MaterialCuda const> m_materials;

    std::unique_ptr<otk::CUDAOutputBuffer<uchar4>> m_output_buffer;

    // Things needed to compute motion vecs.  Could be combined with Gbuffer.
    CuBuffer<HitResult> m_hitBuffer;

    GBuffer::Channel m_displayChannel = GBuffer::Channel::DENOISED;
    std::unique_ptr<GBuffer> m_gbuffer;

    CuBuffer<float> m_scratch;
};

