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
#include "optixRenderer.h"
#include "denoiserstub.h"

#include <statistics.h>
#include <utils.h>

#include <cluster_builder/clusterAccelBuilder.h>

#include <texture/textureCuda.h>
#include <texture/textureCache.h>
#include <texture/texture.h>
#include <texture/tinyexr/tinyexr.h>

#include <OptiXToolkit/Gui/Camera.h>

#include <optix_function_table_definition.h>


#include <cstdio>
#include <filesystem>
#ifndef __GNUC__
#include <format>
#endif
// clang-format on

bool operator == (float3 const& a, float3 const& b)
{
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}

uint2 operator/( const uint2& v, float scale )
{
    return { static_cast<unsigned int>( static_cast<float>( v.x ) / scale ),
             static_cast<unsigned int>( static_cast<float>( v.y ) / scale ) };
}

// First N Halton base2 and base3 points.  We cycle through these for camera jitter similar to TAA
constexpr static int NumHaltonPoints = 36;
const static float2  HaltonPoints[NumHaltonPoints] = {
    { 0.000000f, 0.000000f }, { 0.500000f, 0.333333f }, { 0.250000f, 0.666667f }, { 0.750000f, 0.111111f },
    { 0.125000f, 0.444444f }, { 0.625000f, 0.777778f }, { 0.375000f, 0.222222f }, { 0.875000f, 0.555556f },
    { 0.062500f, 0.888889f }, { 0.562500f, 0.037037f }, { 0.312500f, 0.370370f }, { 0.812500f, 0.703704f },
    { 0.187500f, 0.148148f }, { 0.687500f, 0.481481f }, { 0.437500f, 0.814815f }, { 0.937500f, 0.259259f },
    { 0.031250f, 0.592593f }, { 0.531250f, 0.925926f }, { 0.281250f, 0.074074f }, { 0.781250f, 0.407407f },
    { 0.156250f, 0.740741f }, { 0.656250f, 0.185185f }, { 0.406250f, 0.518519f }, { 0.906250f, 0.851852f },
    { 0.093750f, 0.296296f }, { 0.593750f, 0.629630f }, { 0.343750f, 0.962963f }, { 0.843750f, 0.012346f },
    { 0.218750f, 0.345679f }, { 0.718750f, 0.679012f }, { 0.468750f, 0.123457f }, { 0.968750f, 0.456790f },
    { 0.015625f, 0.790123f }, { 0.515625f, 0.234568f }, { 0.265625f, 0.567901f }, { 0.765625f, 0.901235f },
};


OptixRenderer::OptixRenderer( Options const& opts )
    : m_options( opts )
{   
    createContext( opts.log_level );
 
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &m_d_params ), sizeof( Params ) ) );

}

OptixRenderer::~OptixRenderer()
{
    m_output_buffer.reset();

    m_pipeline.cleanup();

    CUDA_CHECK_NOTHROW( cudaFree( reinterpret_cast<void*>( m_d_params ) ) );

    OPTIX_CHECK( optixDeviceContextDestroy( m_context ) );
}

void OptixRenderer::createContext(int logLevel)
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    CUcontext cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK( optixInit( ) );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = logCallback;
    options.logCallbackData = &m_options;
    options.logCallbackLevel = m_options.log_level;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;

    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &m_context ) );
}

void OptixRenderer::logCallback(uint32_t level, char const* tag, char const* msg, void* data)
{
    if (Options const* opts = reinterpret_cast<Options const*>(data))
        opts->log_callback(level, tag, msg);
}

void OptixRenderer::buildOrUpdatePipelines()
{
    if (m_pipelinesNeedsUpdate)
    {
        m_pipeline.buildOrUpdate(m_context, m_params, m_materials, m_options.enable_instancing, m_options.print_sbt);

        resetSubframes();

        m_pipelinesNeedsUpdate = false;
    }
}

OptixRenderer::OutputBuffer& OptixRenderer::getOutputBuffer()
{
    if( !m_output_buffer )
    {
        uint2 targetResolution = m_options.output_target_resolution;
        m_output_buffer =
            std::make_unique<OutputBuffer>( m_options.output_buffer_type, targetResolution.x, targetResolution.y );
        resizeOutputBuffers( targetResolution );
    }

    return *m_output_buffer;
}


GBuffer& OptixRenderer::getGBuffer()
{
    if ( !m_gbuffer )
    {
        const uint2 targetsize = m_options.output_target_resolution;
        const uint2 rendersize = targetsize;

        m_gbuffer = std::make_unique<GBuffer>( rendersize, targetsize );

        m_hitBuffer.resize( rendersize.x * rendersize.y );
        m_hitBuffer.set(0);
        m_params.hit_buffer = m_hitBuffer.data();
    }
    return *m_gbuffer;
}


void OptixRenderer::resizeOutputBuffers( uint2 targetsize, CUstream stream )
{
    if( targetsize.x == 0 || targetsize.y == 0 )
        return;

    OTK_REQUIRE( m_output_buffer );
    m_output_buffer->resize( targetsize.x, targetsize.y );

    if( !m_gbuffer || ( targetsize != m_gbuffer->m_targetsize ) )
    {
        // since we don't have DLSS yet, render res and target res are the same
        const uint2 rendersize = targetsize;

        m_hitBuffer.resize( rendersize.x * rendersize.y );
        m_hitBuffer.set(0);
        m_params.hit_buffer = m_hitBuffer.data();
        m_gbuffer.reset( new GBuffer(rendersize, targetsize) );
    }

    resetSubframes();
}

void OptixRenderer::launchSubframe( CUstream stream )
{
    buildOrUpdatePipelines();

    assert(m_pipeline.pipeline);

    auto& buffer = getOutputBuffer();

    m_params.frame_buffer = buffer.map();

    OTK_REQUIRE( m_gbuffer );

    m_params.aovAlbedo = m_gbuffer->m_albedo;
    m_params.aovNormals = m_gbuffer->m_normals;
    m_params.aovColor = m_gbuffer->m_color;
    m_params.aovDepth = m_gbuffer->m_depth;
    m_params.aovDepthHires = m_gbuffer->m_depthHires;

    // DLSS-style uniform jitter per frame
    // m_params.jitter =  HaltonPoints[m_params.frame_index % NumHaltonPoints];

    // ... or per sub frame when DLSS isn't hooked up
    m_params.jitter =  HaltonPoints[m_params.subframe_index % NumHaltonPoints];

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(m_d_params), &m_params, sizeof(Params), cudaMemcpyHostToDevice, stream));

    const uint2 size = m_gbuffer->m_rendersize;

    stats::frameSamplers.gpuRenderTime.start();
    OPTIX_CHECK(optixLaunch(m_pipeline.pipeline, stream, m_d_params, sizeof(Params), &m_pipeline.sbt, size.x, size.y, 1 /*depth*/));
    stats::frameSamplers.gpuRenderTime.stop();

    CUDA_SYNC_CHECK();

    ++m_params.subframe_index;
    ++m_params.frame_index;

    buffer.unmap();
}

void OptixRenderer::resetSubframes()
{
    m_params.subframe_index = 0;
}

void OptixRenderer::denoise()
{
    OTK_REQUIRE( m_gbuffer );
    ::denoise( *m_gbuffer, m_params.subframe_index-1 );
}

std::optional<float4> OptixRenderer::pick( uint2 pick_pos, bool yflip )
{
    if ( !m_gbuffer ) return {};

    // Read linear depth value from GBuffer

    m_scratch.resize(1);
    uint2 pixel = {pick_pos.x, yflip ? m_gbuffer->m_targetsize.y - pick_pos.y : pick_pos.y};
    m_gbuffer->pickdepth(pixel, m_scratch.data());
    float depth = 0.0f;
    m_scratch.download(&depth);

    float4 pick_result = {0.f, 0.f, 0.f, depth};

    // Unproject linear depth to world space position

    if ( std::isfinite( depth ) )
    {
        uint2 dims = m_gbuffer->m_targetsize;
        float4 v_ndc  = make_float4( ( make_float2( pixel ) / make_float2( float(dims.x), float(dims.y) ) ) * 2.f - 1.f, 0.f, 1.f );
        float4 v_clip = v_ndc * depth;
        otk::Matrix4x4 projInv = m_params.projectionMatrix.inverse();
        float4 v_view = projInv * v_clip;
        otk::Matrix4x4 viewInv = m_params.viewMatrix.inverse();
        float4 vw     = viewInv * make_float4( v_view.x, v_view.y, v_view.z, 1.0f );

        pick_result = { vw.x, vw.y, vw.z, depth };
    } 

    return pick_result;
}

void OptixRenderer::setColorMode(ColorMode colorMode) 
{
    m_pipelinesNeedsUpdate |= (colorMode != m_params.bound.colorMode);
    m_params.bound.colorMode = colorMode;
    resetSubframes();
}

void OptixRenderer::setAOSamples(int aoSamples)
{
    m_params.aoSamples = aoSamples;
}


void OptixRenderer::setMissColor(float3 missColor)
{
    m_params.missColor = missColor;
    resetSubframes();
}

void OptixRenderer::setWireframe(bool wireframe)
{
    m_pipelinesNeedsUpdate |= (wireframe != m_params.bound.enableWireframe);
    m_params.bound.enableWireframe = wireframe;
    resetSubframes();
}   

void OptixRenderer::setSurfaceWireframe(bool wireframe)
{
    m_pipelinesNeedsUpdate |= (wireframe != m_params.bound.enableSurfaceWireframe);
    m_params.bound.enableSurfaceWireframe = wireframe;
    resetSubframes();
}

void OptixRenderer::setDisplayChannel( GBuffer::Channel channel )
{
    m_displayChannel = channel;
}


void OptixRenderer::setMaterials( std::span<MaterialCuda const> materials )
{
    m_pipelinesNeedsUpdate |= ( materials.data() != m_materials.data() );

    m_materials = materials;
    
    resetSubframes();
}

void OptixRenderer::setGeometry( const ClusterAccels& accels, const TessellatorConfig& config )
{
    m_params.handle = accels.iasHandle;
    m_params.cluster_shading_data = accels.d_clusterShadingData.data();
    m_params.cluster_pattern      = config.cluster_pattern;

    m_params.clusterVertexPositions = accels.d_clusterVertexPositions.data();

    if( config.enableVertexNormals )
        m_params.packedClusterVertexNormals = accels.d_packedClusterVertexNormals.data();
    else
        m_params.packedClusterVertexNormals = nullptr;
}

void OptixRenderer::setRenderCamera(otk::Camera& camera)
{
    float3 eye = camera.getEye();
    auto const& [u, v, w] = camera.getBasis();

    if (eye == m_params.eye && m_params.U == u && m_params.V == v && m_params.W == w)
        return;

    m_params.eye = camera.getEye();;
    m_params.U = u;
    m_params.V = v;
    m_params.W = w;
    m_params.viewProjectionMatrix = camera.getViewProjectionMatrix();
    m_params.viewMatrix = camera.getViewMatrix();
    m_params.projectionMatrix = camera.getProjectionMatrix();
    resetSubframes();
}

void OptixRenderer::setTessellationCamera(const otk::Camera& camera)
{
    m_params.tessViewProjectionMatrix = camera.getProjectionMatrix() * camera.getViewMatrix();
}

void OptixRenderer::saveScreenshot(std::string const& filepath)
{
    static char const base_name[] = "screenshot_";

    auto writeImage = [this](char const* filepath) {

        OutputBuffer& output_buffer = getOutputBuffer();

        otk::ImageBuffer buffer;
        buffer.data = output_buffer.getHostPointer();
        buffer.width = output_buffer.width();
        buffer.height = output_buffer.height();
        buffer.pixel_format = otk::BufferImageFormat::UNSIGNED_BYTE4;

        otk::saveImage(filepath, buffer, false);

        std::fprintf(stdout, "saved screenshot: %s\n", filepath);
    };

    if (filepath.empty())
    {
        // Avoid overwriting an existing screenshot : scan the default output directory
        // for existing files with pattern 'screenshot_xxxx.png' to find the highest index.
        static int index = [&filepath]() -> int {
            int index = -1;
            namespace fs = std::filesystem;
            for (auto it : fs::directory_iterator(fs::current_path()))
            {
                if (it.path().extension() != ".png")
                    continue;
                std::string filename = it.path().filename().generic_string();
                if (std::strstr(filename.c_str(), base_name) != filename.c_str())
                    continue;
                index = std::max(index, std::atoi(filename.c_str() + std::size(base_name) - 1));
            }
            return index + 1;
        }();

#ifndef __GNUC__
        writeImage(std::format("screenshot_{:04d}.png", index++).c_str());
#else
        char buf[32];
        std::snprintf(buf, std::size(buf), "screenshot_%04d.png", index++);
        writeImage(buf);
#endif
    }
    else
        writeImage(filepath.c_str());
}

void 
OptixRenderer::blitFramebuffer( CUstream stream )
{
    OTK_ASSERT( m_gbuffer );

    stats::frameSamplers.gpuBlitTime.start();

    uchar4* output     = getOutputBuffer().map();
    uint2   outputSize = m_gbuffer->m_targetsize;

    m_gbuffer->blit( m_displayChannel, output, outputSize, stream );

    CUDA_SYNC_CHECK();
    getOutputBuffer().unmap();

    stats::frameSamplers.gpuBlitTime.stop();    
}

