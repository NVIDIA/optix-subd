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

#include "optixSubdApp.h"
#include "optixRenderer.h"
#include "statistics.h"

#include <scene/shapeUtils.h>


#include <material/materialCache.h>
#include <material/materialCuda.h>
#include <motionvec/motionvec.h>
#include <wireframe/wireframe.h>
#include <texture/textureCache.h>
#include <scene/scene.h>

#include <chrono>
#include <filesystem>
#include <sstream>
#include <thread>
#include <memory>
namespace fs = std::filesystem;
// clang-format on


fs::path getDirectoryWithExecutable()
{
    char path[2048] = { 0 };
#ifdef _WIN32
    if (GetModuleFileNameA(nullptr, path, std::size(path)) == 0)
        return {};
#else   // _WIN32
    // /proc/self/exe is mostly linux-only, but can't hurt to try it elsewhere
    if (readlink("/proc/self/exe", path, std::size(path)) <= 0)
        if (!getcwd(path, std::size(path)))
            return {};
#endif  // _WIN32

    std::filesystem::path result = path;
    
    if (result = result.parent_path(); fs::is_directory(result))
        return result;

    return {};
}

// search "up-ward" from the start path for a given directory name
static fs::path findDir(fs::path const& startPath, fs::path const& dirname, int maxDepth)
{
    std::filesystem::path searchPath = "";

    for (int depth = 0; depth < maxDepth; depth++)
    {
        fs::path currentPath = startPath / searchPath / dirname;

        if (fs::is_directory(currentPath))
            return currentPath.lexically_normal();

        searchPath = ".." / searchPath;
    }
    return {};
}

static fs::path findMediaFolder(fs::path const& startdir, char const* dirname, int maxdepth = 5)
{
    fs::path mediapath;
    try
    {
        fs::path start = fs::canonical(startdir).parent_path();
        mediapath = findDir(start, dirname, maxdepth);
    }
    catch (std::exception const& e)
    {
        fprintf(stderr, "%s\n", e.what());
    }
    return mediapath;
}

static inline const MaterialCuda* getMaterials( const Scene& scene )
{
    return scene.getMaterialCache().getDeviceData().data();
}

OptixSubdApp::OptixSubdApp( int argc, char const* const* argv )
{
    m_binaryPath = getDirectoryWithExecutable().lexically_normal();

    if( !m_binaryPath.empty() )
        m_mediaPath = findMediaFolder(  m_binaryPath, "assets" );

    m_args.parse( argc, argv );

    fs::path scenePath = m_args.meshInputFile;

    auto context = getOptixRenderer().getContext();

    // Check for cluster support
    int clustersSupported = 0;
    OPTIX_CHECK( optixDeviceContextGetProperty( context, OPTIX_DEVICE_PROPERTY_CLUSTER_ACCEL,
                                                &clustersSupported, sizeof( int ) ) );

    OPTIX_CHECK( optixDeviceContextGetProperty( context, OPTIX_DEVICE_PROPERTY_LIMIT_MAX_STRUCTURED_GRID_RESOLUTION,
                                                &m_maxClusterEdgeSegments, sizeof( uint32_t ) ) );

    OTK_REQUIRE_MSG( clustersSupported && m_maxClusterEdgeSegments > 0, "device does not support clusters" );

    loadScene( scenePath.lexically_normal().generic_string(), getMediaPath().generic_string(), {0, 0});

    // re-apply the CLI settings to override any value potentially set by the scene
    m_args.parse( argc, argv );

    OTK_REQUIRE_MSG( m_scene, "no scene loaded on start" );

    resetCamera();

    m_prevCamera = m_camera;

    createAccelBuilder();

    OptixRenderer& renderer = getOptixRenderer();
    renderer.setRenderCamera( m_camera );
    renderer.setTessellationCamera( m_tessellationCamera );
}

OptixSubdApp::~OptixSubdApp() {}

bool OptixSubdApp::interactiveMode() const
{
    return m_args.outfile.empty();
}

void OptixSubdApp::loadScene( 
    std::string const& filepath, const std::string& mediapath, int2 frameRange )
{
    auto& renderer = getOptixRenderer();
    
    m_args.sceneArgs() = {};

    m_scene.reset();
    if( m_scene = Scene::create( filepath, mediapath, frameRange, m_args ) )
    {
        renderer.setMaterials( m_scene->getMaterialCache().getDeviceData() );

        m_motionVecPass = std::make_unique<MotionVecPass>();

        if( m_wireframePass )
            m_wireframePass = std::make_unique<WireframePass>( *m_scene );

    }
    m_accelBuilderNeedsUpdate = true;
    resetCamera();
}

// Call this after GL initilization in the main loop
void OptixSubdApp::setupGL()
{
    OTK_REQUIRE( m_scene && !m_wireframePass );
    m_wireframePass = std::make_unique<WireframePass>( *m_scene ); 
}

static TessellatorConfig makeTessellatorConfig( const Args& args, const OptixRenderer& renderer, const MaterialCuda* materials, const otk::Camera* camera, uint32_t maxClusterEdgeSegments )
{
    const TessellatorConfig tessConfig {
        .enableVertexNormals          = args.enableVertexNormals,
        .fineTessellationRate         = args.fineTessellationRate,
        .coarseTessellationRate       = args.coarseTessellationRate,
        .enableFrustumVisibility      = args.enableFrustumVisibility,
        .enableBackfaceVisibility     = args.enableBackfaceVisibility,
        .viewport_size                = args.targetResolution,
        .maxClusterEdgeSegments       = maxClusterEdgeSegments,
        .quantNBits                   = args.quantNBits,
        .cluster_pattern              = args.enableSlantedTess ? ClusterPattern::SLANTED : ClusterPattern::REGULAR,
        .displacement_scale           = args.dispScale,
        .displacement_bias            = args.dispBias,
        .displacement_filter_scale    = args.dispFilterScale,
        .displacement_filter_mip_bias = args.dispFilterMipBias,
        .camera                       = camera,
        .materials                    = materials,
    };
    
    return tessConfig;
}


// animate, build accel, and render
void OptixSubdApp::renderSubframe()
{
    auto& renderer = getOptixRenderer();

    if ( m_args.enableAdaptiveTess )
        m_tessellationCamera = m_camera;

    const TessellatorConfig tessConfig = makeTessellatorConfig( m_args, renderer, getMaterials( *m_scene ), &m_tessellationCamera, m_maxClusterEdgeSegments );

    m_accelBuilder->setTessellatorConfig( tessConfig );
    m_accelBuilder->buildAccel( *m_scene, m_accels, m_build_stats );

    renderer.setGeometry( m_accels, tessConfig );

    GBuffer& gbuffer = renderer.getGBuffer();

    // Map any GL interop channels
    gbuffer.map();

    renderer.launchSubframe( m_stream );

    m_motionVecPass->run( *m_scene, m_args.dispScale, m_args.dispBias, m_camera, m_prevCamera, renderer.getParams().jitter,
                          renderer.getHitBuffer(), gbuffer );

    renderer.denoise();

    renderer.blitFramebuffer( m_stream );

    // Unmap any GL interop channels
    renderer.getGBuffer().unmap();

}

void OptixSubdApp::drawGL()
{
    if ( m_wireframePass && m_wireframePassEnabled )
    {
        auto& renderer = getOptixRenderer();
        const auto& depth = renderer.getGBuffer().m_depthHires;
        uint2 res = m_args.targetResolution;
        m_wireframePass->run( *m_scene, m_camera, {0.5f, 0.5f}, res.x, res.y, depth );
    }
}


void OptixSubdApp::renderBatchSubframes()
{
    assert( !m_args.outfile.empty() );

    auto& profiler = Profiler::get();


    for( int frame = 0; frame < m_args.frames; frame++ )
    {
        profiler.frameStart(std::chrono::steady_clock::now());

        stats::frameSamplers.gpuFrameTime.start();
        renderSubframe();
        stats::frameSamplers.gpuFrameTime.stop();

        profiler.frameEnd();
        profiler.frameResolve();
    }

    // print final stats for the *last* run accel build
    m_build_stats.print();

    stats::frameSamplers.gpuFrameTime.print();

    std::printf( "Average frame time %.4f ms\n", stats::frameSamplers.gpuFrameTime.average() );
    std::printf( "Average render time %.4f ms\n", stats::frameSamplers.gpuRenderTime.average() );
    std::printf( "Cluster Tiling Duration %.4f ms\n", stats::clusterAccelSamplers.clusterTilingTime.average() );
    std::printf( "Cluster Fill Duration %.4f ms\n", stats::clusterAccelSamplers.clusterFillTime.average() );
    std::printf( "Build CLAS Duration %.4f ms\n", stats::clusterAccelSamplers.buildClasTime.average() );
    std::printf( "Build GAS Duration %.4f ms\n", stats::clusterAccelSamplers.buildGasTime.average() );

    auto& renderer = getOptixRenderer();
    renderer.saveScreenshot( m_args.outfile );
}

void OptixSubdApp::renderInteractiveSubframe( float animTime, float frameRate )
{
    using namespace std::chrono;

    auto& profiler = Profiler::get();

    m_prevFrameStart = m_frameIndex > 0 ? m_currFrameStart : steady_clock::now();
    m_currFrameStart = steady_clock::now();

    if (profiler.isRecording())
        stats::frameSamplers.cpuFrameTime.push_back(duration<float, std::milli>(m_currFrameStart - m_prevFrameStart).count());

    // stop camera animation on user-input (re-engage with resetCamera())
    m_cameraCanAnimate &= !m_trackball.animate( getCPUFrameTime() / 1000.f );

    profiler.frameStart(m_currFrameStart);  


    if ( m_animationTime != animTime )
    {
        // Update scene to current time
        auto& renderer = getOptixRenderer();
        m_scene->animate( FrameTime{ animTime, frameRate } );
        renderer.resetSubframes();
        m_animationTime = animTime;
    } 
    else 
    {
        // Stop caching animated vertices when playback is off
        m_scene->clearMotionCache();
    }
    updateCamera( m_scene->getView(), true );

    stats::frameSamplers.gpuFrameTime.start();

    if (m_accelBuilderNeedsUpdate)
    {
        createAccelBuilder();
    }

    renderSubframe();

    stats::frameSamplers.gpuFrameTime.stop();

    m_prevCamera = m_camera; 


    profiler.frameEnd(); // no more stopwatch start/stop aFter this point!

    // capture statistics
    if( profiler.isRecording() )
    {
        stats::clusterAccelSamplers.numClusters.push_back( m_build_stats.m_num_clusters );
        stats::clusterAccelSamplers.numTriangles.push_back( m_build_stats.m_num_triangles );
        stats::clusterAccelSamplers.gasSize.push_back( m_build_stats.m_gas_size );

        stats::memUsageSamplers.gasSize.push_back( m_build_stats.m_gas_size );
        stats::memUsageSamplers.gasTempSize.push_back( m_build_stats.m_gas_temp_size );
        stats::memUsageSamplers.clasSize.push_back( m_build_stats.m_clas_size );
        stats::memUsageSamplers.normalBufferSize.push_back( m_build_stats.m_normal_buffer_size );
        stats::memUsageSamplers.vertexBufferSize.push_back( m_build_stats.m_vertex_buffer_size );
        stats::memUsageSamplers.clusterShadingDataSize.push_back( m_build_stats.m_cluster_data_size );
    }

    ++m_frameIndex;
}

void OptixSubdApp::setOutputBufferTargetSize( uint2 targetsize )
{
    getOptixRenderer().resizeOutputBuffers( targetsize );

    m_camera.setAspectRatio( float( targetsize.x ) / float( targetsize.y ) );
    m_args.targetResolution = targetsize;
}

int2 OptixSubdApp::getOutputBufferTargetSize() const
{
    return make_int2( (int)m_args.targetResolution.x, (int)m_args.targetResolution.y );
}


void OptixSubdApp::updateCamera( const View* view, bool interactive )
{
    if( view && view->isAnimated && m_cameraCanAnimate  )
    {
        m_camera.setEye( view->position );
        if( view->rotation )
            m_camera.setRotation( *view->rotation );
        else
        {
            m_camera.setLookat( view->lookat );
            m_camera.setUp( view->up );
        }
        m_camera.setFovY( view->fov );

        // update far clip plane
        otk::Aabb aabb = m_scene->getAttributes().aabb;
        aabb.include( m_camera.getEye() );
        m_camera.setFar( 1.1f * length( aabb.extent() ) );
    }
    else
    {
        if( interactive )
            m_trackball.animate( getCPUFrameTime() / 1000.f );
    }
    
    OptixRenderer& renderer = getOptixRenderer();
    renderer.setRenderCamera( m_camera );
    renderer.setTessellationCamera( m_tessellationCamera );
}


OptixRenderer& OptixSubdApp::getOptixRenderer()
{
    if( !m_optixRenderer )
    {
        // app currently requires GL interop
        const otk::CUDAOutputBufferType outputBufferType = otk::CUDAOutputBufferType::GL_INTEROP;

        OptixRenderer::Options rendererOptions{
            .output_buffer_type = outputBufferType,
            .output_target_resolution = m_args.targetResolution,
            .enable_instancing  = true,
            .print_sbt          = false,
            .log_level          = m_args.logLevel,
        };

        m_optixRenderer = std::make_unique<OptixRenderer>( rendererOptions );
    }
    return *m_optixRenderer;
}

MotionVecPass& OptixSubdApp::getMotionVecPass()
{
    return *m_motionVecPass;
}

void OptixSubdApp::createAccelBuilder()
{
    OptixRenderer& renderer = getOptixRenderer();

    m_tessellationCamera = m_camera;

    const auto tessConfig = makeTessellatorConfig( m_args, renderer, getMaterials( *m_scene ), &m_tessellationCamera, m_maxClusterEdgeSegments );

    m_accelBuilder = std::make_unique<ClusterAccelBuilder>( getOptixRenderer().getContext(), m_stream, tessConfig );

    renderer.resetSubframes();

    m_accelBuilderNeedsUpdate = false;
}


void OptixSubdApp::setAdaptiveTessellation( bool tess )
{
    if( tess != m_args.enableAdaptiveTess )
    {
        m_args.enableAdaptiveTess = tess;
        getOptixRenderer().resetSubframes();
    }
}

void OptixSubdApp::setVertexNormalsEnabled( bool v )
{
    if ( v != m_args.enableVertexNormals )
    {
        m_args.enableVertexNormals = v;
        getOptixRenderer().resetSubframes();
    }
}

void OptixSubdApp::setClusterTessellationPattern( ClusterPattern clusterPattern )
{
    m_args.enableSlantedTess = ( clusterPattern == ClusterPattern::SLANTED );
    getOptixRenderer().resetSubframes();
}

void OptixSubdApp::setFineTessellationRate( float rate )
{
    if( rate != m_args.fineTessellationRate )
    {
        m_args.fineTessellationRate = rate;
        getOptixRenderer().resetSubframes();
    }
}

void OptixSubdApp::setFrustumVisibilityEnabled( bool enabled )
{
    m_args.enableFrustumVisibility = enabled;
    getOptixRenderer().resetSubframes();
}

void OptixSubdApp::setBackfaceVisibilityEnabled( bool enabled )
{
    m_args.enableBackfaceVisibility = enabled;
    getOptixRenderer().resetSubframes();
}

void OptixSubdApp::setCoarseTessellationRate( float rate )
{
    if( rate != m_args.coarseTessellationRate )
    {
        m_args.coarseTessellationRate = rate;
        getOptixRenderer().resetSubframes();
    }
}

void OptixSubdApp::setDisplacementScale( float scale )
{
    if ( scale != m_args.dispScale ) {
        m_args.dispScale = scale;
        getOptixRenderer().resetSubframes();
    }
}

void OptixSubdApp::setDisplacementBias( float offset )
{
    if ( offset != m_args.dispBias ) {
        m_args.dispBias = offset;
        getOptixRenderer().resetSubframes();
    }
}

void OptixSubdApp::setDisplacementFilterScale( float scale )
{
    if ( scale != m_args.dispFilterScale ) {
        m_args.dispFilterScale = scale;
        getOptixRenderer().resetSubframes();
    }
}

void OptixSubdApp::setDisplacementFilterMipBias( float bias )
{
    if ( bias != m_args.dispFilterMipBias ) {
        m_args.dispFilterMipBias = bias;
        getOptixRenderer().resetSubframes();
    }
}

void OptixSubdApp::unlockCamera()
{
    m_cameraCanAnimate = false;
}

void OptixSubdApp::resetCamera()
{
    const otk::Aabb& sceneAabb = m_scene->getAttributes().aabb;

    if( !m_args.camString.empty() )
        m_camera.set( m_args.camString );
    else if ( const View* view = m_scene->getView() )
    {
        m_camera.setEye(view->position);
        m_camera.setLookat(view->lookat);
        m_camera.setUp(view->up);
        m_camera.setFovY(view->fov);
    }
    else
        m_camera.frame( sceneAabb );

    // zFar here is used for OGL rendering.  Initial value includes the camera + scene
    otk::Aabb camSceneUnion = sceneAabb;
    camSceneUnion.include( m_camera.getEye() );
    m_camera.setFar( 1.1f * length(camSceneUnion.extent() ) );

    m_camera.setAspectRatio( float(m_args.targetResolution.x) / float(m_args.targetResolution.y) );

    m_cameraCanAnimate = true;

    m_trackball.setGimbalLock( true );
    m_trackball.setCamera( &m_camera );

    // set default camera move-speed to be dependent on look-at point : seems
    // more reliable than average instance scale in large scenes
    m_trackball.setMoveSpeed( length( m_camera.getLookat() - m_camera.getEye() )  );

    m_trackball.setReferenceFrame( { 1.f, 0.f, 0.f }, { 0.f, 0.f, 1.f }, { 0.f, 1.f, 0.f } );

    // Set static scene AABB so that trackball can update the camera far clip plane.  Ignores animation.
    m_trackball.setSceneAabb( sceneAabb );

    getOptixRenderer().resetSubframes();
}

void OptixSubdApp::screenshot()
{
    getOptixRenderer().saveScreenshot();
}

float OptixSubdApp::getCPUFrameTime() const
{
    return std::chrono::duration<float, std::milli>(m_currFrameStart - m_prevFrameStart).count();
}



