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

#include "shadingTypes.h"
#include "args.h"

#include <cluster_builder/clusterAccelBuilder.h>
#include <texture/textureCuda.h>
#include "OptiXToolkit/Util/CuBuffer.h"

#include <OptiXToolkit/Gui/Camera.h>
#include <OptiXToolkit/Gui/Trackball.h>
#include <OptiXToolkit/ShaderUtil/Aabb.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

// clang-format on

class OptixRenderer;
class TextureCache;
class MaterialCache;

struct MotionVecPass;
struct WireframePass;
class ClusterAccelBuilder;
class Scene;
struct Shape;
class StopwatchGPU;
struct TessellatorConfig;
struct View;

class OptixSubdApp
{
  public:
    OptixSubdApp( int argc, char const* const* argv );

    ~OptixSubdApp();

    const std::filesystem::path getBinaryPath() const { return m_binaryPath; }
    const std::filesystem::path getMediaPath() const { return m_mediaPath; }
    void setMediaPath( std::filesystem::path path ) { m_mediaPath = path; }

    bool interactiveMode() const;

    void loadScene( std::string const& filepath, std::string const& mediapathm, int2 frameRange = {0, 0} );
    void setupGL();

    std::string const& getCurrentShape() const { return m_args.meshInputFile; }

    void renderInteractiveSubframe(float animTime, float animFramerate);
    void renderBatchSubframes();

    void drawGL();
    void setGLWireframeEnabled( bool enabled ) { m_wireframePassEnabled = enabled; }
    bool getGLWireframeEnabled() const { return m_wireframePassEnabled; }

    void setOutputBufferTargetSize( uint2 targetsize );
    int2 getOutputBufferTargetSize() const;

    OptixRenderer& getOptixRenderer();
    MotionVecPass& getMotionVecPass();

    const Args&     getArgs() const { return m_args; }

    const Scene&    getScene() const { return *m_scene; }
    Scene&          getScene() { return *m_scene; }

    otk::Camera&    getCamera() { return m_camera; }
    otk::Trackball& getTrackBall() { return m_trackball; }

    bool getAdaptiveTessellation() const { return m_args.enableAdaptiveTess; }
    void setAdaptiveTessellation( bool tess );

    bool getVertexNormalsEnabled() const  { return m_args.enableVertexNormals; }
    void setVertexNormalsEnabled( bool flag );

    ClusterPattern getClusterTessellationPattern(  )  const { return m_args.enableSlantedTess ? ClusterPattern::SLANTED : ClusterPattern::REGULAR; }
    void setClusterTessellationPattern( ClusterPattern clusterPattern );

    bool getFrustumVisibilityEnabled() const { return m_args.enableFrustumVisibility; }
    void setFrustumVisibilityEnabled(bool enabled);

    bool getBackfaceVisibilityEnabled() const { return m_args.enableBackfaceVisibility; }
    void setBackfaceVisibilityEnabled(bool enabled);

    float getFineTessellationRate() const { return m_args.fineTessellationRate; }
    void  setFineTessellationRate( float rate );

    float getCoarseTessellationRate() const { return m_args.coarseTessellationRate; }
    void  setCoarseTessellationRate( float rate );

    float getDisplacementScale() const { return m_args.dispScale; }
    void  setDisplacementScale( float scale );

    float getDisplacementBias() const { return m_args.dispBias; }
    void  setDisplacementBias( float offset );

    float getDisplacementFilterScale() const { return m_args.dispFilterScale; }
    void  setDisplacementFilterScale( float scale );

    float getDisplacementFilterMipBias() const { return m_args.dispFilterMipBias; }
    void setDisplacementFilterMipBias( float bias );

    // force camera to ignore keyframed animation
    void unlockCamera();
    void resetCamera();

    void screenshot();

    float getCPUFrameTime() const; // std::steady_clock delta-T in milli-seconds

    CUstream getCudaStream() const { return m_stream; }


  private:
    void createAccelBuilder();

    void updateCamera( const View* view, bool interactive );

    // tessellate, build accel, and render one frame
    void renderSubframe();

  private:
    static constexpr CUstream const m_stream = 0;  // named default stream

    Args m_args;

    // path to executable (because argv0 is not reliable)
    std::filesystem::path m_binaryPath;

    // path to local folder w/ media assets
    std::filesystem::path m_mediaPath;  
    
    std::unique_ptr<OptixRenderer> m_optixRenderer;

    std::unique_ptr<MotionVecPass>     m_motionVecPass;
    std::unique_ptr<WireframePass>     m_wireframePass;

    bool m_wireframePassEnabled = false;

    std::unique_ptr<ClusterAccelBuilder>     m_accelBuilder;

    std::unique_ptr<Scene> m_scene;

    bool           m_cameraCanAnimate = true;
    otk::Camera    m_camera;
    otk::Camera    m_tessellationCamera;
    otk::Camera    m_prevCamera;
    otk::Trackball m_trackball;

    uint64_t m_frameIndex = 0;

    std::chrono::steady_clock::time_point m_currFrameStart = {};
    std::chrono::steady_clock::time_point m_prevFrameStart = {};
    std::chrono::steady_clock::time_point m_animStart      = {};

    float m_animationTime = 0.f; // elapsed time in seconds from the beginning of the animation

    bool m_accelBuilderNeedsUpdate = true;

    ClusterAccels     m_accels;
    ClusterStatistics m_build_stats;

    uint32_t m_maxClusterEdgeSegments = 0;
};
