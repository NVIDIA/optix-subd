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


class Scene;
struct DisplacementSampler;
struct HitResult;
struct LimitFrame;
struct Vertex;

namespace otk {
    class Camera;
}

struct GBuffer;

template<typename T>
struct DynamicSubdCUDA;

enum MotionVecDisplacementMode
{
    MOTION_VEC_DISPLACEMENT_FROM_SUBD_EVAL = 0,
    MOTION_VEC_DISPLACEMENT_FROM_MATERIAL,
    COUNT
};

struct MotionVecPass
{
    MotionVecPass( MotionVecDisplacementMode mode  = MOTION_VEC_DISPLACEMENT_FROM_SUBD_EVAL );
    virtual ~MotionVecPass();

    void setDisplacementMode( MotionVecDisplacementMode mode )
    {
        m_displacementMode = mode;
    }
    MotionVecDisplacementMode getDisplacementMode() const
    {
        return m_displacementMode;
    }

    void run( const Scene&               scene,
              float                      displacementScale,
              float                      displacementBias,
              const otk::Camera&         cam,
              const otk::Camera&         prevCam,
              float2                     jitter,
              const CuBuffer<HitResult>& hits,
              GBuffer&     gbuffer );

    struct SubdInstance;

    private:

    MotionVecDisplacementMode m_displacementMode = MOTION_VEC_DISPLACEMENT_FROM_MATERIAL;

    std::vector<DynamicSubdCUDA<Vertex>> m_subds;
    CuBuffer<DynamicSubdCUDA<Vertex>>    m_deviceSubds;

    std::vector<SubdInstance> m_instances;
    CuBuffer<SubdInstance>    m_deviceInstances;
};
