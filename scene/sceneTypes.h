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

#include <scene/animation.h>

#include <OptiXToolkit/ShaderUtil/Aabb.h>
#include <OptiXToolkit/ShaderUtil/Affine.h>
#include <OptiXToolkit/ShaderUtil/Quaternion.h>
#include <cuda.h>

#include <cstdint>


struct FrameTime
{
    float currentTime = 0.f;
    float frameRate = 0.f;
};

struct Instance
{
    otk::Affine localToWorld = otk::Affine::identity();

    otk::Aabb   aabb;

    float3     translation = { 0.f, 0.f, 0.f };
    otk::quat  rotation    = { 1.f, 0.f, 0.f, 0.f };
    float3     scaling     = { 1.f, 1.f, 1.f };

    uint32_t meshID     = ~uint32_t( 0 );

    // Instances cannot hold std containers because they are
    // hoisted to device memory by CuBuffer
    const char* name = nullptr;

    void updateLocalTransform();
};

struct View
{
    float3 position = { 0.f, 0.f, -1.f };
    
    float3 lookat = { 0.f, 0.f, 0.f };
    float3 up = { 0.f, 1.f, 0.f };

    std::optional<otk::quat> rotation;
    
    float fov = 35.f;

    bool  isAnimated = false;
};

struct Sequence
{
    std::string name;

    float start = std::numeric_limits<float>::max();
    float end = std::numeric_limits<float>::min();

    std::vector<std::unique_ptr<anim::ChannelInterface>> channels;

    void animate( const FrameTime& frameTime );
};

struct Animation
{
    std::string name;

    std::vector<std::unique_ptr<Sequence>> sequences;

    float start = std::numeric_limits<float>::max();
    float end = std::numeric_limits<float>::min();

    float duration() const { return end > start ? end - start : 0.f; }

    void animate( const FrameTime& frameTime );
};

