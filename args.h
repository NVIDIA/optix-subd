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

#include <cluster_builder/tessellatorConfig.h>
#include <material/materialCache.h>
#include <motionvec/motionvec.h>

#include <array>
#include <string>

// clang-format on

// Things that can be set from a scene.json file
struct SceneArgs
{

    // intentionally empty.  If you put things here, add parsing in scene.cpp

};

// Things we might want to set from command line.  Not all are currently exposed.
struct Args : private SceneArgs
{
    SceneArgs& sceneArgs()
    { 
        return *this;
    }

    bool showUIonStart = true;
    bool showOverlayOnStart = false;

    uint2  targetResolution = { 1920, 1080 };

    std::string outfile;
    std::string meshInputFile;
    std::string camString;

    unsigned char quantNBits = 0;

    bool enableFrustumVisibility = true;
    bool enableBackfaceVisibility = true;

    bool  enableAdaptiveTess  = true;
    float fineTessellationRate = TessellatorConfig::DEFAULT_FINE_TESSELLATION_RATE;
    float coarseTessellationRate = TessellatorConfig::DEFAULT_COARSE_TESSELLATION_RATE;
    bool  enableSlantedTess   = true;
    bool enableVertexNormals = false;

    float dispScale = 1.f;
    float dispBias  = 0.f;

    float dispFilterScale = TessellatorConfig::DEFAULT_DISPLACEMENT_FILTER_SCALE;
    float dispFilterMipBias = TessellatorConfig::DEFAULT_DISPLACEMENT_FILTER_MIP_BIAS;

    int frames           = 1;
    int logLevel         = 2;

    void parse( int argc, char const* const* argv );
};
