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

/// \file SelfIntersectionAvoidance.h
/// Primary interface of Self Intersection Avoidance library.
///
/// #define OTK_SIA_DISABLE_TRANSFORM_TRAVERSABLES // Disables support for optix transform traversables (motion matrix, motion srt and static matrix). Only instances are supported.
///
///
/// /// Example use:
///
///     ...
///
///     float3 objPos, objNorm;
///     float objOffset;
///
///     // generate object space spawn point and offset
///     getSafeTriangleSpawnOffset( objPos, objNorm, objOffset, ... );
///
///     float3 wldPos, wldNorm;
///     float wldOffset;
///
///     // convert object space spawn point and offset to world space
///     transformSafeSpawnOffset( wldPos, wldNorm, wldOffset, objPos, objNorm, objOffset, ... );
///
///     float3 front, back;
///     // offset world space spawn point to generate self intersection safe front and back spawn points
///     offsetSpawnPoint( front, back, wldPos, wldNorm, wldOffset );
///
///     // flip normal to point towards incoming direction
///     if( dot( wldNorm, wldInDir ) > 0.f )
///     {
///         wldNorm = -wldNorm;
///         swap( front, back );
///     }
///     ...
///
///     // pick safe spawn point for secondary scatter ray
///     float3 wldOutPos = ( dot( wldOutDir, wldNorm ) > 0.f ) ? front : back
/// 

#include <OptiXToolkit/ShaderUtil/CudaSelfIntersectionAvoidance.h>
#include <OptiXToolkit/ShaderUtil/OptixSelfIntersectionAvoidance.h>
