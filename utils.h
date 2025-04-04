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


// -----------------------------------------------------------------------------
// Utility functions that are compatible with .cu and .cpp files, i.e.
// this utils.h file may be included into regular c-code as well as CUDA code
// (cuda runtime, cuda driver, or optix shader code).
// -----------------------------------------------------------------------------

#include "OptiXToolkit/ShaderUtil/Affine.h"
#include <OptiXToolkit/ShaderUtil/Matrix.h>

#include <cstdint>
#include <span>


constexpr float PI = M_PIf;
constexpr float INV_PI = M_1_PIf;

constexpr float PI_OVER_2 = .5f * M_PIf;
constexpr float INV_PI_OVER_2 = 1.f / PI_OVER_2;

constexpr float PI_OVER_4 = .25f * M_PIf;
constexpr float INV_PI_OVER_4 = 1.f / PI_OVER_4;

constexpr float TWO_PI = 2.f * M_PIf;
constexpr float INV_2PI = .5f * M_1_PIf;


template <typename T>
constexpr T div_up( T dividend, T divisor )
{
    return ( dividend + divisor - 1 ) / divisor;
}


constexpr float smoothstep(const float edge0, const float edge1, const float x)
{
  /** assert( edge1 > edge0 ); */
  const float t = std::min( std::max( ( x - edge0 ) / ( edge1 - edge0 ), 0.f ), 1.f );
  return t*t * ( 3.0f - 2.0f*t );
}


// Write row major order to array
__host__ __device__ inline void copy( const otk::Affine& a, float b[12] )
{
    //copy( a.matrix3x4(), b );
    b[ 0] = a.linear[0]; b[ 1] = a.linear[1]; b[ 2] = a.linear[2]; b[ 3] = a.translation.x;
    b[ 4] = a.linear[3]; b[ 5] = a.linear[4]; b[ 6] = a.linear[5]; b[ 7] = a.translation.y;
    b[ 8] = a.linear[6]; b[ 9] = a.linear[7]; b[10] = a.linear[8]; b[11] = a.translation.z;
}

