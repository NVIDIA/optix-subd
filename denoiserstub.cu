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

#include <OptiXToolkit/ShaderUtil/color.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include "shadingTypes.h"

#include "GBuffer.cuh"

 __global__
void accumulationKernel( const RwFloat4 input, RwFloat4 output, int subframe )
{
    uint2 idx = {
        .x = blockIdx.x * blockDim.x + threadIdx.x,
        .y = blockIdx.y * blockDim.y + threadIdx.y,
    };

    uint2 outputSize = output.m_size;
    uint2 inputSize = input.m_size;

    if ( ( idx.x >= outputSize.x ) || ( idx.y >= outputSize.y ) )
        return;

    float4 c = make_float4( 0.f, 0.f, 0.f, 1.f );
    if( idx.x < inputSize.x && idx.y < inputSize.y )
    {
        c = gbuffer::read( input, idx );
        if ( subframe > 0 )
        {
            float4 oldval = gbuffer::read( output, idx );
            c = lerp( oldval, c, 1.0f / float(subframe + 1 ) );
        }
    }
    gbuffer::write( c, output, idx );
}

void denoise( GBuffer& gbuffer, int subframe )
{
    // Stub placeholder for DLSS.
    // Just accumulates when the camera stops moving.  No motion vecs or TAA

    if ( !gbuffer.m_color.isValid() || !gbuffer.m_denoised.isValid() ) return;

    const int blockSize1D = 32;
    const uint2 targetsize = gbuffer.m_denoised.m_size;
    dim3 numBlocks( targetsize.x / blockSize1D + 1, targetsize.y / blockSize1D + 1, 1 );
    dim3 numThreadsPerBlock( 32, 32, 1 );
    
    accumulationKernel<<<numBlocks, numThreadsPerBlock>>>( gbuffer.m_color, gbuffer.m_denoised, subframe );
}


