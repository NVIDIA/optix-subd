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


#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include <glad/glad.h>

class Scene;

namespace otk {
  class Camera;
  template <typename PIXEL_FORMAT> class CUDAOutputBuffer;
}  // namespace otk


template <typename T> struct ReadWriteResourceInterop;

struct WireframePass
{
    WireframePass( const Scene& scene );
    virtual ~WireframePass();

    void run( const Scene& scene, const otk::Camera& cam, float2 jitter, uint32_t width, uint32_t height, const ReadWriteResourceInterop<float>& depth );

  private:
    GLuint m_program              = 0;
    GLint  m_uniformMvpMatrixLoc  = 0;
    GLint  m_uniformScreenDimsLoc = 0;
    GLint  m_uniformDepthBiasLoc  = 0;
    GLint  m_depthTextureLoc      = 0;
    GLint  m_jitterLoc            = 0;

    // per subd (not per instance)
    std::vector<std::unique_ptr<otk::CUDAOutputBuffer<float3>>> m_vertexBuffers;
    std::vector<GLuint> m_vertex_arrays;
    std::vector<GLuint> m_index_buffers;
    std::vector<size_t> m_index_buffer_sizes;
    
    // per instance
    struct SubdInstance;
    std::vector<SubdInstance> m_instances;
};
