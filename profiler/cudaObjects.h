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

#include <OptiXToolkit/Util/Exception.h>
#include <cuda_runtime.h>
#include <OptiXToolkit/Util/CuBuffer.h>

class CudaEvent
{
  private:
    // Avoid accidentally creating new events with the default constructor. Use
    // CudaEvent::create() instead. This does break std::make_unique() but this
    // class is move-only anyway.
    // *just an experiment. delete if needed
    CudaEvent() { CUDA_CHECK( cudaEventCreate( &m_event ) ); }

  public:
    static CudaEvent create() { return {}; }
    CudaEvent(const cudaStream_t& stream) : CudaEvent()
    {
        record( stream );
    }
    CudaEvent( const CudaEvent& other ) = delete;
    CudaEvent( CudaEvent&& other ) noexcept
        : m_event( other.m_event )
    {
        other.m_event = 0;
    }
    ~CudaEvent()
    {
        if( m_event != 0 )
            CUDA_CHECK_NOTHROW( cudaEventDestroy( m_event ) );
    }
    CudaEvent& operator=( const CudaEvent& other ) = delete;
    CudaEvent& operator=( CudaEvent&& other ) noexcept
    {
        if( m_event != 0 )
            CUDA_CHECK_NOTHROW( cudaEventDestroy( m_event ) );
        m_event       = other.m_event;
        other.m_event = 0;
        return *this;
    };
    operator const cudaEvent_t&() const { return m_event; }
    void record( const cudaStream_t& stream ) { CUDA_CHECK( cudaEventRecord( m_event, stream ) ); }
    void wait( const cudaStream_t& stream, unsigned int flags = 0 ) const
    {
        CUDA_CHECK( cudaStreamWaitEvent( stream, m_event, flags ) );
    }
    void wait() const { CUDA_CHECK( cudaEventSynchronize( m_event ) ); }
    bool query() const {
        cudaError_t eventResult = cudaEventQuery( m_event );
        if( eventResult == cudaError::cudaSuccess )
            return true;
        if( eventResult == cudaError::cudaErrorNotReady )
            return false;
        CUDA_CHECK( eventResult );
        return false;
    }

  private:
    cudaEvent_t m_event;
};


