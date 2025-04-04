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

#include "./stopwatch.h"
#include "cudaObjects.h"
#include <OptiXToolkit/Util/Exception.h>

#include <cassert>

// clang-format on

//
// StopwatchCPU
//

void StopwatchCPU::start()
{
    m_startTime = steady_clock::now();
    assert( m_stopTime < m_startTime );
}

void StopwatchCPU::stop()
{
    assert( m_startTime.time_since_epoch().count() > 0 );
    m_stopTime = steady_clock::now();
}

std::optional<float> StopwatchCPU::elapsed()
{
    assert( m_stopTime >= m_startTime );

    if( m_startTime == steady_clock::time_point{} )
        return {};

    float elapsed = static_cast<float>( duration( m_stopTime - m_startTime ).count() );
    m_startTime = m_stopTime = {};
    return elapsed;
}

std::optional<float> StopwatchCPU::before( steady_clock::time_point t )
{
    assert( m_startTime >= t && m_startTime > steady_clock::time_point{});
    return (float)duration( m_startTime - t).count();
}

std::optional<float> StopwatchCPU::after( steady_clock::time_point t )
{
    assert( m_stopTime <= t && m_stopTime > steady_clock::time_point{});
    return (float)duration( t - m_stopTime ).count();
}

//
// StopwatchGPU
//

StopwatchGPU::Event::Event()
{
    CUDA_CHECK( cudaEventCreate( &m_event ) );
}

StopwatchGPU::Event::~Event() noexcept
{
    if( m_event )
        CUDA_CHECK( cudaEventDestroy( m_event ) );
}

#if 0
StopwatchGPU::State& operator++(StopwatchGPU::State& s)
{
    using State = StopwatchGPU::State;
    // circular increment operator
    return s = State((int(s) + 1) % int(State::count));
}
#endif

void StopwatchGPU::start( CUstream stream )
{
    // all but 'stopped' states are valid, so can advance up to 3 times
    assert( state != State::ticking );
    CUDA_CHECK( cudaEventRecord( *m_startEvent, m_stream = stream ) );
    state = State::ticking;
}

void StopwatchGPU::stop()
{
    assert( state == State::ticking );
    CUDA_CHECK( cudaEventRecord( *m_stopEvent, m_stream ) );
    state = State::stopped;
}

void StopwatchGPU::sync()
{
    assert( state == State::stopped );
    CUDA_CHECK( cudaEventSynchronize( *m_stopEvent ) );
    state = State::synced;
}

std::optional<float> StopwatchGPU::elapsed()
{
    if( state == State::reset )
        return {};

    assert( state == State::stopped );

    sync();

    state         = State::reset;
    float elapsed = 0.f;
    CUDA_CHECK( cudaEventElapsedTime( &elapsed, *m_startEvent, *m_stopEvent ) );
    return elapsed;
}

std::optional<float> StopwatchGPU::elapsedAsync()
{
    if( state == State::reset )
        return {};

    // user is responsible for device sync, so we can't track it
    assert( state != State::ticking );

    state         = State::reset;
    float elapsed = 0.f;
    CUDA_CHECK( cudaEventElapsedTime( &elapsed, *m_startEvent, *m_stopEvent ) );
    return elapsed;
}

std::optional<float> StopwatchGPU::beforeAsync( CUevent t )
{
    if( state == State::reset )
        return {};

    assert( state != State::ticking );

    float elapsed = 0.f;
    CUDA_CHECK( cudaEventElapsedTime( &elapsed, t, *m_startEvent ) );
    return elapsed;
}

std::optional<float> StopwatchGPU::afterAsync( CUevent t )
{
    if( state == State::reset )
        return {};

    assert( state != State::ticking );

    float elapsed = 0.f;
    CUDA_CHECK( cudaEventElapsedTime( &elapsed, t, *m_stopEvent ) );
    return elapsed;
}

void StopwatchGPUAsync::start(CUstream stream = 0)
{
    if(m_pool.empty())
    {
        // It is expected elapsed() is polled until empty on the same thread so
        // the queue size should generally just be one frame ahead, or a few if
        // truly async.
        assert(m_queued.size() < 10000);
        m_pool.emplace_back( StartStop{ CudaEvent::create(), CudaEvent::create() } );
    }
    m_started.emplace(std::move(m_pool.back()));
    m_pool.pop_back();
    m_started->first.record(stream);
}

void StopwatchGPUAsync::stop(CUstream stream = 0)
{
    m_started.value().second.record(stream);
    m_queued.emplace_back( std::move( *m_started ) );
    m_started.reset();
}

std::optional<float> StopwatchGPUAsync::elapsed()
{
    if(!m_queued.empty() && m_queued.front().second.query())
    {
        float elapsed = 0.f;
        CUDA_CHECK( cudaEventElapsedTime( &elapsed, m_queued.front().first, m_queued.front().second ) );
        m_pool.emplace_back( std::move( m_queued.front() ) );  // return the events to the pool
        m_queued.erase( m_queued.begin() );                    // O(n) but n should be small
        return elapsed;
    }
    return {};
}
