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

#include "./profiler.h"
#include "./stopwatch.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <OptiXToolkit/Util/Exception.h>

#include <cassert>
#include <type_traits>
#include <variant>

// clang-format on

Profiler& Profiler::get()
{
    static Profiler profiler;
    return profiler;
}

void Profiler::terminate()
{
    for (auto& timer : m_gpuTimers)
        timer.reset();
    
    m_streamMonitors.clear();
}

using CPUTimer = Profiler::Timer<StopwatchCPU>;
using GPUTimer = Profiler::Timer<StopwatchGPU>;

template <>
CPUTimer& Profiler::Timer<StopwatchCPU>::resolve()
{
    if( auto e = elapsed() )
        push_back( *e );
    return *this;
}
template <>
GPUTimer& Profiler::Timer<StopwatchGPU>::resolve()
{
    if( auto e = elapsedAsync() )
        push_back( *e );
    return *this;
}

template <>
CPUTimer& Profiler::Timer<StopwatchCPU>::profile()
{
    if (Profiler::get().isRecording())
    {
        auto e = elapsed();
        push_back( e ? *e : 0.f );
    }
    return *this;
}
template <>
GPUTimer& Profiler::Timer<StopwatchGPU>::profile()
{
    if (Profiler::get().isRecording())
    {
        auto e = elapsedAsync();
        push_back( e ? *e : 0.f );
    }
    return *this;
}

void Profiler::updateStreamMonitors( CUstream* streams, int numStreams )
{
    if( numStreams == 0 )
    {
        if( m_streamMonitors.empty() )
            m_streamMonitors.push_back( {} );
        m_streamMonitors.front().start();
    }
    else if( numStreams > 0 )
    {
        m_streamMonitors.resize( numStreams );
        for( int i = 0; i < numStreams; ++i )
            m_streamMonitors[i].start( streams[i] );
    }
}

void Profiler::frameStart( std::chrono::steady_clock::time_point time, CUstream* streams, int numStreams )
{
    int frequency = recordingFrequency;
    if( frequency > 0 )
    {
        double period = 1000. / double( frequency );

        m_isRecording = std::chrono::duration<double, std::milli>( time - m_prevTime ).count() >= period
                        || m_prevTime == std::chrono::steady_clock::time_point{};
    }
    else if( frequency < 0 )
        m_isRecording = true;
    else
        m_isRecording = false;

    if (m_isRecording)
        m_prevTime = time;

    updateStreamMonitors( streams, numStreams );
}

void Profiler::frameEnd()
{
    assert( !m_streamMonitors.empty() );
    for( auto& monitor : m_streamMonitors )
        monitor.stop();
}

void Profiler::frameSync()
{
    assert( !m_streamMonitors.empty() );
    for( auto& stream : m_streamMonitors )
        stream.sync();
}


void Profiler::frameResolve()
{
    auto resolveTimers = []( auto& timers ) {
        for( auto& timer : timers )
            timer->resolve();
    };

    resolveTimers( m_cpuTimers );

    if( !m_gpuTimers.empty() )
    {
        frameSync();
        resolveTimers( m_gpuTimers );
    }
}
