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

#include <profiler/stopwatch.h>
#include <profiler/sampler.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// clang-format on

struct ImGuiContext;

// Generic execution framework for host/device profiling data.
// 
// * Typical benchmark usage pattern:
//   
//          Timer<> t0 = profiler.initTimer("timer name");
//          for (frame loop) 
//          {
//              profiler.frameStart(steady_clock::now());
//   
//              t0.start();
//              // ... excecute profiled task
//              t0.stop();
//           
//              profiler.frameStop(); // all timers have been stopped
//              profiler.frameResolve(); 
//          }
//          float avg = t0.average();
//          profiler.terminate();
//   
//    
// * Typical (interactive) profiling usage pattern:
//   
//          Timer<> t0 = profiler.initTimer("timer name");
//          for (frame loop) 
//          {
//              profiler.frameStart(steady_clock::now());
//   
//              t0.start();
//              // ... excecute profiled task
//              t0.stop();
//           
//              // ... 
// 
//              profiler.frameStop(); // all timers have been stopped
// 
//              // ...  
// 
//              profiler.frameSync(); //before any timer is polled
// 
//              float ravg = t0.resolve().runningAverage();
//          }
//          profiler.terminate();
//
class Profiler
{
  public:
    constexpr static size_t BENCH_FRAME_COUNT = 400;  

    // returns the singleton Profiler
    static Profiler& get();

    // force the immediate release all device resources
    void terminate();

    // frequency < 0 : profile every frame (benchmark mode)
    // frequency == 0 : disable profiling
    // frequency > 0 : records samples at the given pace (in Hz)
    int recordingFrequency = 60;

    // returns true if the Profiler is recording data for the current frame
    bool isRecording() const { return m_isRecording; }

    // insert at the start of every frame (allows to pace sampling and skip
    // some frames if the frame-rate is too high)
    // 
    // note: the profiler will only monitor events on stream 0 if no dedicated 
    // streams are specified here. This can cause run-time exceptions if device
    // timers are polled without host synchronization
    void frameStart( std::chrono::steady_clock::time_point time, CUstream* streams = nullptr, int numStreams = 0 );

    // insert after the last timer is stopped in the frame
    void frameEnd();

    // blocks until all monitored device streams are synchronized with host
    void frameSync();

    // benchmarks data for the frame: forces all timers to resolve elapsed time
    // note: will block the calling thread until frameSync() completes
    void frameResolve();

    // Generic profiling timer with benchmarking functionality
    template <typename clock_type>
    struct Timer : public Sampler<float, BENCH_FRAME_COUNT>, private clock_type
    {
        Timer( char const* name ) : Sampler( {.name = name} ) { }

        using clock_type::start;
        using clock_type::stop;

        // note: user is responsible for device synchronization: use frameSync()
        Timer& resolve();  // record duration if the timer was active
        Timer& profile();  // record duration or 0. if the timer was inactive
    };

    typedef Timer<StopwatchCPU> CPUTimer;
    typedef Timer<StopwatchGPU> GPUTimer;

    template <typename timer_type>
    static inline timer_type& initTimer( char const* name );

  private:
    Profiler() noexcept         = default;
    Profiler( Profiler const& ) = delete;
    Profiler& operator=( Profiler const& ) = delete;

    std::chrono::steady_clock::time_point m_prevTime;

    bool m_isRecording = false;

  private:
    void updateStreamMonitors( CUstream* streams, int numStreams );

    std::vector<StopwatchGPU> m_streamMonitors;

  private:
    std::vector<std::unique_ptr<CPUTimer>> m_cpuTimers;
    std::vector<std::unique_ptr<GPUTimer>> m_gpuTimers;
};

template <typename timer_type>
inline timer_type& Profiler::initTimer( char const* name )
{
    Profiler& profiler = get();
    assert( profiler.m_prevTime.time_since_epoch().count() == 0 );
    if constexpr( std::is_same_v<timer_type, CPUTimer> )
        return *profiler.m_cpuTimers.emplace_back( std::make_unique<CPUTimer>( name ) );
    else if constexpr( std::is_same_v<timer_type, GPUTimer> )
        return *profiler.m_gpuTimers.emplace_back( std::make_unique<GPUTimer>( name ) );
}
