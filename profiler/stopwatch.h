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

// clang-format off

#include <cuda.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <optional>
#include "cudaObjects.h"

// clang-format on

class StopwatchCPU
{
  public:
    void start();
    void stop();

    std::optional<float> elapsed();  // returns dt = stop - start
    std::optional<float> before( std::chrono::steady_clock::time_point t );
    std::optional<float> after( std::chrono::steady_clock::time_point t );

  private:
    using steady_clock = std::chrono::steady_clock;
    using duration     = std::chrono::duration<double, std::milli>;

    steady_clock::time_point m_startTime;
    steady_clock::time_point m_stopTime;
};

class StopwatchGPU
{
  public:
    void start( CUstream stream = 0 );
    void stop();

    std::optional<float> elapsed();  // returns dt = stop - start

    void sync();  // host-side fence synchronizing with device-side stop event

    std::optional<float> elapsedAsync();            // returns dt = stop - start
    std::optional<float> beforeAsync( CUevent t );  // returns dt = start - t
    std::optional<float> afterAsync( CUevent t );   // returns dt = t - stop

  private:

    struct Event
    {
        CUevent m_event = nullptr;
        Event();
        ~Event() noexcept;
        Event( Event&& e )
        {
            m_event   = e.m_event;
            e.m_event = nullptr;
        }
        Event&  operator=( const Event& event ) = delete;
        CUevent operator*() { return m_event; }
    };

    Event m_startEvent;
    Event m_stopEvent;

    CUstream m_stream = nullptr;

    enum class State : uint8_t
    {
        reset = 0,
        ticking,
        stopped,
        synced
    } state = State::reset;
};

class StopwatchGPUAsync {
  public:
    void start( CUstream stream );
    void stop( CUstream stream );

    // Returns a sample if one is ready
    std::optional<float> elapsed();

  private:
    using StartStop = std::pair<CudaEvent, CudaEvent>;
    std::optional<StartStop> m_started;
    std::vector<StartStop>   m_queued;
    std::vector<StartStop>   m_pool;
};
