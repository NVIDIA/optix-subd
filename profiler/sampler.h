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

#include <array>
#include <string>

// clang-format on

// A generic data sampler with basic statistics functionality

// default running avg window = 1 second @ 60Hz
template <typename T, size_t _size = 60>
struct Sampler : public std::array<T, _size>
{
    std::string name;

    // values tracked in running circular buffer
    T samples_sum = T( 0 );

    // values tracked since most recent reset
    size_t samples_count = 0;
    T      latest        = {};
    T      total         = T( 0 );
    T      min           = std::numeric_limits<T>::max();
    T      max           = std::numeric_limits<T>::lowest();

    void push_back( T sample );

    void reset();

    T median() const { return .5f * ( double( max ) - double( min ) ); }
    T average() const { return static_cast<T>( double( total ) / double( samples_count ) ); }
    T runningAverage() const
    {
        return static_cast<T>( double( samples_sum ) / double( std::min( samples_count, _size ) ) );
    }

    // current position in circular buffer
    uint32_t offset() const { return static_cast<uint32_t>( samples_count % _size ); }

    void print()
    {
        size_t n = std::min( _size, samples_count );
        for(size_t i = 0; i < n; i++ )
        {
            std::printf( "Benchmark: frame %d time %.4f ms\n", (int)i, (*this)[i]);
        }
    }
};

template <typename T, size_t _size>
inline void Sampler<T, _size>::push_back( T sample )
{
    latest = sample;
    total += latest;
    min = std::min( latest, min );
    max = std::max( latest, max );

    samples_sum += sample;

    if( samples_count < _size )
        ( *this )[samples_count++] = sample;
    else
    {
        T& oldest = ( *this )[samples_count++ % _size];
        samples_sum -= oldest;
        oldest = sample;
    }
}

template <typename T, size_t _size>
inline void Sampler<T, _size>::reset()
{
    samples_count = 0;
    latest        = {};
    samples_sum   = T( 0 );
    min           = std::numeric_limits<T>::max();
    max           = std::numeric_limits<T>::min();
#if !defined( NDEBUG )
    fill( T( 0 ) );
#endif
}