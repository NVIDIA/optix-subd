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
#include <ocb_version.h>

#include <array>
#include <cstring>

// fix for g++ still missing copy_n()
#if defined( __GNUC__ ) && defined( __cplusplus )
namespace std {
    template<class InputIt, class Size, class OutputIt> constexpr
    OutputIt copy_n( InputIt first, Size count, OutputIt result )
    {
        if( count > 0 )
        {
            *result = *first;
            ++result;
            for( Size i = 1; i != count; ++i, ++result )
                *result = *++first;
        }
        return result;
    }
}
#endif

template<unsigned ...Len> constexpr auto concat( const char( &...strings )[Len] )
{
    constexpr unsigned N = ( ... + Len ) - sizeof...( Len );
    std::array<char, N + 1> result = {};
    result[N] = '\0';
    auto it = result.begin();
    (void)( ( it = std::copy_n( strings, Len - 1, it ), 0 ), ... );
    return result;
}

template<class T, size_t N> using c_array = T[N];

template<class T, size_t N> c_array<T, N> const& as_c_array( std::array<T, N> const& a ) {
    return reinterpret_cast<T const(&)[N]>( *a.data() );
}

constexpr auto _windowName = concat( "OptixSubd ", git_branch, " ", git_commit, " (", build_date, ") - ", build_config );

char const* windowName = as_c_array( _windowName );
