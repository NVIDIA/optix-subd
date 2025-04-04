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

#include <algorithm>
#include <limits>
#include "utils.h"


// -----------------------------------------------------------------------------
// CUDA device code utility functions. This header may only be included into
// CUDA files (cuda runtime, cuda driver, or optix shader code).
// -----------------------------------------------------------------------------


__forceinline__ __device__ float safeLength(float3 v)
{
    const float3 n = make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
    // avoid overflow by dividing by the max component
    const float m = fmaxf(n.x, fmaxf(n.y, n.z));
    const float x = n.x / m;
    const float y = n.y / m;
    const float z = n.z / m;
    // scale back by the max component
    const float len = m * (sqrtf(x * x + y * y + z * z));
    return len;
}

__forceinline__ __device__ float3 safeNormalize(float3 n)
{
    // avoid division by 0 by adding numeric_limits::min
    const float len = safeLength(n) + std::numeric_limits<float>::min();
    return n * (1.f / len);
}

struct Onb
{
    __forceinline__ __device__ Onb( float3 normal )
    {
        m_normal = normal;

        if( fabs( m_normal.x ) > fabs( m_normal.z ) )
        {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0.f;
        }
        else
        {
            m_binormal.x = 0.f;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = safeNormalize( m_binormal );
        m_tangent  = cross( m_binormal, m_normal );
    }
    // transform from the coordinate system represented by ONB
    __device__ float3 toWorld( const float3& v ) const { return ( v.x * m_tangent + v.y * m_binormal + v.z * m_normal ); }

    // transform to the coordinate system represented by ONB
    __device__ float3 toLocal( const float3& v ) const
    {
        const float x = dot( v, m_tangent );
        const float y = dot( v, m_binormal );
        const float z = dot( v, m_normal );
        return make_float3( x, y, z );
    }

    float3 m_tangent;
    float3 m_binormal;
    float3 m_normal;
};

__forceinline__ __device__ float2 concentricSampleDisk( float2 u )
{
    // Map uniform random numbers to $[-1,1]^2$
    const float2 uOffset = 2.f * u - make_float2( 1.f );

    // Handle degeneracy at the origin
    if( uOffset.x == 0.f && uOffset.y == 0.f )
        return make_float2( 0.f );

    // Apply concentric mapping to point
    float theta, r;
    if( fabsf( uOffset.x ) > fabsf( uOffset.y ) )
    {
        r     = uOffset.x;
        theta = PI_OVER_4 * ( uOffset.y / uOffset.x );
    }
    else
    {
        r     = uOffset.y;
        theta = PI_OVER_2 - PI_OVER_4 * ( uOffset.x / uOffset.y );
    }
    return r * make_float2( cosf( theta ), sinf( theta ) );
}

__forceinline__ __device__ float3 cosineSampleHemisphere( float2 u )
{
    const float2 d = concentricSampleDisk( u );
    const float  z = sqrtf( fmaxf( 0.0f, 1.f - d.x * d.x - d.y * d.y ) );
    return make_float3( d.x, d.y, z );
}


__device__ inline
float3 temperature( const float t )
{
    const float b = t < 0.25f ? smoothstep( -0.25f, 0.25f, t ) : 1.0f - smoothstep( 0.25f, 0.5f, t );
    const float g = t < 0.5f ? smoothstep( 0.0f, 0.5f, t ) : ( t < 0.75f ? 1.0f : 1.0f - smoothstep( 0.75f, 1.0f, t ) );
    const float r = smoothstep( 0.5f, 0.75f, t );
    return make_float3( r, g, b );
}


// generalized lerp; map a value in [a, b] to the range [c, d] without clamping
__device__ inline
float remap(float x, float a, float b, float c, float d) 
{
    return c + (d - c) * (x - a) / (b - a);
}


__device__ inline 
unsigned floatToUInt(float _V, float _Scale)
{
    return (unsigned)floorf(_V * _Scale + 0.5f);
}

/**
* Octahedral normal vector encoding.
* param N must be a unit vector.
* return An octahedral vector on the [-1, +1] square.
*/
__device__ inline float2 unitVectorToOctahedron( float3 N )
{
    const float d = dot( make_float3( 1.f ), make_float3( fabsf( N.x ), fabsf( N.y ), fabsf( N.z ) ) );
    N.x /= d;
    N.y /= d;
    if( N.z <= 0.f )
    {
        float2 signs;
        signs.x = N.x >= 0.f ? 1.f : -1.f;
        signs.y = N.y >= 0.f ? 1.f : -1.f;

        const float2 k = ( make_float2( 1.f, 1.f ) - make_float2( fabsf( N.y ), fabsf( N.x ) ) ) * signs;

        N.x = k.x;
        N.y = k.y;
    }
    return make_float2( N.x, N.y );
}

/**
* Octahedral normal vector decoding.
* param Oct An octahedral vector as returned from UnitVectorToOctahedron, on the [-1, +1] square.
* return a unit vector.
*/
__device__ inline float3 octahedronToUnitVector( float2 Oct )
{
    float3 N = make_float3( Oct.x, Oct.y, 1.f - dot( make_float2( 1.f ), make_float2( fabsf( Oct.x ), fabsf( Oct.y ) ) ) );
    if( N.z < 0 )
    {
        float2 signs;
        signs.x = N.x >= 0.f ? 1.f : -1.f;
        signs.y = N.y >= 0.f ? 1.f : -1.f;

        const float2 k = ( make_float2( 1.f, 1.f ) - make_float2( fabsf( N.y ), fabsf( N.x ) ) ) * signs;

        N.x = k.x;
        N.y = k.y;
    }
    return normalize( N );
}

__device__ inline unsigned packNormalizedVector( float3 x )
{
    float2 XY = unitVectorToOctahedron( x );

    XY = XY * make_float2( .5f, .5f ) + make_float2( .5f, .5f );

    unsigned X = floatToUInt( __saturatef( XY.x ), ( 1 << 16 ) - 1 );
    unsigned Y = floatToUInt( __saturatef( XY.y ), ( 1 << 16 ) - 1 );

    unsigned PackedOutput = X;
    PackedOutput |= Y << 16;
    return PackedOutput;
}

__device__ inline float3 unpackNormalizedVector( unsigned PackedInput )
{
    unsigned X = PackedInput & ( ( 1 << 16 ) - 1 );
    unsigned Y = PackedInput >> 16;
    float2   XY = make_float2( 0.f, 0.f );
    XY.x = (float)X / ( ( 1 << 16 ) - 1 );
    XY.y = (float)Y / ( ( 1 << 16 ) - 1 );
    XY   = XY * make_float2(2.f) + make_float2( -1.f );
    return octahedronToUnitVector( XY );
}


// Random numbers

template <unsigned int N>
static __host__ __device__ __inline__ unsigned int tea( unsigned int val0, unsigned int val1 )
{
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for( unsigned int n = 0; n < N; n++ )
    {
        s0 += 0x9e3779b9;
        v0 += ( ( v1 << 4 ) + 0xa341316c ) ^ ( v1 + s0 ) ^ ( ( v1 >> 5 ) + 0xc8013ea4 );
        v1 += ( ( v0 << 4 ) + 0xad90777d ) ^ ( v0 + s0 ) ^ ( ( v0 >> 5 ) + 0x7e95761e );
    }

    return v0;
}

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg( unsigned int& prev )
{
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev                     = ( LCG_A * prev + LCG_C );
    return prev & 0x00FFFFFF;
}

static __host__ __device__ __inline__ unsigned int lcg2( unsigned int& prev )
{
    prev = ( prev * 8121 + 28411 ) % 134456;
    return prev;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd( unsigned int& prev )
{
    return ( (float)lcg( prev ) / (float)0x01000000 );
}

static __host__ __device__ __inline__ unsigned int rot_seed( unsigned int seed, unsigned int frame )
{
    return seed ^ frame;
}

static __host__ __device__ __inline__ float2 randomStrat( const int sampleOffset, const int strataCount, unsigned int& prev )
{
    const int   sy        = sampleOffset / strataCount;
    const int   sx        = sampleOffset - sy * strataCount;
    const float invStrata = 1.0f / strataCount;
    float2      result    = make_float2( ( sx + rnd( prev ) ) * invStrata, ( sy + rnd( prev ) ) * invStrata );
    return result;
}

