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

#include <opensubdiv/far/patchDescriptor.h>
#include <opensubdiv/far/patchParam.h>
#include <opensubdiv/tmr/nodeDescriptor.h>
#include <opensubdiv/tmr/subdivisionPlan.h>
#include <opensubdiv/tmr/types.h>

#include <span>
#include <numeric>
#include <iterator>
#include <cuda/std/mdspan>

// Compile-time constant for kernels and warp-parallel functions.
const uint32_t WARPS_PER_BLOCK = 4;
// clang-format on

using namespace OpenSubdiv;


struct PatchWeights
{
    float p;
    float du;
    float dv;

    __host__ __device__ float& operator[]( uint32_t i ) { return reinterpret_cast<float*>( &p )[i]; }

    __host__ __device__ const float& operator[]( uint32_t i ) const { return reinterpret_cast<const float*>( &p )[i]; }

    __host__ __device__ size_t size() const { return 3; }

    __host__ __device__ PatchWeights& operator+=( const PatchWeights& other )
    {
        for( int i = 0; i < this->size(); ++i )
        {
            this->operator[]( i ) += other[i];
        }
        return *this;
    }

    __host__ __device__ PatchWeights& operator-=( const PatchWeights& other )
    {
        for( int i = 0; i < this->size(); ++i )
        {
            this->operator[]( i ) -= other[i];
        }
        return *this;
    }

    __host__ __device__ PatchWeights operator*( float scalar )
    {
        PatchWeights result;

        for( int i = 0; i < this->size(); ++i )
        {
            result[i] = this->operator[]( i ) * scalar;
        }
        return result;
    }
};


__host__ __device__ inline OpenSubdiv::Far::PatchDescriptor::Type regularBasisType( OpenSubdiv::Sdc::SchemeType scheme )
{
    using namespace OpenSubdiv;
    using enum Far::PatchDescriptor::Type;
    switch( scheme )
    {
        case Sdc::SCHEME_CATMARK:
            return REGULAR;
        case Sdc::SCHEME_LOOP:
            return LOOP;
    }
    return NON_PATCH;
}

__host__ __device__ inline OpenSubdiv::Far::PatchDescriptor::Type irregularBasisType( OpenSubdiv::Sdc::SchemeType scheme,
                                                                                      OpenSubdiv::Tmr::EndCapType endcap )
{
    using namespace OpenSubdiv;
    using enum Far::PatchDescriptor::Type;
    if( scheme == Sdc::SCHEME_CATMARK )
    {
        switch( endcap )
        {
            case Tmr::EndCapType::ENDCAP_BILINEAR_BASIS:
                return QUADS;
            case Tmr::EndCapType::ENDCAP_BSPLINE_BASIS:
                return REGULAR;
            case Tmr::EndCapType::ENDCAP_GREGORY_BASIS:
                return GREGORY_BASIS;
        }
    }
    else if( scheme == Sdc::SCHEME_LOOP )
    {
        switch( endcap )
        {
            case Tmr::EndCapType::ENDCAP_BILINEAR_BASIS:
                return TRIANGLES;
            case Tmr::EndCapType::ENDCAP_BSPLINE_BASIS:
                return LOOP;
            case Tmr::EndCapType::ENDCAP_GREGORY_BASIS:
                return GREGORY_TRIANGLE;
        }
    }
    return Far::PatchDescriptor::NON_PATCH;
}

// both Loop & Catmark quadrant traversals expect Z-curve winding
// (see subdivisionPlanBuilder for details)
__host__ __device__ inline void traverseCatmark( float& u, float& v, unsigned char& quadrant )
{
    if( u >= 0.5f )
    {
        quadrant ^= 1;
        u = 1.0f - u;
    }
    if( v >= 0.5f )
    {
        quadrant ^= 2;
        v = 1.0f - v;
    }
    u *= 2.0f;
    v *= 2.0f;
}

// note: Z-winding of triangle faces rotates sub-domains every subdivision level,
// but the center face is always at index (2)
//
//                0,1                                    0,1
//                 *                                      *
//               /   \                                  /   \
//              /     \                                /  3  \
//             /       \                              /       \
//            /         \           ==>        0,0.5 . ------- . 0.5,0.5
//           /           \                          /   \ 2 /   \
//          /             \                        /  0  \ /  1  \
//         * ------------- *                      * ----- . ----- *
//      0,0                 1,0                0,0      0.5,0      1,0

__host__ __device__ inline unsigned char traverseLoop( float median, float& u, float& v, bool& rotated )
{
    if( !rotated )
    {
        if( u >= median )
        {
            u -= median;
            return 1;
        }
        if( v >= median )
        {
            v -= median;
            return 3;
        }
        if( ( u + v ) >= median )
        {
            rotated = true;
            return 2;
        }
    }
    else
    {
        if( u < median )
        {
            v -= median;
            return 1;
        }
        if( v < median )
        {
            u -= median;
            return 3;
        }
        u -= median;
        v -= median;
        if( ( u + v ) < median )
        {
            rotated = true;
            return 2;
        }
    }
    return 0;
}

//
//  Weight adjustments to account for phantom end points:
//
inline __host__ __device__ void adjustBSplineBoundaryWeights( int boundary, float w[16] )
{

    if( ( boundary & 1 ) != 0 )
    {
        for( int i = 0; i < 4; ++i )
        {
            w[i + 8] -= w[i + 0];
            w[i + 4] += w[i + 0] * 2.0f;
            w[i + 0] = 0.0f;
        }
    }
    if( ( boundary & 2 ) != 0 )
    {
        for( int i = 0; i < 16; i += 4 )
        {
            w[i + 1] -= w[i + 3];
            w[i + 2] += w[i + 3] * 2.0f;
            w[i + 3] = 0.0f;
        }
    }
    if( ( boundary & 4 ) != 0 )
    {
        for( int i = 0; i < 4; ++i )
        {
            w[i + 4] -= w[i + 12];
            w[i + 8] += w[i + 12] * 2.0f;
            w[i + 12] = 0.0f;
        }
    }
    if( ( boundary & 8 ) != 0 )
    {
        for( int i = 0; i < 16; i += 4 )
        {
            w[i + 2] -= w[i + 0];
            w[i + 1] += w[i + 0] * 2.0f;
            w[i + 0] = 0.0f;
        }
    }
}


inline __host__ __device__ void boundBasisBSpline( int boundary, float wP[16], float wDs[16], float wDt[16] )
{
    adjustBSplineBoundaryWeights( boundary, wP );

    adjustBSplineBoundaryWeights( boundary, wDs );
    adjustBSplineBoundaryWeights( boundary, wDt );
}

inline __host__ __device__ void adjustBSplineBoundaryTop( float w[4] )
{
    w[1] -= w[3];
    w[2] += w[3] * 2.0f;
    w[3] = 0.0f;
}

inline __host__ __device__ void adjustBSplineBoundaryBottom( float w[4] )
{
    w[2] -= w[0];
    w[1] += w[0] * 2.0f;
    w[0] = 0.0f;
}


//
//  Cubic BSpline curve basis evaluation:
//
inline __host__ __device__ void evalBSplineCurve( float t, float wP[4], float wDP[4], float wDP2[4] )
{
    float const one6th = (float)( 1.0 / 6.0 );

    float t2 = t * t;
    float t3 = t * t2;

    wP[0] = one6th * ( 1.0f - 3.0f * ( t - t2 ) - t3 );
    wP[1] = one6th * ( 4.0f - 6.0f * t2 + 3.0f * t3 );
    wP[2] = one6th * ( 1.0f + 3.0f * ( t + t2 - t3 ) );
    wP[3] = one6th * ( t3 );

    if( wDP )
    {
        wDP[0] = -0.5f * t2 + t - 0.5f;
        wDP[1] = 1.5f * t2 - 2.0f * t;
        wDP[2] = -1.5f * t2 + t + 0.5f;
        wDP[3] = 0.5f * t2;
    }
    if( wDP2 )
    {
        wDP2[0] = -t + 1.0f;
        wDP2[1] = 3.0f * t - 2.0f;
        wDP2[2] = -3.0f * t + 1.0f;
        wDP2[3] = t;
    }
}


inline __device__ float cubicBSplineWeight( float t, uint32_t index )
{
    // clang-format off
    static const float weights[] = {
        // cubic b-spline weight matrix
        1/6.0f, -3/6.0f,  3/6.0f, -1/6.0f,
        4/6.0f,  0/6.0f, -6/6.0f,  3/6.0f,
        1/6.0f,  3/6.0f,  3/6.0f, -3/6.0f,
        0/6.0f,  0/6.0f,  0/6.0f,  1/6.0f,
    };
    // clang-format on

    auto W = cuda::std::mdspan( weights, 4, 4 );  // weight matrix

    float result = 0.0f;
    float factor = 1.0f;
#pragma unroll
    for( uint32_t i = 0; i < 4; ++i )
    {
        result += W( index, i ) * factor;
        factor *= t;
    }
    return result;
}

inline __device__ float cubicBSplineDerivativeWeight( float t, uint32_t index )
{
    // clang-format off
    static const float derivative_weights[] = {
        // cubic b-spline derivative weight matrix
        -1/2.0f,  2/2.0f, -1/2.0f,
         0/2.0f, -4/2.0f,  3/2.0f,
         1/2.0f,  2/2.0f, -3/2.0f,
         0/2.0f,  0/2.0f,  1/2.0f
    };
    // clang-format on

    auto D = cuda::std::mdspan( derivative_weights, 4, 3 );  // derivative weight matrix

    float result = 0.0f;
    float factor = 1.0f;
#pragma unroll
    for( uint32_t i = 0; i < 3; ++i )
    {
        result += D( index, i ) * factor;
        factor *= t;
    }
    return result;
}

inline __host__ __device__ float mix( float s1, float s2, float t )
{
    return ( (float)1.0 - t ) * s1 + t * s2;
}


inline __host__ __device__ void flipMatrix( float const* a, float* m )
{
    m[0]  = a[15];
    m[1]  = a[14];
    m[2]  = a[13];
    m[3]  = a[12];
    m[4]  = a[11];
    m[5]  = a[10];
    m[6]  = a[9];
    m[7]  = a[8];
    m[8]  = a[7];
    m[9]  = a[6];
    m[10] = a[5];
    m[11] = a[4];
    m[12] = a[3];
    m[13] = a[2];
    m[14] = a[1];
    m[15] = a[0];
}

inline __host__ __device__ void flipMatrix( float* m )
{
    cuda::std::swap(m[0]  , m[15]);
    cuda::std::swap(m[1]  , m[14]);
    cuda::std::swap(m[2]  , m[13]);
    cuda::std::swap(m[3]  , m[12]);
    cuda::std::swap(m[4]  , m[11]);
    cuda::std::swap(m[5]  , m[10]);
    cuda::std::swap(m[6]  , m[9]);
    cuda::std::swap(m[7]  , m[8]);
}

inline __host__ __device__ void flipMatrix( std::span<float> m )
{
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
    for(auto i = 0; i < 8; ++i)
        cuda::std::swap(m[i], m[15 - i]);
}



    // v x m (column major)
inline __host__ __device__ void applyMatrix( float* v, float const* m )
{
    float r[4];
    r[0] = v[0] * m[0] + v[1] * m[4] + v[2] * m[8] + v[3] * m[12];
    r[1] = v[0] * m[1] + v[1] * m[5] + v[2] * m[9] + v[3] * m[13];
    r[2] = v[0] * m[2] + v[1] * m[6] + v[2] * m[10] + v[3] * m[14];
    r[3] = v[0] * m[3] + v[1] * m[7] + v[2] * m[11] + v[3] * m[15];
    memcpy( v, r, 4 * sizeof( float ) );
}

inline __host__ __device__ void computeMixedCreaseMatrix( float sharp1, float sharp2, float t, float tInf, std::span<float> m )
{
    float s1 = (float)exp2( sharp1 ), s2 = (float)exp2( sharp2 );

    float sOver3 = mix( s1, s2, t ) / float( 3 ), oneOverS1 = (float)1 / s1, oneOverS2 = (float)1 / s2,
          oneOver6S = mix( oneOverS1, oneOverS2, t ) / (float)6, sSqr = mix( s1 * s1, s2 * s2, t );

    float A = -sSqr + sOver3 * (float)5.5 + oneOver6S - (float)1.0, B = sOver3 + oneOver6S + (float)0.5,
          C = sOver3 - oneOver6S * (float)2.0 + (float)1.0, E = sOver3 + oneOver6S - (float)0.5,
          F = -sOver3 * (float)0.5 + oneOver6S;

    m[0]  = (float)1.0;
    m[1]  = A * tInf;
    m[2]  = (float)-2.0 * A * tInf;
    m[3]  = A * tInf;
    m[4]  = (float)0.0;
    m[5]  = mix( (float)1.0, B, tInf );
    m[6]  = (float)-2.0 * E * tInf;
    m[7]  = E * tInf;
    m[8]  = (float)0.0;
    m[9]  = F * tInf;
    m[10] = mix( (float)1.0, C, tInf );
    m[11] = F * tInf;
    m[12] = (float)0.0;
    m[13] = mix( (float)-1.0, E, tInf );
    m[14] = mix( (float)2.0, -(float)2.0 * E, tInf );
    m[15] = B * tInf;
}

// compute the "crease matrix" for modifying basis weights at parametric
// location 't', given a sharpness value (see Matthias Niessner derivation
// for 'single crease' regular patches)
inline __host__ __device__ void computeCreaseMatrix( float sharpness, float t, std::span<float> m )
{

    float sharpFloor = (float)floor( sharpness ), sharpCeil = sharpFloor + 1, sharpFrac = sharpness - sharpFloor;

    float creaseWidthFloor = (float)1.0 - exp2( -sharpFloor ), creaseWidthCeil = (float)1.0 - exp2( -sharpCeil );

    // we compute the matrix for both the floor and ceiling of
    // the sharpness value, and then interpolate between them
    // as needed.
    float tA = ( t > creaseWidthCeil ) ? sharpFrac : (float)0.0, tB = (float)0.0;
    if( t > creaseWidthFloor )
        tB = (float)1.0 - sharpFrac;
    if( t > creaseWidthCeil )
        tB = (float)1.0;

    computeMixedCreaseMatrix( sharpFloor, sharpCeil, tA, tB, m );
}

// compute the "crease matrix" for modifying basis weights at parametric
// location 't', given a sharpness value (see Matthias Niessner derivation
// for 'single crease' regular patches)
inline __host__ __device__ void computeCreaseMatrixTop( float sharpness, float t, std::span<float> m )
{

    float sharpFloor = (float)floor( sharpness ), sharpCeil = sharpFloor + 1, sharpFrac = sharpness - sharpFloor;
    float creaseWidthFloor = (float)1.0 - exp2( -sharpFloor ), creaseWidthCeil = (float)1.0 - exp2( -sharpCeil );

    // we compute the matrix for both the floor and ceiling of
    // the sharpness value, and then interpolate between them
    // as needed.
    float tA = ( t > creaseWidthCeil ) ? sharpFrac : 0.0f;
    float tB = 0.0f;
    if( t > creaseWidthFloor )
        tB = 1.0f - sharpFrac;
    if( t > creaseWidthCeil )
        tB = 1.0f;

    computeMixedCreaseMatrix( sharpFloor, sharpCeil, tA, tB, m );
}


inline __host__ __device__ void computeCreaseMatrixBottom( float sharpness, float t, std::span<float> m )
{
    computeCreaseMatrixTop( sharpness, 1.0f - t, m );
    flipMatrix( m );
}


template <bool PURE_BSPLINE>
__host__ __device__ int EvalBasisBSpline( float2 st, float wP[16], float wDs[16], float wDt[16], int boundaryMask, float sharpness )
{

    float sWeights[4], tWeights[4], dsWeights[4], dtWeights[4];

    evalBSplineCurve( st.x, wP ? sWeights : 0, wDs ? dsWeights : 0, 0 );
    evalBSplineCurve( st.y, wP ? tWeights : 0, wDt ? dtWeights : 0, 0 );

    if( ( boundaryMask != 0 && sharpness > (float)0.0 ) && !PURE_BSPLINE )
    {
        float m[16], mflip[16];
        if( boundaryMask & 1 )
        {
            computeCreaseMatrix( sharpness, (float)1.0 - st.y, m );
            flipMatrix( m, mflip );
            applyMatrix( tWeights, mflip );
            applyMatrix( dtWeights, mflip );
        }
        if( boundaryMask & 2 )
        {
            computeCreaseMatrix( sharpness, st.x, m );
            applyMatrix( sWeights, m );
            applyMatrix( dsWeights, m );
        }
        if( boundaryMask & 4 )
        {
            computeCreaseMatrix( sharpness, st.y, m );
            applyMatrix( tWeights, m );
            applyMatrix( dtWeights, m );
        }
        if( boundaryMask & 8 )
        {
            computeCreaseMatrix( sharpness, (float)1.0 - st.x, m );
            flipMatrix( m, mflip );
            applyMatrix( sWeights, mflip );
            applyMatrix( dsWeights, mflip );
        }
    }

    for( int i = 0; i < 4; ++i )
    {
        for( int j = 0; j < 4; ++j )
        {
            wP[4 * i + j]  = sWeights[j] * tWeights[i];
            wDs[4 * i + j] = dsWeights[j] * tWeights[i];
            wDt[4 * i + j] = sWeights[j] * dtWeights[i];
        }
    }

    return 16;
}

inline __device__ void adjustBoundaries( float2 st, float sWeights[4], float tWeights[4], float dsWeights[4], float dtWeights[4], int boundaryMask )
{
    if( ( boundaryMask & 1 ) != 0 )
    {
        adjustBSplineBoundaryBottom( tWeights );
        adjustBSplineBoundaryBottom( dtWeights );
    }
    if( ( boundaryMask & 4 ) != 0 )
    {
        adjustBSplineBoundaryTop( tWeights );
        adjustBSplineBoundaryTop( dtWeights );
    }
    if( ( boundaryMask & 2 ) != 0 )
    {
        adjustBSplineBoundaryTop( sWeights );
        adjustBSplineBoundaryTop( dsWeights );
    }
    if( ( boundaryMask & 8 ) != 0 )
    {
        adjustBSplineBoundaryBottom( sWeights );
        adjustBSplineBoundaryBottom( dsWeights );
    }
}

inline __device__ void adjustCreases( float2 st, float sWeights[4], float tWeights[4], float dsWeights[4], float dtWeights[4], int boundaryMask, float sharpness )
{
    if( boundaryMask & 1 )
    {
        float m[16];
        computeCreaseMatrixBottom( sharpness, st.y, m );
        applyMatrix( tWeights, m );
        applyMatrix( dtWeights, m );
    }
    if( boundaryMask & 4 )
    {
        float m[16];
        computeCreaseMatrixTop( sharpness, st.y, m );
        applyMatrix( tWeights, m );
        applyMatrix( dtWeights, m );
    }
    if( boundaryMask & 2 )
    {
        float m[16];
        computeCreaseMatrixTop( sharpness, st.x, m );
        applyMatrix( sWeights, m );
        applyMatrix( dsWeights, m );
    }
    if( boundaryMask & 8 )
    {
        float m[16];
        computeCreaseMatrixBottom( sharpness, st.x, m );
        applyMatrix( sWeights, m );
        applyMatrix( dsWeights, m );
    }
}

inline __device__ float bSplinePatchWeight_warp( float2 uv, int2 ij )
{
    return cubicBSplineWeight( uv.x, ij.x ) * cubicBSplineWeight( uv.y, ij.y );
}

inline __device__ float bSplinePatchWeight_ds_warp( float2 uv, int2 ij )
{
    return cubicBSplineDerivativeWeight( uv.x, ij.x ) * cubicBSplineWeight( uv.y, ij.y );
}

inline __device__ float bSplinePatchWeight_dt_warp( float2 uv, int2 ij )
{
    return cubicBSplineWeight( uv.x, ij.x ) * cubicBSplineDerivativeWeight( uv.y, ij.y );
}


//
//  Higher level basis evaluation functions that deal with parameterization and
//  boundary issues (reflected in PatchParam) for all patch types:
//
inline __host__ __device__ int EvaluatePatchBasisNormalized( int                                patchType,
                                                             OpenSubdiv::Far::PatchParam const& param,
                                                             float2                             st,
                                                             float                              wP[],
                                                             float                              wDs[],
                                                             float                              wDt[],
                                                             float                              sharpness )
{
    using namespace OpenSubdiv::Far;

    int boundaryMask = param.GetBoundary();

    int nPoints = 0;
    if( patchType == PatchDescriptor::REGULAR )
    {
        nPoints = EvalBasisBSpline<false>( st, wP, wDs, wDt, boundaryMask, sharpness );
        if( boundaryMask && ( sharpness == float( 0 ) ) )
        {
            boundBasisBSpline( boundaryMask, wP, wDs, wDt );
        }
    }
    else
    {
        assert( false );
    }
    return nPoints;
}


inline __device__ void adjustBSplineBounds( float w[4], float dw[4], int boundaryMask )
{
    assert( boundaryMask != 0 );      // must have boundary situations

    if( ( boundaryMask & 1 ) != 0 )
    {
        adjustBSplineBoundaryBottom( w );
        adjustBSplineBoundaryBottom( dw );
    }
    if( ( boundaryMask & 4 ) != 0 )
    {
        adjustBSplineBoundaryTop( w );
        adjustBSplineBoundaryTop( dw );
    }
}

inline __device__ void adjustBSplineCreases( float t, float w[4], float dw[4], int boundaryMask, float sharpness )
{
    assert( !( sharpness < 0.0f ) );  // sharpness must not be negative
    assert( boundaryMask != 0 );      // must have boundary situations

    float m[16], mflip[16];
    if( boundaryMask & 1 )
    {
        computeCreaseMatrix( sharpness, 1.0f - t, m );
        flipMatrix( m, mflip );
        applyMatrix( w, mflip );
        applyMatrix( dw, mflip );
    }
    if( boundaryMask & 4 )
    {
        computeCreaseMatrix( sharpness, t, m );
        applyMatrix( w, m );
        applyMatrix( dw, m );
    }
}


inline __host__ __device__ int EvaluatePatchBasis( int                                patchType,
                                                   OpenSubdiv::Far::PatchParam const& param,
                                                   float2                             st,
                                                   float                              wP[],
                                                   float                              wDs[],
                                                   float                              wDt[],
                                                   float                              sharpness = 0 )
{
    using namespace OpenSubdiv::Far;

    float derivSign = 1.0f;

    if( ( patchType == PatchDescriptor::LOOP ) || ( patchType == PatchDescriptor::GREGORY_TRIANGLE )
        || ( patchType == PatchDescriptor::TRIANGLES ) )
    {
        param.NormalizeTriangle( st.x, st.y );
        if( param.IsTriangleRotated() )
        {
            derivSign = -1.0f;
        }
    }
    else
    {
        param.Normalize( st.x, st.y );
    }

    int nPoints = EvaluatePatchBasisNormalized( patchType, param, st, wP, wDs, wDt, sharpness );

    float d1Scale = derivSign * (float)( 1 << param.GetDepth() );

    for( int i = 0; i < nPoints; ++i )
    {
        wDs[i] *= d1Scale;
        wDt[i] *= d1Scale;
    }

    return nPoints;
}

struct SubdivisionPlanCUDA
{
    uint16_t _numControlPoints = 0;

    // note: schemes & end-cap maths should not be dynamic conditional paths in
    // the run-time kernels, so both of these should be moved out of this struct
    //
    // note: we can save 8 bytes on the struct size by switching these
    // enums to bit-fields ; however, the current size is 64B, so 2 plans
    // line up w/ a cache line size.
    OpenSubdiv::Sdc::SchemeType _scheme = Sdc::SCHEME_CATMARK;
    OpenSubdiv::Tmr::EndCapType _endCap = Tmr::EndCapType::ENDCAP_NONE;

    uint16_t _coarseFaceSize     = 4;
    int16_t  _coarseFaceQuadrant = -1;  // locates a surface within a non-quad parent face

    std::span<uint32_t> _tree;
    static_assert( sizeof( OpenSubdiv::Tmr::NodeDescriptor ) == sizeof( decltype( _tree )::value_type ) );

    // Stencil indices into the _weights
    std::span<int> _patchPoints;

    // Stencil matrix for computing patch points from control points:
    // - columns contain 1 scalar weight per control point of the 1-ring
    // - rows contain a stencil of weights for each patch point
    std::span<float> _stencilMatrix;

    __device__ Tmr::TreeDescriptor const* GetTreeDescriptor() const
    {
        return reinterpret_cast<Tmr::TreeDescriptor const*>( _tree.data() );
    }

    struct Node : public OpenSubdiv::Tmr::NodeBase
    {
        SubdivisionPlanCUDA const* pPlan = nullptr;

        __device__ Node( SubdivisionPlanCUDA const* pPlan = nullptr, int treeOffset = -1 )
            : pPlan( pPlan )
        {
            this->treeOffset = treeOffset;
        }

        /// \brief Returns false if un-initialized
        bool IsValid() const { return pPlan && treeOffset >= 0; }

        /// \brief Returns a pointer to the plan that owns this node
        __device__ SubdivisionPlanCUDA const* GetSubdivisionPlan() const { return pPlan; }

        /// \brief Returns the node's descriptor
        __device__ inline OpenSubdiv::Tmr::NodeDescriptor GetDescriptor() const
        {
            return Tmr::NodeDescriptor( static_cast<uint32_t>( pPlan->_tree[descriptorOffset()] ) );
        }

        /// \brief Returns the crease sharpness of the node is 'Regular' and
        /// flagged as a 'single-crease' patch
        __device__ inline float GetSharpness() const
        {
            assert( GetDescriptor().GetType() == Tmr::NodeType::NODE_REGULAR && GetDescriptor().HasSharpness() );

            uint32_t const* tree = pPlan->_tree.data();
            return *reinterpret_cast<float const*>( tree + sharpnessOffset() );
        }

        /// \brief Returns the number of children of the node
        inline int GetNumChildren() const;

        /// \brief Returns the child of the node
        inline Node GetChild( int childIndex = 0 ) const;

        /// \brief Returns the number of patch points points supporting the sub-patch
        __device__ int GetPatchSize( int quadrant, unsigned short maxLevel = OpenSubdiv::Tmr::kMaxIsolationLevel + 1 ) const
        {
            Sdc::SchemeType scheme = GetSubdivisionPlan()->getSchemeType();

            int regularPatchSize   = 0;
            int irregularPatchSize = 0;

            if( scheme == Sdc::SCHEME_CATMARK )
            {
                regularPatchSize   = catmarkRegularPatchSize();
                irregularPatchSize = catmarkIrregularPatchSize( pPlan->getEndCapType() );
            }
            else if( scheme == Sdc::SCHEME_LOOP )
            {
                regularPatchSize   = loopRegularPatchSize();
                irregularPatchSize = loopIrregularPatchSize( pPlan->getEndCapType() );
            }

            Tmr::NodeDescriptor desc = GetDescriptor();

            int numPatchPoints = 0;
            switch( desc.GetType() )
            {

                using enum Tmr::NodeType;

                case NODE_REGULAR:
                    numPatchPoints = regularPatchSize;
                    break;

                case NODE_END:
                    numPatchPoints = irregularPatchSize;
                    break;

                case NODE_RECURSIVE:
                    numPatchPoints = desc.HasEndcap() ? irregularPatchSize : 0;
                    break;
            }
            return numPatchPoints;
        }


        /// \brief Returns the index of the requested patch point in the sub-patch
        /// described by the node (pointIndex is in range [0, GetPatchSize()])
        ///
        /// @oaram pointIndex  Index of the patch point
        /// @param quadrant    One of the 3 quadrants of a Terminal patch that are
        ///                    not the quadrant containing the extraordinary vertex
        /// @param level       Desired dynamic level of isolation
        __device__ OpenSubdiv::Tmr::Index GetPatchPoint( int            pointIndex,
                                                         int            quadrant = 0,
                                                         unsigned short maxLevel = OpenSubdiv::Tmr::kMaxIsolationLevel + 1 ) const
        {
            uint32_t const* tree = pPlan->_tree.data();
            int offset = tree[patchPointsOffset()];

            if( offset == Tmr::INDEX_INVALID )
                return Tmr::INDEX_INVALID;

            Tmr::NodeDescriptor desc = GetDescriptor();
            switch( desc.GetType() )
            {
                using enum Tmr::NodeType;
                case NODE_REGULAR:
                case NODE_END:
                    offset += pointIndex;
                    break;
                case NODE_RECURSIVE:
                    offset = ( desc.GetDepth() >= maxLevel ) && desc.HasEndcap() ? offset + pointIndex : Tmr::INDEX_INVALID;
            }
            assert( offset != Tmr::INDEX_INVALID );
            return pPlan->_patchPoints[offset];
        }

        /// \brief Returns the row corresponding to the patch point in the
        /// dense stencil matrix
        OpenSubdiv::Tmr::ConstFloatArray GetPatchPointWeights( int pointIndex ) const;

        /// \brief Returns the next node in the tree (serial traversal). Check
        //  IsValid() for iteration past end-of-tree
        Node operator++();

        /// \brief Returns true if the nodes are identical
        bool operator==( Node const& other ) const;

        void PrintStencil() const;
    };

    template <typename T, typename U>
    __device__ void WarpEvaluatePatchPoints( U*                  patchPoints,
                                             const std::span<T>& controlPoints,
                                             const Tmr::Index*   controlPointIndices,
                                             const uint8_t       level ) const
    {
        assert( level <= Tmr::kMaxIsolationLevel );
        const uint32_t numPatchPoints = GetTreeDescriptor()->numPatchPoints[level];
        const auto     stencilMatrix  = cuda::std::mdspan( _stencilMatrix.data(), numPatchPoints, _numControlPoints );
        for( auto i_patch_point = threadIdx.x; i_patch_point < stencilMatrix.extent( 0 ); i_patch_point += 32 )  // advance warp
        {
            auto patchPoint = U{};
            for( auto i = 0; i < _numControlPoints; ++i )
                patchPoint += controlPoints[controlPointIndices[i]] * stencilMatrix( i_patch_point, i );
            patchPoints[i_patch_point] = patchPoint;
        }
    }

    // note: this is not an adequate test to check that this is a 'pure' regular bspline patch ; it is possible
    // for a patch to not require patch-points, but still have edge boundaries or corners. This patches have
    // fewer rows in the stencil matrix and require the correct boundary mask to be set when computing the basis
    // matrix.
    __device__ bool isBSplinePatch( int level ) const { return GetTreeDescriptor()->numPatchPoints[level] == 0; }

    __device__ Node GetNode( float2 uv, unsigned char& quadrant, unsigned short level ) const
    {
        using namespace OpenSubdiv;
        using enum Tmr::NodeType;

        // traverse the sub-patch tree to the (s,t) coordinates

        Sdc::SchemeType scheme = getSchemeType();

        Tmr::NodeBase node = { .treeOffset = Tmr::NodeBase::rootNodeOffset() };

        Tmr::NodeDescriptor desc = Tmr::NodeDescriptor( _tree[node.descriptorOffset()] );
        Tmr::NodeType       type = desc.GetType();

        bool  is_irregular = !GetTreeDescriptor()->IsRegularFace();
        bool  rotated      = false;
        float median       = 0.5f;

        while( type == NODE_RECURSIVE )
        {
            if( (short)desc.GetDepth() == level )
                break;

            switch( scheme )
            {
                case Sdc::SchemeType::SCHEME_CATMARK:
                    traverseCatmark( uv.x, uv.y, quadrant );
                    break;
                case Sdc::SchemeType::SCHEME_LOOP:
                    quadrant = traverseLoop( median, uv.x, uv.y, rotated );
                    break;
            }

            if( type == NODE_RECURSIVE )
            {
                // traverse to child node
                node = Tmr::NodeBase( _tree[node.childOffset( quadrant )] );
            }

            desc = Tmr::NodeDescriptor( _tree[node.descriptorOffset()] );
            type = desc.GetType();
            median *= 0.5f;
        }
        return Node( this, node.treeOffset );
    }

    __device__ Node getRootNode() const
    {
        using namespace OpenSubdiv;

        Tmr::NodeBase node = { .treeOffset = Tmr::NodeBase::rootNodeOffset() };
        return Node( this, node.treeOffset );
    }


    __device__ Node EvaluateBasis( float2 st, float wP[], float wDs[], float wDt[], unsigned char* subpatch, short level ) const
    {
        using namespace OpenSubdiv;

        Sdc::SchemeType const            scheme         = getSchemeType();
        Far::PatchDescriptor::Type const regularBasis   = regularBasisType( scheme );
        Far::PatchDescriptor::Type const irregularBasis = irregularBasisType( scheme, getEndCapType() );

        bool is_irregular = !GetTreeDescriptor()->IsRegularFace();

        unsigned char quadrant = 0;
        Node          node     = GetNode( st, quadrant, level );

        Tmr::NodeDescriptor desc = node.GetDescriptor();

        Tmr::NodeType nodeType = desc.GetType();
        int           depth    = desc.GetDepth();

        using enum Tmr::NodeType;
        bool dynamicIsolation = ( nodeType == NODE_RECURSIVE ) && ( depth >= level ) && desc.HasEndcap();

        unsigned short u = desc.GetU();
        unsigned short v = desc.GetV();

        Far::PatchParam param;

        if( dynamicIsolation )
        {
            param.Set( Tmr::INDEX_INVALID, u, v, depth, is_irregular, 0, 0, true );
            EvaluatePatchBasis( irregularBasis, param, st, wP, wDs, wDt );
        }
        else
        {
            switch( nodeType )
            {

                case NODE_REGULAR: {
                    param.Set( Tmr::INDEX_INVALID, u, v, depth, is_irregular, desc.GetBoundaryMask(), 0, true );
                    float sharpness = desc.HasSharpness() ? node.GetSharpness() : 0.f;
                    EvaluatePatchBasis( regularBasis, param, st, wP, wDs, wDt, sharpness );
                }
                break;

                case NODE_END: {
                    param.Set( Tmr::INDEX_INVALID, u, v, depth, is_irregular, desc.GetBoundaryMask(), 0, true );
                    EvaluatePatchBasis( irregularBasis, param, st, wP, wDs, wDt );
                }
                break;

                default:
                    assert( 0 );
            }
        }
        if( subpatch )
            *subpatch = quadrant;

        return node;
    }

    __device__ Node WarpEvaluateBSplineBasis( float2 st, float wP[], float wDs[], float wDt[], unsigned char* subpatch, short level ) const
    {
        using namespace OpenSubdiv;

        Sdc::SchemeType const            scheme         = getSchemeType();
        Far::PatchDescriptor::Type const regularBasis   = regularBasisType( scheme );
        Far::PatchDescriptor::Type const irregularBasis = irregularBasisType( scheme, getEndCapType() );

        bool is_irregular = !GetTreeDescriptor()->IsRegularFace();

        unsigned char quadrant = 0;
        Node          node     = GetNode( st, quadrant, level );

        Tmr::NodeDescriptor desc = node.GetDescriptor();

        Tmr::NodeType nodeType = desc.GetType();
        int           depth    = desc.GetDepth();

        using enum Tmr::NodeType;

        unsigned short u = desc.GetU();
        unsigned short v = desc.GetV();
        printf( "u = %d, v = %d\n", u, v );

        Far::PatchParam param;

        switch( nodeType )
        {
            case NODE_REGULAR: {
                param.Set( Tmr::INDEX_INVALID, u, v, depth, is_irregular, desc.GetBoundaryMask(), 0, true );
                float sharpness = desc.HasSharpness() ? node.GetSharpness() : 0.f;
                EvaluatePatchBasis( regularBasis, param, st, wP, wDs, wDt, sharpness );
            }
            break;
            default:
                assert( 0 );
        }
        if( subpatch )
            *subpatch = quadrant;

        return node;
    }

  protected:
    __device__ OpenSubdiv::Sdc::SchemeType getSchemeType() const
    {
#if 0
        return GetTopologyMap()->GetTraits().GetSchemeType();
#else
        return _scheme;
#endif
    }

    __device__ OpenSubdiv::Tmr::EndCapType getEndCapType() const
    {
        return _endCap;
    }
};
