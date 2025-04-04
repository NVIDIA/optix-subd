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

#include <vector_functions.h>
#include <vector_types.h>

#if !defined( __CUDACC_RTC__ )
#include <cmath>
#include <cstdlib>
#endif

#include "Matrix.h"

namespace otk {
struct Affine
{
    Matrix<3, 3> linear;
    float3       translation;

    OTK_INLINE OTK_HOSTDEVICE static Affine identity() { return {Matrix<3, 3>::identity(), {0.f, 0.f, 0.f}}; }

    OTK_INLINE OTK_HOSTDEVICE static Affine from_cols(float3 col0, float3 col1, float3 col2, float3 translation);

    OTK_INLINE OTK_HOSTDEVICE float3 transformPoint( float3 const& p ) const { return linear * p + translation; }

    OTK_INLINE OTK_HOSTDEVICE float3 transformVector( float3 const& v ) const { return linear * v; }

    OTK_INLINE OTK_HOSTDEVICE otk::Matrix3x4 matrix3x4() const;

    static OTK_INLINE OTK_HOSTDEVICE Affine translate( float3 const& a );

    static OTK_INLINE OTK_HOSTDEVICE Affine scale( float3 const& a );

    static OTK_INLINE OTK_HOSTDEVICE Affine rotate( float3 const& euler );
};

OTK_INLINE OTK_HOSTDEVICE Affine Affine::from_cols(float3 col0, float3 col1, float3 col2, float3 translation)
{
    Affine result;
    result.linear.setCol(0, col0);
    result.linear.setCol(1, col1);
    result.linear.setCol(2, col2);
    result.translation = translation;
    return result;
}

OTK_INLINE OTK_HOSTDEVICE otk::Matrix3x4 Affine::matrix3x4() const
{
    otk::Matrix3x4 result;
    result.setRow( 0, make_float4( linear.getRow( 0 ), translation.x ) );
    result.setRow( 1, make_float4( linear.getRow( 1 ), translation.y ) );
    result.setRow( 2, make_float4( linear.getRow( 2 ), translation.z ) );
    return result;
}


OTK_INLINE OTK_HOSTDEVICE Affine Affine::translate(float3 const& a)
{
    return { Matrix3x3::identity(), a };
}

OTK_INLINE OTK_HOSTDEVICE Affine Affine::scale(float3 const& a)
{
    return { Matrix3x3{ a.x, 0.f, 0.f, 0.f, a.y, 0.f, 0.f, 0.f, a.z }, make_float3(0.) };
}


OTK_INLINE OTK_HOSTDEVICE Affine Affine::rotate(float3 const& euler)
{
    float sinX = sinf(euler.x);
    float cosX = cosf(euler.x);
    float sinY = sinf(euler.y);
    float cosY = cosf(euler.y);
    float sinZ = sinf(euler.z);
    float cosZ = cosf(euler.z);

    Matrix3x3 matX = {
        1.f,   0.f,  0.f,
        0.f,  cosX, -sinX,
        0.f,  sinX, cosX, };

    Matrix3x3 matY = {
        cosY, 0.f, sinY,
         0.f, 1.f, 0.f,
        -sinY, 0.f, cosY };

    Matrix3x3 matZ = {
         cosZ, -sinZ, 0.f,
         sinZ, cosZ, 0.f,
          0.f,  0.f, 1.f };

    return { matX * matY * matZ, make_float3(0.f) };
}

OTK_INLINE OTK_HOSTDEVICE Affine operator*( Affine const& a, Affine const& b )
{
    return { a.linear * b.linear, a.linear * b.translation + a.translation };
}

OTK_INLINE OTK_HOSTDEVICE Affine operator*=( Affine& a, Affine const& b )
{
    a = a * b;
    return a;
}

OTK_INLINE OTK_HOSTDEVICE Affine fromCols( float3 const& col0, float3 const& col1, float3 const& col2, float3 const& translation )
{
    return {{
                col0.x, col1.x, col2.x,
                col0.y, col1.y, col2.y,
                col0.z, col1.z, col2.z,
            }, { translation }};
}

OTK_INLINE OTK_HOSTDEVICE Affine transpose( Affine const& a )
{
    Matrix<3, 3> transposed = a.linear.transpose();
    return { transposed, transposed * -a.translation };
}

OTK_INLINE OTK_HOSTDEVICE Affine inverse( Affine const& a )
{
    Matrix<3, 3> inverted = a.linear.inverse( );
    return { inverted, inverted * -a.translation };
}

OTK_INLINE OTK_HOSTDEVICE Matrix4x4 toHomogeneous( Affine const& a )
{
    Matrix4x4 result;
    for( int i = 0; i < 3; ++i )
        result.setCol( i, make_float4(a.linear.getCol(i), 0.f) );
    result.setCol( 3, { a.translation.x, a.translation.y, a.translation.z, 1.0f } );
    return result;
}

inline otk::Affine fromHomogeneous( const otk::Matrix4x4& m )
{
    return otk::fromCols( make_float3( m.getCol( 0 ) ), 
                          make_float3( m.getCol( 1 ) ), 
                          make_float3( m.getCol( 2 ) ), 
                          make_float3( m.getCol( 3 ) ) );
        
}


}  // namespace otk
