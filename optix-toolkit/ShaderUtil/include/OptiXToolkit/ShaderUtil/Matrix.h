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

#include <OptiXToolkit/ShaderUtil/Preprocessor.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#if !defined(__CUDACC_RTC__)
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <type_traits>
#endif

#define RT_MAT_DECL template <unsigned int M, unsigned int N>

namespace otk
{

  template <unsigned int DIM> struct VectorDim { };
  template <> struct VectorDim<2> { typedef float2 VectorType; };
  template <> struct VectorDim<3> { typedef float3 VectorType; };
  template <> struct VectorDim<4> { typedef float4 VectorType; };


  template <unsigned int M, unsigned int N> class Matrix;

   template <unsigned int M> OTK_INLINE OTK_HOSTDEVICE Matrix<M,M>& operator*=(Matrix<M,M>& m1, const Matrix<M,M>& m2);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE bool         operator==(const Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE bool         operator!=(const Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE Matrix<M,N>& operator-=(Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE Matrix<M,N>& operator+=(Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE Matrix<M,N>& operator*=(Matrix<M,N>& m1, float f);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE Matrix<M,N>&operator/=(Matrix<M,N>& m1, float f);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE Matrix<M,N> operator-(const Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE Matrix<M,N> operator+(const Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE Matrix<M,N> operator/(const Matrix<M,N>& m, float f);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE Matrix<M,N> operator*(const Matrix<M,N>& m, float f);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE Matrix<M,N> operator*(float f, const Matrix<M,N>& m);
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE typename Matrix<M,N>::floatM operator*(const Matrix<M,N>& m, const typename Matrix<M,N>::floatN& v );
   RT_MAT_DECL OTK_INLINE OTK_HOSTDEVICE typename Matrix<M,N>::floatN operator*(const typename Matrix<M,N>::floatM& v, const Matrix<M,N>& m);
   template<unsigned int M, unsigned int N, unsigned int R> OTK_INLINE OTK_HOSTDEVICE Matrix<M,R> operator*(const Matrix<M,N>& m1, const Matrix<N,R>& m2);


  // Partial specializations to make matrix vector multiplication more efficient
  template <unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE float2 operator*(const Matrix<2,N>& m, const typename Matrix<2,N>::floatN& vec );
  template <unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE float3 operator*(const Matrix<3,N>& m, const typename Matrix<3,N>::floatN& vec );
  OTK_INLINE OTK_HOSTDEVICE float3 operator*(const Matrix<3,4>& m, const float4& vec );
  template <unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE float4 operator*(const Matrix<4,N>& m, const typename Matrix<4,N>::floatN& vec );
  OTK_INLINE OTK_HOSTDEVICE float4 operator*(const Matrix<4,4>& m, const float4& vec );

  /** Shortcut for specializing functions only for specific matrix sizes */
  // TODO: this would be better handled with template specialization and
  // inheriting from a common detail:: type
  template <class T, class U>
  using if_same_t = std::enable_if_t<std::is_same_v<T, U>, T>;

  /**
  * @brief A matrix with M rows and N columns
  *
  * @ingroup CUDACTypes
  *
  * <B>Description</B>
  *
  * @ref Matrix provides a utility class for small-dimension floating-point
  * matrices, such as transformation matrices.  @ref Matrix may also be useful
  * in other computation and can be used in both host and device code.
  * Typedefs are provided for 2x2 through 4x4 matrices.
  *
  */
  template <unsigned int M, unsigned int N>
  class Matrix
  {
  public:
    typedef typename VectorDim<N>::VectorType  floatN; /// A row of the matrix
    typedef typename VectorDim<M>::VectorType  floatM; /// A column of the matrix

    /** Create an uninitialized matrix */
    OTK_HOSTDEVICE              Matrix();

    OTK_HOSTDEVICE explicit     Matrix( const float f ) { for (unsigned int i = 0; i < M * N; ++i) m_data[i] = f; }

    /** Create a matrix from the specified float array */
    OTK_HOSTDEVICE explicit     Matrix( const float data[M*N] ) { for(unsigned int i = 0; i < M*N; ++i) m_data[i] = data[i]; }

    /** Copy the matrix */
    OTK_HOSTDEVICE              Matrix( const Matrix& m );

    /** Conversion between matrices */
    template <unsigned int O, unsigned int P>
    OTK_HOSTDEVICE static Matrix makeFrom( const Matrix<O, P>& m );

    OTK_HOSTDEVICE              Matrix( const std::initializer_list<float>& list );

    /** Assignment operator */
    OTK_HOSTDEVICE Matrix&      operator=( const Matrix& b );

    /** Access the specified element 0..N*M-1  */
    OTK_HOSTDEVICE float        operator[]( unsigned int i )const { return m_data[i]; }

    /** Access the specified element 0..N*M-1  */
    OTK_HOSTDEVICE float&       operator[]( unsigned int i )      { return m_data[i]; }

    /** Access the specified row 0..M.  Returns float, float2, float3 or float4 depending on the matrix size  */
    OTK_HOSTDEVICE floatN       getRow( unsigned int m )const;

    /** Access the specified column 0..N.  Returns float, float2, float3 or float4 depending on the matrix size */
    OTK_HOSTDEVICE floatM       getCol( unsigned int n )const;

    OTK_HOSTDEVICE constexpr float&       get( unsigned int i, unsigned int j ) { return m_data[i * N + j]; }
    OTK_HOSTDEVICE constexpr const float& get( unsigned int i, unsigned int j ) const { return m_data[i * N + j]; }

    /** Returns a pointer to the internal data array.  The data array is stored in row-major order. */
    OTK_HOSTDEVICE float*       getData();

    /** Returns a const pointer to the internal data array.  The data array is stored in row-major order. */
    OTK_HOSTDEVICE const float* getData()const;

    /** Assign the specified row 0..M.  Takes a float, float2, float3 or float4 depending on the matrix size */
    OTK_HOSTDEVICE void         setRow( unsigned int m, const floatN &r );

    /** Assign the specified column 0..N.  Takes a float, float2, float3 or float4 depending on the matrix size */
    OTK_HOSTDEVICE void         setCol( unsigned int n, const floatM &c );

    /** Returns the transpose of the matrix */
    OTK_HOSTDEVICE Matrix<N, M>        transpose() const;

    /** Returns the inverse of the matrix */
    OTK_HOSTDEVICE Matrix              inverse() const;

    /** Returns the determinant of the matrix */
    OTK_HOSTDEVICE float               det() const;

    /** Returns a rotation matrix */
    template<class T = Matrix>
    OTK_HOSTDEVICE static if_same_t<T, Matrix<4, 4>> rotate(const float radians, const float3& axis);

    /** Returns a translation matrix */
    template<class T = Matrix>
    OTK_HOSTDEVICE static if_same_t<T, Matrix<4, 4>> translate(const float3& vec);

    /** Returns a scale matrix */
    template<class T = Matrix>
    OTK_HOSTDEVICE static if_same_t<T, Matrix<4, 4>> scale(const float3& vec);

    /** Creates a matrix from an ONB and center point */
    template<class T = Matrix>
    OTK_HOSTDEVICE static if_same_t<T, Matrix<4, 4>> fromBasis( const float3& u, const float3& v, const float3& w, const float3& c );

    /** Returns diagonal(1.0f) for 3x4 matrices */
    template<class T = Matrix>
    OTK_HOSTDEVICE static if_same_t<T, Matrix<3, 4>> affineIdentity();

    /** Returns a matrix with 'v' diagonal values and 0.0f elsewhere */
    OTK_HOSTDEVICE static Matrix diagonal(float v);

    /** Returns the identity matrix (square matrices only) */
    template<class T = Matrix>
    OTK_HOSTDEVICE static if_same_t<T, Matrix<N, N>> identity();

    /** Ordered comparison operator so that the matrix can be used in an STL container */
    OTK_HOSTDEVICE bool         operator<( const Matrix<M, N>& rhs ) const;


  private:
    /** The data array is stored in row-major order */
    float m_data[M*N];
  };


  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE Matrix<M,N>::Matrix()
  {
  }


  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE Matrix<M,N>::Matrix( const Matrix<M,N>& m )
  {
    for(unsigned int i = 0; i < M*N; ++i)
      m_data[i] = m[i];
  }

  template <unsigned int M, unsigned int N>
  template <unsigned int O, unsigned int P>
  OTK_INLINE OTK_HOSTDEVICE Matrix<M, N> Matrix<M, N>::makeFrom( const Matrix<O, P>& m )
  {
      Matrix result( Matrix<M, N>::diagonal( 1.0f ) );
      for( unsigned int i = 0; i < std::min( N, P ); ++i )
      {
          for( unsigned int j = 0; j < std::min( M, O ); ++j )
          {
              result.get( j, i ) = m.get( j, i );
          }
      }
      return result;
  }

  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE Matrix<M,N>::Matrix( const std::initializer_list<float>& list )
  {
      int i = 0;
      for( auto it = list.begin(); it != list.end(); ++it )
          m_data[ i++ ] = *it;
  }


  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE Matrix<M,N>&  Matrix<M,N>::operator=( const Matrix& b )
  {
    for(unsigned int i = 0; i < M*N; ++i)
      m_data[i] = b[i];
    return *this;
  }


  /*
  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE float Matrix<M,N>::operator[]( unsigned int i )const
  {
  return m_data[i];
  }


  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE float& Matrix<M,N>::operator[]( unsigned int i )
  {
  return m_data[i];
  }
  */

  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE typename Matrix<M,N>::floatN Matrix<M,N>::getRow( unsigned int m )const
  {
    typename Matrix<M,N>::floatN temp;
    float* v = reinterpret_cast<float*>( &temp );
    const float* row = &( m_data[m*N] );
    for(unsigned int i = 0; i < N; ++i)
      v[i] = row[i];

    return temp;
  }


  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE typename Matrix<M,N>::floatM Matrix<M,N>::getCol( unsigned int n )const
  {
    typename Matrix<M,N>::floatM temp;
    float* v = reinterpret_cast<float*>( &temp );
    for ( unsigned int i = 0; i < M; ++i )
      v[i] = get( i, n );

    return temp;
  }


  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE float* Matrix<M,N>::getData()
  {
    return m_data;
  }


  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE const float* Matrix<M,N>::getData() const
  {
    return m_data;
  }


  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE void Matrix<M,N>::setRow( unsigned int m, const typename Matrix<M,N>::floatN &r )
  {
    const float* v = reinterpret_cast<const float*>( &r );
    float* row = &( m_data[m*N] );
    for(unsigned int i = 0; i < N; ++i)
      row[i] = v[i];
  }


  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE void Matrix<M,N>::setCol( unsigned int n, const typename Matrix<M,N>::floatM &c )
  {
    const float* v = reinterpret_cast<const float*>( &c );
    for ( unsigned int i = 0; i < M; ++i )
      get( i, n ) = v[i];
  }


  // Compare two matrices using exact float comparison
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE bool operator==(const Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      if ( m1[i] != m2[i] ) return false;
    return true;
  }

  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE bool operator!=(const Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      if ( m1[i] != m2[i] ) return true;
    return false;
  }

  // Subtract two matrices of the same size.
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE Matrix<M,N> operator-(const Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    Matrix<M,N> temp( m1 );
    temp -= m2;
    return temp;
  }


  // Subtract two matrices of the same size.
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE Matrix<M,N>& operator-=(Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      m1[i] -= m2[i];
    return m1;
  }


  // Add two matrices of the same size.
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE Matrix<M,N> operator+(const Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    Matrix<M,N> temp( m1 );
    temp += m2;
    return temp;
  }


  // Add two matrices of the same size.
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE Matrix<M,N>& operator+=(Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      m1[i] += m2[i];
    return m1;
  }


  // Multiply two compatible matrices.
  template<unsigned int M, unsigned int N, unsigned int R>
  OTK_HOSTDEVICE Matrix<M,R> operator*( const Matrix<M,N>& m1, const Matrix<N,R>& m2)
  {
    Matrix<M,R> temp;

    for ( unsigned int i = 0; i < M; ++i ) {
      for ( unsigned int j = 0; j < R; ++j ) {
        float sum = 0.0f;
        for ( unsigned int k = 0; k < N; ++k ) {
          float ik = m1[ i*N+k ];
          float kj = m2[ k*R+j ];
          sum += ik * kj;
        }
        temp[i*R+j] = sum;
      }
    }
    return temp;
  }


  // Multiply two compatible matrices.
  template<unsigned int M>
  OTK_HOSTDEVICE Matrix<M,M>& operator*=(Matrix<M,M>& m1, const Matrix<M,M>& m2)
  {
    m1 = m1*m2;
    return m1;
  }


  // Multiply matrix by vector
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE typename Matrix<M,N>::floatM operator*(const Matrix<M,N>& m, const typename Matrix<M,N>::floatN& vec )
  {
    typename Matrix<M,N>::floatM temp;
    float* t = reinterpret_cast<float*>( &temp );
    const float* v = reinterpret_cast<const float*>( &vec );

    for (unsigned int i = 0; i < M; ++i) {
      float sum = 0.0f;
      for (unsigned int j = 0; j < N; ++j) {
          sum += m.get( i, j ) * v[j];
      }
      t[i] = sum;
    }

    return temp;
  }

  // Multiply matrix2xN by floatN
  template<unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE float2 operator*(const Matrix<2,N>& m, const typename Matrix<2,N>::floatN& vec )
  {
    float2 temp = { 0.0f, 0.0f };
    const float* v = reinterpret_cast<const float*>( &vec );

    int index = 0;
    for (unsigned int j = 0; j < N; ++j)
      temp.x += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.y += m[index++] * v[j];

    return temp;
  }

  // Multiply matrix3xN by floatN
  template<unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE float3 operator*(const Matrix<3,N>& m, const typename Matrix<3,N>::floatN& vec )
  {
    float3 temp = { 0.0f, 0.0f, 0.0f };
    const float* v = reinterpret_cast<const float*>( &vec );

    int index = 0;
    for (unsigned int j = 0; j < N; ++j)
      temp.x += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.y += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.z += m[index++] * v[j];

    return temp;
  }

  // Multiply matrix4xN by floatN
  template<unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE float4 operator*(const Matrix<4,N>& m, const typename Matrix<4,N>::floatN& vec )
  {
    float4 temp = { 0.0f, 0.0f, 0.0f, 0.0f };

    const float* v = reinterpret_cast<const float*>( &vec );

    int index = 0;
    for (unsigned int j = 0; j < N; ++j)
      temp.x += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.y += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.z += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.w += m[index++] * v[j];

    return temp;
  }

  // Multiply matrix4x4 by float4
  OTK_INLINE OTK_HOSTDEVICE float3 operator*(const Matrix<3,4>& m, const float4& vec )
  {
    float3 temp;
    temp.x  = m[ 0] * vec.x +
              m[ 1] * vec.y +
              m[ 2] * vec.z +
              m[ 3] * vec.w;
    temp.y  = m[ 4] * vec.x +
              m[ 5] * vec.y +
              m[ 6] * vec.z +
              m[ 7] * vec.w;
    temp.z  = m[ 8] * vec.x +
              m[ 9] * vec.y +
              m[10] * vec.z +
              m[11] * vec.w;
    return temp;
  }

  // Multiply matrix4x4 by float4
  OTK_INLINE OTK_HOSTDEVICE float4 operator*(const Matrix<4,4>& m, const float4& vec )
  {
    float4 temp;
    temp.x  = m[ 0] * vec.x +
              m[ 1] * vec.y +
              m[ 2] * vec.z +
              m[ 3] * vec.w;
    temp.y  = m[ 4] * vec.x +
              m[ 5] * vec.y +
              m[ 6] * vec.z +
              m[ 7] * vec.w;
    temp.z  = m[ 8] * vec.x +
              m[ 9] * vec.y +
              m[10] * vec.z +
              m[11] * vec.w;
    temp.w  = m[12] * vec.x +
              m[13] * vec.y +
              m[14] * vec.z +
              m[15] * vec.w;

    return temp;
  }

  // Multiply vector by matrix
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE typename Matrix<M,N>::floatN operator*(const typename Matrix<M,N>::floatM& vec, const Matrix<M,N>& m)
  {
    typename Matrix<M,N>::floatN  temp;
    float* t = reinterpret_cast<float*>( &temp );
    const float* v = reinterpret_cast<const float*>( &vec);

    for (unsigned int i = 0; i < N; ++i) {
      float sum = 0.0f;
      for (unsigned int j = 0; j < M; ++j) {
        sum += v[j] * m.get( j, i ) ;
      }
      t[i] = sum;
    }

    return temp;
  }


  // Multply matrix by a scalar.
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE Matrix<M,N> operator*(const Matrix<M,N>& m, float f)
  {
    Matrix<M,N> temp( m );
    temp *= f;
    return temp;
  }


  // Multply matrix by a scalar.
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE Matrix<M,N>& operator*=(Matrix<M,N>& m, float f)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      m[i] *= f;
    return m;
  }


  // Multply matrix by a scalar.
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE Matrix<M,N>  operator*(float f, const Matrix<M,N>& m)
  {
    Matrix<M,N> temp;

    for ( unsigned int i = 0; i < M*N; ++i )
      temp[i] = m[i]*f;

    return temp;
  }


  // Divide matrix by a scalar.
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE Matrix<M,N> operator/(const Matrix<M,N>& m, float f)
  {
    Matrix<M,N> temp( m );
    temp /= f;
    return temp;
  }


  // Divide matrix by a scalar.
  template<unsigned int M, unsigned int N>
  OTK_HOSTDEVICE Matrix<M,N>& operator/=(Matrix<M,N>& m, float f)
  {
    float inv_f = 1.0f / f;
    for ( unsigned int i = 0; i < M*N; ++i )
      m[i] *= inv_f;
    return m;
  }

  // Returns the transpose of the matrix.
  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE Matrix<N,M> Matrix<M,N>::transpose() const
  {
    Matrix<N,M> ret;
    for( unsigned int row = 0; row < M; ++row )
      for( unsigned int col = 0; col < N; ++col )
        ret[col*M+row] = m_data[row*N+col];
    return ret;
  }

  // Returns the determinant of the matrix.
  template<>
  OTK_INLINE OTK_HOSTDEVICE float Matrix<2,2>::det() const
  {
    const float* m = m_data;
    float d = m[0]*m[3] - m[1]*m[2];
    return d;
  }

  template<>
  OTK_INLINE OTK_HOSTDEVICE float Matrix<3,3>::det() const
  {
    const float* m   = m_data;
    float d = m[0]*m[4]*m[8] + m[1]*m[5]*m[6] + m[2]*m[3]*m[7]
      - m[0]*m[5]*m[7] - m[1]*m[3]*m[8] - m[2]*m[4]*m[6];
    return d;
  }

  template<>
  OTK_INLINE OTK_HOSTDEVICE float Matrix<4,4>::det() const
  {
    const float* m   = m_data;
    float d =
      m[0]*m[5]*m[10]*m[15]-
      m[0]*m[5]*m[11]*m[14]+m[0]*m[9]*m[14]*m[7]-
      m[0]*m[9]*m[6]*m[15]+m[0]*m[13]*m[6]*m[11]-
      m[0]*m[13]*m[10]*m[7]-m[4]*m[1]*m[10]*m[15]+m[4]*m[1]*m[11]*m[14]-
      m[4]*m[9]*m[14]*m[3]+m[4]*m[9]*m[2]*m[15]-
      m[4]*m[13]*m[2]*m[11]+m[4]*m[13]*m[10]*m[3]+m[8]*m[1]*m[6]*m[15]-
      m[8]*m[1]*m[14]*m[7]+m[8]*m[5]*m[14]*m[3]-
      m[8]*m[5]*m[2]*m[15]+m[8]*m[13]*m[2]*m[7]-
      m[8]*m[13]*m[6]*m[3]-
      m[12]*m[1]*m[6]*m[11]+m[12]*m[1]*m[10]*m[7]-
      m[12]*m[5]*m[10]*m[3]+m[12]*m[5]*m[2]*m[11]-
      m[12]*m[9]*m[2]*m[7]+m[12]*m[9]*m[6]*m[3];
    return d;
  }

  template <unsigned int n>
  OTK_INLINE OTK_HOSTDEVICE Matrix<n, n> inverse(Matrix<n, n> const& m)
  {
    constexpr float epsilon = 1e-6f;
    constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

    // Gaussian elimination
    Matrix<n, n> a = m;
    Matrix<n, n> b = Matrix<n, n>::identity();

    for (int j = 0; j < n; ++j)
    {
      int pivot = j;
      for (int i = j + 1; i < n; ++i)
        if (abs(a[i * n + j]) > abs(a[pivot * n + j]))
          pivot = i;
      if (abs(a[pivot * n + j]) < epsilon)
        return Matrix<n, n>(NaN);

      if (pivot != j)
      {
        auto tmp = a.getRow(pivot);
        a.setRow(pivot, a.getRow(j));
        a.setRow(j, tmp);

        tmp = b.getRow(pivot);
        b.setRow(pivot, a.getRow(j));
        b.setRow(j, tmp);
      }

      if (a[j *n + j] != 1.f)
      {
        float scale = a[j * n + j];
        a.setRow(j, a.getRow(j) / scale);
        b.setRow(j, b.getRow(j) / scale);
      }

      for (int i = 0; i < n; ++i)
      {
        if ((i != j) && (abs(a[i * n + j]) > epsilon))
        {
          float scale = -a[i * n + j];
          a.setRow(i, a.getRow(j) * scale);
          b.setRow(i, b.getRow(j) * scale);
        }
      }
    }
    return b;
  }

  // Returns the inverse of the matrix.

  template<>
  OTK_INLINE OTK_HOSTDEVICE Matrix<2,2> Matrix<2,2>::inverse() const
  {
    float d = 1.0f / det();
    return Matrix<2,2>( { m_data[3], -m_data[1], -m_data[2], m_data[0] } ) * d;
  }

  template<>
  OTK_INLINE OTK_HOSTDEVICE Matrix<3,3> Matrix<3,3>::inverse() const
  {
      Matrix<3, 3> dst;
      const float* m = m_data;
      const float d = 1.0f / det();

      dst[0*3 + 0] = d * (m[1*3 + 1] * m[2*3 + 2] - m[2*3 + 1] * m[1*3 + 2]);
      dst[0*3 + 1] = d * (m[0*3 + 2] * m[2*3 + 1] - m[0*3 + 1] * m[2*3 + 2]);
      dst[0*3 + 2] = d * (m[0*3 + 1] * m[1*3 + 2] - m[0*3 + 2] * m[1*3 + 1]);
      dst[1*3 + 0] = d * (m[1*3 + 2] * m[2*3 + 0] - m[1*3 + 0] * m[2*3 + 2]);
      dst[1*3 + 1] = d * (m[0*3 + 0] * m[2*3 + 2] - m[0*3 + 2] * m[2*3 + 0]);
      dst[1*3 + 2] = d * (m[1*3 + 0] * m[0*3 + 2] - m[0*3 + 0] * m[1*3 + 2]);
      dst[2*3 + 0] = d * (m[1*3 + 0] * m[2*3 + 1] - m[2*3 + 0] * m[1*3 + 1]);
      dst[2*3 + 1] = d * (m[2*3 + 0] * m[0*3 + 1] - m[0*3 + 0] * m[2*3 + 1]);
      dst[2*3 + 2] = d * (m[0*3 + 0] * m[1*3 + 1] - m[1*3 + 0] * m[0*3 + 1]);
      return dst;
  }

  template<>
  OTK_INLINE OTK_HOSTDEVICE Matrix<4,4> Matrix<4,4>::inverse() const
  {
    Matrix<4,4> dst;
    const float* m   = m_data;
    const float d = 1.0f / det();

    dst[0]  = d * (m[5] * (m[10] * m[15] - m[14] * m[11]) + m[9] * (m[14] * m[7] - m[6] * m[15]) + m[13] * (m[6] * m[11] - m[10] * m[7]));
    dst[4]  = d * (m[6] * (m[8] * m[15] - m[12] * m[11]) + m[10] * (m[12] * m[7] - m[4] * m[15]) + m[14] * (m[4] * m[11] - m[8] * m[7]));
    dst[8]  = d * (m[7] * (m[8] * m[13] - m[12] * m[9]) + m[11] * (m[12] * m[5] - m[4] * m[13]) + m[15] * (m[4] * m[9] - m[8] * m[5]));
    dst[12] = d * (m[4] * (m[13] * m[10] - m[9] * m[14]) + m[8] * (m[5] * m[14] - m[13] * m[6]) + m[12] * (m[9] * m[6] - m[5] * m[10]));
    dst[1]  = d * (m[9] * (m[2] * m[15] - m[14] * m[3]) + m[13] * (m[10] * m[3] - m[2] * m[11]) + m[1] * (m[14] * m[11] - m[10] * m[15]));
    dst[5]  = d * (m[10] * (m[0] * m[15] - m[12] * m[3]) + m[14] * (m[8] * m[3] - m[0] * m[11]) + m[2] * (m[12] * m[11] - m[8] * m[15]));
    dst[9]  = d * (m[11] * (m[0] * m[13] - m[12] * m[1]) + m[15] * (m[8] * m[1] - m[0] * m[9]) + m[3] * (m[12] * m[9] - m[8] * m[13]));
    dst[13] = d * (m[8] * (m[13] * m[2] - m[1] * m[14]) + m[12] * (m[1] * m[10] - m[9] * m[2]) + m[0] * (m[9] * m[14] - m[13] * m[10]));
    dst[2]  = d * (m[13] * (m[2] * m[7] - m[6] * m[3]) + m[1] * (m[6] * m[15] - m[14] * m[7]) + m[5] * (m[14] * m[3] - m[2] * m[15]));
    dst[6]  = d * (m[14] * (m[0] * m[7] - m[4] * m[3]) + m[2] * (m[4] * m[15] - m[12] * m[7]) + m[6] * (m[12] * m[3] - m[0] * m[15]));
    dst[10] = d * (m[15] * (m[0] * m[5] - m[4] * m[1]) + m[3] * (m[4] * m[13] - m[12] * m[5]) + m[7] * (m[12] * m[1] - m[0] * m[13]));
    dst[14] = d * (m[12] * (m[5] * m[2] - m[1] * m[6]) + m[0] * (m[13] * m[6] - m[5] * m[14]) + m[4] * (m[1] * m[14] - m[13] * m[2]));
    dst[3]  = d * (m[1] * (m[10] * m[7] - m[6] * m[11]) + m[5] * (m[2] * m[11] - m[10] * m[3]) + m[9] * (m[6] * m[3] - m[2] * m[7]));
    dst[7]  = d * (m[2] * (m[8] * m[7] - m[4] * m[11]) + m[6] * (m[0] * m[11] - m[8] * m[3]) + m[10] * (m[4] * m[3] - m[0] * m[7]));
    dst[11] = d * (m[3] * (m[8] * m[5] - m[4] * m[9]) + m[7] * (m[0] * m[9] - m[8] * m[1]) + m[11] * (m[4] * m[1] - m[0] * m[5]));
    dst[15] = d * (m[0] * (m[5] * m[10] - m[9] * m[6]) + m[4] * (m[9] * m[2] - m[1] * m[10]) + m[8] * (m[1] * m[6] - m[5] * m[2]));
    return dst;
  }

  // Returns a rotation matrix.
  // This is a static member.
  template<> template<>
  OTK_INLINE OTK_HOSTDEVICE Matrix<4,4> Matrix<4,4>::rotate<Matrix<4,4>>(const float radians, const float3& axis)
  {
    Matrix<4,4> Mat = Matrix<4,4>::identity();
    float *m = Mat.getData();

    // NOTE: Element 0,1 is wrong in Foley and Van Dam, Pg 227!
    float sintheta=sinf(radians);
    float costheta=cosf(radians);
    float ux=axis.x;
    float uy=axis.y;
    float uz=axis.z;
    m[0*4+0]=ux*ux+costheta*(1-ux*ux);
    m[0*4+1]=ux*uy*(1-costheta)-uz*sintheta;
    m[0*4+2]=uz*ux*(1-costheta)+uy*sintheta;
    m[0*4+3]=0;

    m[1*4+0]=ux*uy*(1-costheta)+uz*sintheta;
    m[1*4+1]=uy*uy+costheta*(1-uy*uy);
    m[1*4+2]=uy*uz*(1-costheta)-ux*sintheta;
    m[1*4+3]=0;

    m[2*4+0]=uz*ux*(1-costheta)-uy*sintheta;
    m[2*4+1]=uy*uz*(1-costheta)+ux*sintheta;
    m[2*4+2]=uz*uz+costheta*(1-uz*uz);
    m[2*4+3]=0;

    m[3*4+0]=0;
    m[3*4+1]=0;
    m[3*4+2]=0;
    m[3*4+3]=1;

    return Matrix<4,4>( m );
  }

  // Returns a translation matrix.
  // This is a static member.
  template<> template<>
  OTK_INLINE OTK_HOSTDEVICE Matrix<4,4> Matrix<4,4>::translate<Matrix<4,4>>(const float3& vec)
  {
    Matrix<4,4> Mat = Matrix<4,4>::identity();
    float *m = Mat.getData();

    m[3] = vec.x;
    m[7] = vec.y;
    m[11]= vec.z;

    return Matrix<4,4>( m );
  }

  // Returns a scale matrix.
  // This is a static member.
  template<> template<>
  OTK_INLINE OTK_HOSTDEVICE Matrix<4,4> Matrix<4,4>::scale<Matrix<4,4>>(const float3& vec)
  {
    Matrix<4,4> Mat = Matrix<4,4>::identity();
    float *m = Mat.getData();

    m[0] = vec.x;
    m[5] = vec.y;
    m[10]= vec.z;

    return Matrix<4,4>( m );
  }


  // This is a static member.
  template<> template<>
  OTK_INLINE OTK_HOSTDEVICE Matrix<4,4>  Matrix<4,4>::fromBasis<Matrix<4,4>>( const float3& u, const float3& v, const float3& w, const float3& c )
  {
    float m[16];
    m[ 0] = u.x;
    m[ 1] = v.x;
    m[ 2] = w.x;
    m[ 3] = c.x;

    m[ 4] = u.y;
    m[ 5] = v.y;
    m[ 6] = w.y;
    m[ 7] = c.y;

    m[ 8] = u.z;
    m[ 9] = v.z;
    m[10] = w.z;
    m[11] = c.z;

    m[12] = 0.0f;
    m[13] = 0.0f;
    m[14] = 0.0f;
    m[15] = 1.0f;

    return Matrix<4,4>( m );
  }

  template<> template<>
  OTK_INLINE OTK_HOSTDEVICE Matrix<3,4> Matrix<3,4>::affineIdentity<Matrix<3,4>>()
  {
      return Matrix<3,4>::diagonal(1.0f);
  }
  
  // Returns a matrix with 'v' diagonal values and 0.0f elsewhere.
  // This is a static member.
  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE Matrix<M,N> Matrix<M,N>::diagonal(float v)
  {
    Matrix<M,N> result(0.0f);
    constexpr unsigned int minMN = M < N ? M : N;
    for( unsigned int i = 0; i < minMN; ++i )
      result.get( i,i ) = v;
    return result;
  }

  // Returns the identity matrix (square matrices only).
  // This is a static member.
  template<unsigned int M, unsigned int N> template<class T>
  OTK_INLINE OTK_HOSTDEVICE if_same_t<T, Matrix<N, N>> Matrix<M,N>::identity()
  {
    static_assert(N == M, "non-square matrices don't have an identity. use ::diagonal(1.0f)");
    return Matrix<M,N>::diagonal(1.0f);
  }

  // Ordered comparison operator so that the matrix can be used in an STL container.
  template<unsigned int M, unsigned int N>
  OTK_INLINE OTK_HOSTDEVICE bool Matrix<M,N>::operator<( const Matrix<M, N>& rhs ) const
  {
    for( unsigned int i = 0; i < N*M; ++i ) {
      if( m_data[i] < rhs[i] )
        return true;
      else if( m_data[i] > rhs[i] )
        return false;
    }
    return false;
  }

  typedef Matrix<2, 2> Matrix2x2;
  typedef Matrix<2, 3> Matrix2x3;
  typedef Matrix<2, 4> Matrix2x4;
  typedef Matrix<3, 2> Matrix3x2;
  typedef Matrix<3, 3> Matrix3x3;
  typedef Matrix<3, 4> Matrix3x4;
  typedef Matrix<4, 2> Matrix4x2;
  typedef Matrix<4, 3> Matrix4x3;
  typedef Matrix<4, 4> Matrix4x4;


  OTK_INLINE OTK_HOSTDEVICE Matrix<3,3> make_matrix3x3(const Matrix<4,4> &matrix)
  {
    Matrix<3,3> Mat;
    float *m = Mat.getData();
    const float *m4x4 = matrix.getData();

    m[0*3+0]=m4x4[0*4+0];
    m[0*3+1]=m4x4[0*4+1];
    m[0*3+2]=m4x4[0*4+2];

    m[1*3+0]=m4x4[1*4+0];
    m[1*3+1]=m4x4[1*4+1];
    m[1*3+2]=m4x4[1*4+2];

    m[2*3+0]=m4x4[2*4+0];
    m[2*3+1]=m4x4[2*4+1];
    m[2*3+2]=m4x4[2*4+2];

    return Mat;
  }

  OTK_INLINE OTK_HOSTDEVICE Matrix<3,4> make_matrix3x4(const Matrix<4,4> &matrix)
  {
    Matrix<3,4> M = otk::Matrix3x4::affineIdentity();
    M.setRow(0, matrix.getRow(0));
    M.setRow(1, matrix.getRow(1));
    M.setRow(2, matrix.getRow(2));
    return M;
  }

  OTK_INLINE OTK_HOSTDEVICE Matrix<4, 4> make_matrix4x4( const Matrix<3, 4>& matrix )
  {
      Matrix<4, 4> M = otk::Matrix4x4::identity();
      M.setRow( 0, matrix.getRow( 0 ) );
      M.setRow( 1, matrix.getRow( 1 ) );
      M.setRow( 2, matrix.getRow( 2 ) );
      M.setRow( 3, make_float4( 0.f, 0.f, 0.f, 1.f ) );
      return M;
  }

} // end namespace otk

#undef RT_MAT_DECL
