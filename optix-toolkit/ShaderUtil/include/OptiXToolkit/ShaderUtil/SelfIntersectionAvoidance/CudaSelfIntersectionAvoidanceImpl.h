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

/// \file CudaSelfIntersectionAvoidanceImpl.h
/// Cuda implementation of Self Intersection Avoidance library.
///

#include "SelfIntersectionAvoidanceImpl.h"

namespace SelfIntersectionAvoidance {

// Generic transform
struct Transform
{
    OptixTransformType type;

    union
    {
#ifndef OTK_SIA_DISABLE_TRANSFORM_TRAVERSABLES
        const OptixMatrixMotionTransform* matrixMotionTransform;
        const OptixSRTMotionTransform*    srtMotionTransform;
        const OptixStaticTransform*       staticTransform;
#endif
        struct
        {
            const float4* __restrict o2w;
            const float4* __restrict w2o;
        } instanceTransform;
    };

    OTK_INLINE __device__ OptixTransformType getTransformTypeFromHandle() const { return type; }

#ifndef OTK_SIA_DISABLE_TRANSFORM_TRAVERSABLES
    OTK_INLINE __device__ const OptixMatrixMotionTransform* getMatrixMotionTransformFromHandle() const
    {
        return matrixMotionTransform;
    }

    OTK_INLINE __device__ const OptixSRTMotionTransform* getSRTMotionTransformFromHandle() const
    {
        return srtMotionTransform;
    }

    OTK_INLINE __device__ const OptixStaticTransform* getStaticTransformFromHandle() const { return staticTransform; }
#endif

    OTK_INLINE __device__ const float4* getInstanceTransformFromHandle() const { return instanceTransform.o2w; }

    OTK_INLINE __device__ const float4* getInstanceInverseTransformFromHandle() const { return instanceTransform.w2o; }
};

// List of generic transforms
class TransformList
{
  public:
    typedef Transform value_t;

    OTK_INLINE __device__ TransformList( int size, const Transform* transforms )
        : m_size( size )
        , m_transforms( transforms ){};

    OTK_INLINE __device__ unsigned int getTransformListSize() const { return m_size; }

    OTK_INLINE __device__ Transform getTransform( unsigned int index ) const { return m_transforms[index]; }

  private:
    unsigned int m_size;
    const Transform* __restrict m_transforms;
};

// Specialized instance transform
class InstanceTransform
{
  public:
    OTK_INLINE __device__ InstanceTransform( const Matrix3x4* o2w, const Matrix3x4* w2o )
        : m_o2w( o2w )
        , m_w2o( w2o ){};

    OTK_INLINE __device__ OptixTransformType getTransformTypeFromHandle() const
    {
        return OPTIX_TRANSFORM_TYPE_INSTANCE;
    }

#ifndef OTK_SIA_DISABLE_TRANSFORM_TRAVERSABLES
    OTK_INLINE __device__ const OptixMatrixMotionTransform* getMatrixMotionTransformFromHandle() const { return 0; }

    OTK_INLINE __device__ const OptixSRTMotionTransform* getSRTMotionTransformFromHandle() const { return 0; }

    OTK_INLINE __device__ const OptixStaticTransform* getStaticTransformFromHandle() const { return 0; }
#endif

    OTK_INLINE __device__ const float4* getInstanceTransformFromHandle() const { return &m_o2w->row0; }

    OTK_INLINE __device__ const float4* getInstanceInverseTransformFromHandle() const { return &m_w2o->row0; }

  private:
    const Matrix3x4* __restrict m_o2w;
    const Matrix3x4* __restrict m_w2o;
};

// Specialized instance transform list
class InstanceTransformList
{
  public:
    typedef InstanceTransform value_t;

    OTK_INLINE __device__ InstanceTransformList( int size, const Matrix3x4* o2w, const Matrix3x4* w2o )
        : m_size( size )
        , m_o2w( o2w )
        , m_w2o( w2o ){};

    OTK_INLINE __device__ unsigned int getTransformListSize() const { return m_size; }

    OTK_INLINE __device__ InstanceTransform getTransform( unsigned int index ) const
    {
        return InstanceTransform( m_o2w + index, m_w2o + index );
    }

  private:
    unsigned int m_size;
    const Matrix3x4* __restrict m_o2w;
    const Matrix3x4* __restrict m_w2o;
};

OTK_INLINE __device__ void transformSafeSpawnOffset( float3&            outPosition,
                                                     float3&            outNormal,
                                                     float&             outOffset,
                                                     const float3&      obj_p,
                                                     const float3&      obj_n,
                                                     const float        obj_offset,
                                                     const unsigned int numTransforms,
                                                     const Matrix3x4* const __restrict o2w,
                                                     const Matrix3x4* const __restrict w2o )
{
    safeInstancedSpawnOffsetImpl<InstanceTransformList>( outPosition, outNormal, outOffset, obj_p, obj_n, obj_offset,
                                                         0.f, InstanceTransformList( numTransforms, o2w, w2o ) );
}

OTK_INLINE __device__ void transformSafeSpawnOffset( float3&            outPosition,
                                                     float3&            outNormal,
                                                     float&             outOffset,
                                                     const float3&      obj_p,
                                                     const float3&      obj_n,
                                                     const float        obj_offset,
                                                     const float        time,
                                                     const unsigned int numTransforms,
                                                     const Transform* const __restrict transforms )
{
    safeInstancedSpawnOffsetImpl<TransformList>( outPosition, outNormal, outOffset, obj_p, obj_n, obj_offset, time,
                                                 TransformList( numTransforms, transforms ) );
}

OTK_INLINE __device__ void transformSafeSpawnOffsetScaledOrthogonal( float3&       outPosition,
                                                                     float3&       outNormal,
                                                                     float&        outOffset,
                                                                     const float3& inPosition,
                                                                     const float3& inNormal,
                                                                     const float   inOffset,
                                                                     const Matrix3x4* const __restrict o2w )
{
    transformSafeSpawnOffsetScaledOrthogonal( outPosition, outNormal, outOffset, inPosition, inNormal, inOffset, &o2w->row0 );
}

}  // namespace SelfIntersectionAvoidance
