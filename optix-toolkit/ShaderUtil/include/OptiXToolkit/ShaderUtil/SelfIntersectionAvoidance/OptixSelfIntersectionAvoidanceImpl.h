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

/// \file OptixSelfIntersectionAvoidanceImpl.h
/// Optix implementation of Self Intersection Avoidance library.

#include "SelfIntersectionAvoidanceImpl.h"
#include <assert.h>

namespace SelfIntersectionAvoidance {

class OptixTransform
{
  public:
    OTK_INLINE __device__ OptixTransform( OptixTraversableHandle handle )
        : m_handle( handle ){};

    OTK_INLINE __device__ OptixTransformType getTransformTypeFromHandle() const
    {
        return optixGetTransformTypeFromHandle( m_handle );
    }

    OTK_INLINE __device__ const OptixSRTMotionTransform* getSRTMotionTransformFromHandle() const
    {
        return optixGetSRTMotionTransformFromHandle( m_handle );
    }

    OTK_INLINE __device__ const OptixMatrixMotionTransform* getMatrixMotionTransformFromHandle() const
    {
        return optixGetMatrixMotionTransformFromHandle( m_handle );
    }

    OTK_INLINE __device__ const OptixStaticTransform* getStaticTransformFromHandle() const
    {
        return optixGetStaticTransformFromHandle( m_handle );
    }

    OTK_INLINE __device__ const float4* getInstanceTransformFromHandle() const
    {
        return optixGetInstanceTransformFromHandle( m_handle );
    }

    OTK_INLINE __device__ const float4* getInstanceInverseTransformFromHandle() const
    {
        return optixGetInstanceInverseTransformFromHandle( m_handle );
    }

  private:
    OptixTraversableHandle m_handle;
};

class OptixLocalTransformList
{
  public:
    typedef OptixTransform value_t;

    OTK_INLINE __device__ unsigned int getTransformListSize() const { return optixGetTransformListSize(); }

    OTK_INLINE __device__ OptixTransform getTransform( unsigned int index ) const
    {
        return OptixTransform( optixGetTransformListHandle( index ) );
    }
};

class OptixTransformList
{
  public:
    typedef OptixTransform value_t;

    OTK_INLINE __device__ OptixTransformList( int size, const OptixTraversableHandle* handles )
        : m_size( size )
        , m_handles( handles ){};

    OTK_INLINE __device__ unsigned int getTransformListSize() const { return m_size; }

    OTK_INLINE __device__ OptixTransform getTransform( unsigned int index ) const
    {
        return OptixTransform( m_handles[index] );
    }

  private:
    unsigned int m_size;
    const OptixTraversableHandle* __restrict m_handles;
};

OTK_INLINE __device__ void getSafeTriangleSpawnOffset( float3& outPosition, float3& outNormal, float& outOffset )
{
    assert( optixIsTriangleHit() );

    float3 data[3];
    optixGetTriangleVertexData( optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
                                optixGetRayTime(), data );

    getSafeTriangleSpawnOffset( outPosition, outNormal, outOffset, data[0], data[1], data[2], optixGetTriangleBarycentrics() );
}

OTK_INLINE __device__ void transformSafeSpawnOffset( float3&            outPosition,
                                                     float3&            outNormal,
                                                     float&             outOffset,
                                                     const float3&      inPosition,
                                                     const float3&      inNormal,
                                                     const float        inOffset,
                                                     const float        time,
                                                     const unsigned int numTransforms,
                                                     const OptixTraversableHandle* const __restrict transformHandles )
{
    safeInstancedSpawnOffsetImpl<OptixTransformList>( outPosition, outNormal, outOffset, inPosition, inNormal, inOffset,
                                                      time, OptixTransformList( numTransforms, transformHandles ) );
}

OTK_INLINE __device__ void transformSafeSpawnOffset( float3&       outPosition,
                                                     float3&       outNormal,
                                                     float&        outOffset,
                                                     const float3& inPosition,
                                                     const float3& inNormal,
                                                     const float   inOffset )
{
    safeInstancedSpawnOffsetImpl<OptixLocalTransformList>( outPosition, outNormal, outOffset, inPosition, inNormal,
                                                           inOffset, optixGetRayTime(), OptixLocalTransformList() );
}

OTK_INLINE __device__ void transformSafeSpawnOffsetScaledOrthogonal( float3&                      outPosition,
                                                                     float3&                      outNormal,
                                                                     float&                       outOffset,
                                                                     const float3&                inPosition,
                                                                     const float3&                inNormal,
                                                                     const float                  inOffset,
                                                                     const OptixTraversableHandle transform )
{
    assert( optixGetTransformTypeFromHandle( transform ) == OPTIX_TRANSFORM_TYPE_INSTANCE );

    transformSafeSpawnOffsetScaledOrthogonal( outPosition, outNormal, outOffset, inPosition, inNormal, inOffset,
                                              optixGetInstanceTransformFromHandle( transform ) );
}


}  // namespace SelfIntersectionAvoidance
