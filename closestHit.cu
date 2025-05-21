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

#include "GBuffer.cuh"
#include "shadingTypes.h"
#include "utils.cuh"

#include <OptiXToolkit/ShaderUtil/color.h>
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/Matrix.h>
#include <OptiXToolkit/ShaderUtil/SelfIntersectionAvoidance.h>

#include <optix.h>
#include <optix_device.h>

#include <cluster_builder/cluster.h>
#include <material/materialCuda.h>
#include <scene/sceneTypes.h>

#include <cfloat>

extern "C" {
   extern __constant__ Params params;
}

__device__ inline
uint32_t murmurAdd( uint32_t hash, uint32_t element )
{
    element *= 0xcc9e2d51;
    element = ( element << 15 ) | ( element >> ( 32 - 15 ) );
    element *= 0x1b873593;
    hash ^= element;
    hash = ( hash << 13 ) | ( hash >> ( 32 - 13 ) );
    hash = hash * 5 + 0xe6546b64;
    return hash;
}
__device__ inline
uint32_t murmurMix( uint32_t hash )
{
    hash ^= hash >> 16;
    hash *= 0x85ebca6b;
    hash ^= hash >> 13;
    hash *= 0xc2b2ae35;
    hash ^= hash >> 16;
    return hash;
}
__device__ inline
float3 intToColor( uint32_t index )
{
    // 1+ ... skip first entry which is black
    uint32_t hash = murmurMix( 1 + index );  
    float3 color = {
        float( ( hash >> 0 ) & 255 ) / 255.f,
        float( ( hash >> 8 ) & 255 ) / 255.f,
        float( ( hash >> 16 ) & 255 ) / 255.f
    };
    return color;
}

__device__ inline 
ushort2 clusterGetEdgeSize( uint32_t clusterId )
{
    const uchar2 cluster_size = params.cluster_shading_data[clusterId].m_cluster_size;
    return { cluster_size.x, cluster_size.y };
}


// return U-major linear indices for the 3 vertex locations of a triangle 
// within the cluster.
__device__ inline
uint3 clusterGetVertexIndices( uint32_t primId )
{
    const uint32_t      clusterId   = optixGetClusterId();
    const uint8_t       triID = (uint8_t)primId;

    // vertex quad ordering: 
    // 23
    // 01
    // triangle ordering: left edge first -- 032+013 (diagonal:03) or 012+132 (diagonal:12)
    // 21 .5    or   2. 54
    // 0. 34    or   01 .3
    // vx,vy are row-major vertex indices in range [0..sx][0..sy] sx,sy are cluster edge size
    // if vx,vy are the lower left corner vtx idxs, then diagonal:03 == ((vx & 1) == (vy & 1))

    ushort2 clusterEdgeSize = clusterGetEdgeSize( clusterId );

    const uint8_t qs     = clusterEdgeSize.x;      // quad stride
    const uint8_t vs     = clusterEdgeSize.x + 1;  // vert stride
    const uint8_t qid    = triID >> 1;             // quad id
    const uint8_t qx     = qid % qs;               // quad x
    const uint8_t qy     = qid / qs;               // quad y
    const uint8_t vid    = qy * vs + qx;           // lower-left vertex id
    const bool    diag03 = ((qx&1)==(qy&1));       // is diag 0-3 (true) or 1-2 (false)

    const uint8_t df = uint8_t(diag03) << 1 | uint8_t(triID & 1);

    uint3 indices;
    switch(df) {
        case 0b00: indices = make_uint3(vid,   vid+1,     vid+vs  ); break;
        case 0b01: indices = make_uint3(vid+1, vid+1+vs,  vid+vs  ); break;
        case 0b10: indices = make_uint3(vid,   vid+1+vs,  vid+vs  ); break;
        case 0b11: indices = make_uint3(vid,   vid+1,     vid+1+vs); break;
    }

    return indices;
}


struct IntersectionRecord
{
    float3 p{ 0.f };             // world space intersection point
    float3 n{ 0.f };             // world space shading normal
    float3 gn{ 0.f };            // world space geometry normal
    float2 texcoord{ 0.f };      // user-assigned texcoord from base mesh
    float3 barycentrics{ 0.f };  
    float wireframeMask { 1.0f };
    float  hitT{ std::numeric_limits<float>::infinity() };

    // Needed for AOVs
    uint32_t surfaceIndex = ~uint32_t( 0 );
    float2   surfaceUV    = { 0.f, 0.f };
};


__device__ inline 
float3 interpolateVertexNormals( const unsigned int primId, const float3 barycentrics )
{
    const uint32_t clusterId       = optixGetClusterId();
    const uint3    localVtxIndices = clusterGetVertexIndices( primId );
    const uint3    globalVtxIdx    = localVtxIndices + make_uint3( params.cluster_shading_data[clusterId].m_cluster_vertex_offset );

    const float3 n0 = unpackNormalizedVector( params.packedClusterVertexNormals[globalVtxIdx.x] );
    const float3 n1 = unpackNormalizedVector( params.packedClusterVertexNormals[globalVtxIdx.y] );
    const float3 n2 = unpackNormalizedVector( params.packedClusterVertexNormals[globalVtxIdx.z] );

    return normalize( barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2 );
}

__device__ inline float3 makeShadingNormal( unsigned int primId, float3 barycentrics, float3 geomNormal, float3 rayDirection )
{
    float3 shadingNormal = interpolateVertexNormals( primId, barycentrics );
    shadingNormal        = normalize( optixTransformNormalFromObjectToWorldSpace( shadingNormal ) );
    shadingNormal        = faceforward( shadingNormal, geomNormal, shadingNormal );
    return shadingNormal;
}



// Compute 2D vertex index from linear index.
__device__ ushort2 index_2d( uint32_t index_linear, uint16_t line_stride )
{
    return { static_cast<uint16_t>(index_linear % line_stride), static_cast<uint16_t>(index_linear / line_stride) };
}


__device__ inline ushort2 operator+(const ushort2 a, const ushort2 b)
{
    return {static_cast<uint16_t>(a.x + b.x), static_cast<uint16_t>(a.y + b.y)};
}


// Given a cluster triangle id, find the uv coordinates in the parametric surface
// that generated the triangle's three corners.
//
__device__ inline void getSurfaceUV( unsigned int primID, float2* uvs )
{
    const uint3    uMajorVtxIDs         = clusterGetVertexIndices( primID );
    const uint32_t clusterId            = optixGetClusterId();
    const auto     cluster_shading_data = params.cluster_shading_data[clusterId];

    const uchar2  cluster_size   = cluster_shading_data.m_cluster_size;
    const ushort2 cluster_offset = cluster_shading_data.m_cluster_offset;
    const ushort4 edgeSegments   = cluster_shading_data.m_surface_edge_segments;

    const GridSampler sampler{ edgeSegments };

    // offset local i,j index to surface index
    auto vertexIndex2d = index_2d( uMajorVtxIDs.x, cluster_size.x + 1 ) + cluster_offset;
    if( params.cluster_pattern == ClusterPattern::SLANTED )
    {
        uvs[0] = sampler.uv<ClusterPattern::SLANTED>( vertexIndex2d );
        vertexIndex2d = index_2d( uMajorVtxIDs.y, cluster_size.x + 1 ) + cluster_offset;
        uvs[1] = sampler.uv<ClusterPattern::SLANTED>( vertexIndex2d );
        vertexIndex2d = index_2d( uMajorVtxIDs.z, cluster_size.x + 1 ) + cluster_offset;
        uvs[2] = sampler.uv<ClusterPattern::SLANTED>( vertexIndex2d );
    }
    else
    {
        uvs[0] = sampler.uv<ClusterPattern::REGULAR>( vertexIndex2d );
        vertexIndex2d = index_2d( uMajorVtxIDs.y, cluster_size.x + 1 ) + cluster_offset;
        uvs[1] = sampler.uv<ClusterPattern::REGULAR>( vertexIndex2d );
        vertexIndex2d = index_2d( uMajorVtxIDs.z, cluster_size.x + 1 ) + cluster_offset;
        uvs[2] = sampler.uv<ClusterPattern::REGULAR>( vertexIndex2d );
    }
}


// color function for visualizing different attributes of clusters
__device__ inline float3 getClusterColor( unsigned primID )
{
    float3 color = make_float3( .8f );

    const uint32_t clusterId = optixGetClusterId();
    const auto cluster_shading_data = params.cluster_shading_data[clusterId];
    const uchar2 cluster_size = cluster_shading_data.m_cluster_size;
    const ushort2 cluster_offset = cluster_shading_data.m_cluster_offset;
    const uint32_t linear_cluster_offset = (cluster_offset.y * cluster_size.x) + cluster_offset.x;

    if( params.bound.colorMode == ColorMode::COLOR_BY_TRIANGLE )
    {
        uint32_t index = 0;
        index          = murmurAdd( index, cluster_shading_data.m_surface_id );
        index          = murmurAdd( index, linear_cluster_offset );
        index          = murmurAdd( index, primID );
        color = intToColor( index );
    }
    else if( params.bound.colorMode == ColorMode::COLOR_BY_MATERIAL )
    {
        const unsigned int materialID = optixGetSbtGASIndex();
        color = intToColor( materialID );
    }
    else if( params.bound.colorMode == ColorMode::COLOR_BY_CLUSTER_ID )
    {
        uint32_t index = 0;
        index          = murmurAdd( index, cluster_shading_data.m_surface_id );
        index          = murmurAdd( index, linear_cluster_offset );
        color = intToColor( index );
    }
    else if( params.bound.colorMode == ColorMode::COLOR_BY_CLUSTER_UV )
    {
        float2 clusterUVs[3];
        getSurfaceUV( primID, clusterUVs );
        const float2 bary = optixGetTriangleBarycentrics();
        float2 hitpointUV = ( 1.0f - bary.x - bary.y ) * clusterUVs[0] + bary.x * clusterUVs[1] + bary.y * clusterUVs[2];
        color             = make_float3( 1.0f, 0.f, 0.f ) * hitpointUV.x + make_float3( 0.0f, 1.f, 0.f ) * hitpointUV.y;
    }
    return color;
}

template <typename T>
__device__ inline
T bilerp( const T* values, float2 uv )
{
    // bilerp from 4 corner attributes
    const float u = uv.x;
    const float v = uv.y;

    T result = values[0] * ( 1.0f - u ) * ( 1.0f - v ) + values[1] * u * ( 1.0f - v )
        + values[2] * u * v + values[3] * ( 1.0f - u ) * v;

    return result;
}

__device__ inline float3 transformPointFromObjectToScreenSpace( const float3& v )
{
    float3 vw = optixTransformPointFromObjectToWorldSpace( v );
    float4 s = params.viewProjectionMatrix * make_float4( vw.x, vw.y, vw.z, 1.f );
    return make_float3( s.x / s.w, s.y / s.w, s.z / s.w );
}

__device__ inline 
IntersectionRecord makeIntersectionRecord()
{
    IntersectionRecord ir = {0};
    ir.hitT = optixGetRayTmax();
    const unsigned int primIdx = optixGetPrimitiveIndex();

    const float2 barycentrics = optixGetTriangleBarycentrics();
    ir.barycentrics           = make_float3( 1.f - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y );

    float3 v[3];
    optixGetTriangleVertexData( v );

    if( params.bound.enableWireframe || params.bound.enableSurfaceWireframe )
    {
        // Build a wireframe mask that is roughly constant width in screen space

        float3 v0 = transformPointFromObjectToScreenSpace( v[0] );
        float3 v1 = transformPointFromObjectToScreenSpace( v[1] );
        float3 v2 = transformPointFromObjectToScreenSpace( v[2] );
        
        constexpr float  thickness  = 5e-4f;
        constexpr float  smoothness = 5e-5f;
        
        ir.wireframeMask = 1.f;

        if ( params.bound.enableWireframe )
        {
            // Wireframe for triangles

            // Clamp line width in barycentric space to avoid overly thick lines for far away triangles
            constexpr float maxParametricWidth = 0.1f;
            if( ir.barycentrics.x < maxParametricWidth || ir.barycentrics.y < maxParametricWidth || ir.barycentrics.z < maxParametricWidth )
            {
                // Compute distance to edges in screen space
                float  area = 0.5f * length( cross( v1-v0, v2-v0 ) );
                float3 invEdgeLengths = { 1.0f / length( v2-v1 ), 1.0f / length( v2-v0 ), 1.0f / length( v1-v0 ) };
                float3 distToEdge = 2.0f * ir.barycentrics * area * invEdgeLengths;

                const float minDist = fminf( distToEdge.x, fminf( distToEdge.y, distToEdge.z ) );
                ir.wireframeMask *= smoothstep( thickness, thickness + smoothness, minDist );
            }
        }

        if ( params.bound.enableSurfaceWireframe )
        {
            // Wireframe for parametric surface patch 
           
            float2 U[3];
            getSurfaceUV( primIdx, U );
            const float2 uv = ir.barycentrics.x * U[0] + ir.barycentrics.y * U[1] + ir.barycentrics.z * U[2];
            const float  du = fminf( uv.x, 1 - uv.x );
            const float  dv = fminf( uv.y, 1 - uv.y );

            // Build the 2x2 matrices Ep = [P1-P0, P2-P0] and Eu = [U1-U0, U2-U0]
            otk::Matrix2x2 Ep = otk::Matrix2x2( { v1.x - v0.x, v2.x - v0.x, v1.y - v0.y, v2.y - v0.y } );
            otk::Matrix2x2 Eu = otk::Matrix2x2( { U[1].x - U[0].x, U[2].x - U[0].x, U[1].y - U[0].y, U[2].y - U[0].y } );

            constexpr float maxParametricWidth = 0.1f;
            if( fabsf( Eu.det() ) > 1e-7f && ( du < maxParametricWidth || dv < maxParametricWidth ) )
            {
                otk::Matrix2x2 Eu_inv = Eu.inverse();

                // Solve for J = Ep * Eu^(-1) = [dPdu, dPdv]
                otk::Matrix2x2 J = Ep * Eu_inv;

                float2 dPdu = J.getCol( 0 );
                float2 dPdv = J.getCol( 1 );

                // Compute distance from uv to closest edge in screen space
                float2 distToEdge = { length( dPdu * du ), length( dPdv * dv ) };
                float  minDist    = fminf( distToEdge.x, distToEdge.y );

                ir.wireframeMask *= smoothstep( 1.5f*thickness, 1.5f*thickness + smoothness, minDist );
            }
        }
    }

    float worldOffset = 0.f;
    {
        float3 objPos    = { 0.f };
        float3 objNorm   = { 0.f };
        float  objOffset = 0.f;

        SelfIntersectionAvoidance::getSafeTriangleSpawnOffset( objPos, objNorm, objOffset, v[0], v[1], v[2],
                                                               optixGetTriangleBarycentrics() );

        // convert object space spawn point and offset to world space
        SelfIntersectionAvoidance::transformSafeSpawnOffset( ir.p, ir.gn, worldOffset, objPos, objNorm, objOffset );
    }

    float3 front = { 0.f };
    float3 back  = { 0.f };

    // offset world space spawn point to generate self intersection safe front and back spawn points
    SelfIntersectionAvoidance::offsetSpawnPoint( front, back, ir.p, ir.gn, worldOffset );

    // flip normal to point towards incoming direction
    if( dot( ir.gn, optixGetWorldRayDirection() ) > 0.f )
    {
        ir.gn = -ir.gn;
        std::swap( front, back );
    }
    ir.p = front;

    ir.n = params.packedClusterVertexNormals ?
               makeShadingNormal( primIdx, ir.barycentrics, ir.gn, optixGetWorldRayDirection() ) :
               ir.gn;

    // Bilinear texcoords
    float2 uvs[3];
    getSurfaceUV( primIdx, uvs );
    float2 uv = ir.barycentrics.x * uvs[0] + ir.barycentrics.y * uvs[1] + ir.barycentrics.z * uvs[2];
    ir.surfaceUV = uv;

    const uint32_t clusterId = optixGetClusterId();
    ir.surfaceIndex = params.cluster_shading_data[clusterId].m_surface_id;

    const float2* texcoords = params.cluster_shading_data[clusterId].m_surface_texcoords;
    ir.texcoord = bilerp( texcoords, uv );
    
    return ir;

}

__device__ inline 
float3 getMaterialBaseColor( const IntersectionRecord ir, unsigned& seed )
{ 
    float3 baseColor = make_float3( 0.8f );

    const OptixTraversableHandle gas     = optixGetGASTraversableHandle();
    const unsigned int           primIdx = optixGetPrimitiveIndex();
    
    {
        if( params.bound.colorMode == ColorMode::COLOR_BY_NORMAL )
        {
            baseColor = 0.5f * ( make_float3(1.0f) + ir.n );
        }

        else if( params.bound.colorMode == ColorMode::COLOR_BY_TEXCOORD )
        {
            baseColor = make_float3( ir.texcoord.x, ir.texcoord.y, 0.0f );
        }
        else if( params.bound.colorMode == ColorMode::COLOR_BY_MICROTRI_AREA )
        {
            float3 v[3];
            optixGetTriangleVertexData( v );
            
            // transform vertices to world space
            for( int i = 0; i < 3; ++i )
            {
                v[i] = optixTransformPointFromObjectToWorldSpace( v[i] );
            }

            // project

            const float4 sv0 = params.tessViewProjectionMatrix * make_float4( v[0].x, v[0].y, v[0].z, 1.f );
            const float4 sv1 = params.tessViewProjectionMatrix * make_float4( v[1].x, v[1].y, v[1].z, 1.f );
            const float4 sv2 = params.tessViewProjectionMatrix * make_float4( v[2].x, v[2].y, v[2].z, 1.f );

            const float3 spv0 = make_float3( sv0.x / sv0.w, sv0.y / sv0.w, 0.f );
            const float3 spv1 = make_float3( sv1.x / sv1.w, sv1.y / sv1.w, 0.f );
            const float3 spv2 = make_float3( sv2.x / sv2.w, sv2.y / sv2.w, 0.f );

            const uint2 dims = make_uint2( optixGetLaunchDimensions() );
            const float uTriScreenArea = .5f * length( cross( spv0 - spv1, spv2 - spv0 ) );
            const float uTriAreaInPixels = dims.x * dims.y * uTriScreenArea / 4.f;  // the area of the screen is 4.f since the screen vertices range from [-1, 1]

            const float normUTriAreaInPixels = remap( clamp( uTriAreaInPixels, 0.f, 2.f ), 0.f, 2.f, 1.0f, 0.0f );
            baseColor = temperature( normUTriAreaInPixels );
        }
        else if (params.bound.colorMode == ColorMode::BASE_COLOR)
        {
            // Get material albedo
            HitGroupData const *data = (HitGroupData const *)optixGetSbtDataPointer();
            if (data && data->material)
            {
                MaterialCuda const *m = data->material;
                baseColor = m->albedo;
                if (m->albedoMap)
                {
                    float4 c = tex2D<float4>(m->albedoMap->getTexref(), ir.texcoord.x, ir.texcoord.y);
                    baseColor = lerp(baseColor, make_float3(c.x, c.y, c.z), c.w);
                }
            }
        }
        else
        {
            baseColor = getClusterColor( primIdx );
        }
    }
    return baseColor;
}

__device__ inline 
float occlusion( OptixTraversableHandle handle, float3 rayOrigin, float3 rayDirection )
{
    
    optixTraverse( handle, rayOrigin, rayDirection,
                0.0f,     // tmin
                FLT_MAX,  // tmax
                0.0f,     // time
                OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0,        // SBT offset
                0,        // SBT stride
                RAY_TYPE_OCCLUSION );

    float occluded = optixHitObjectIsMiss() ? 0.0f : 1.0f;
    return occluded;

}


__device__ inline 
float3 aoSample( const float3 normal, const float2 u )
{
    const Onb onb{ normal };
    float3    dir = cosineSampleHemisphere( u );
    dir = onb.toWorld( dir );
    return normalize(dir);
}

extern "C"
__global__ void __closesthit__radiance()
{
    const float hitT = optixGetRayTmax();
    
    const IntersectionRecord ir = makeIntersectionRecord();
    float3 pathWeight = make_float3(1.f); 

    const uint2 idx = make_uint2( optixGetLaunchIndex() );

    {
        unsigned seed = optixGetPayload_3();
        const float3 baseColor = getMaterialBaseColor( ir, seed );

        // Super-sample hemisphere AO
        const int strataCount = (int)sqrtf( params.aoSamples );
        float occl  = 0.f;
        for( int k = 0; k < params.aoSamples; ++k )
        {
            const float2 rs = randomStrat( k, strataCount, seed );
            const float3 L        = aoSample( ir.n, rs );
            occl += occlusion( params.handle, ir.p, L );
        }
        occl /= params.aoSamples;
        pathWeight *= baseColor * ( 1.0f - occl ) * ir.wireframeMask;

        optixSetPayload_0( __float_as_uint( pathWeight.x ) );
        optixSetPayload_1( __float_as_uint( pathWeight.y ) );
        optixSetPayload_2( __float_as_uint( pathWeight.z ) );
        optixSetPayload_3( seed );

        // write AOVs for motion vectors, post effects, etc
        {
            const float depth = dot( normalize( params.W ), ir.p - params.eye );
            gbuffer::write( depth, params.aovDepth, idx );

            gbuffer::write( make_float4( ir.n, 1.f ), params.aovNormals, idx );
            gbuffer::write( make_float4( baseColor, 1.f ), params.aovAlbedo, idx );

            HitResult hit = {
                .instanceId   = optixGetInstanceId(),
                .surfaceIndex = ir.surfaceIndex,
                .u            = ir.surfaceUV.x,
                .v            = ir.surfaceUV.y,
                .texcoord     = ir.texcoord,
            };
            const uint2  dims            = make_uint2( optixGetLaunchDimensions() );
            unsigned int linearIdx       = dims.x * idx.y + idx.x;
            params.hit_buffer[linearIdx] = hit;
        }
    }
}

