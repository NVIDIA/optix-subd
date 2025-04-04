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

#include "animation.h"

#include <OptiXToolkit/ShaderUtil/Quaternion.h>

#include <json/json.h>

#include <algorithm>
#include <cassert>
#include <string>

using quat = otk::quat;

namespace anim {

float lerp( const float& a, const float& b, const float t )
{
    return a + t * (b - a);
}

template <typename T> inline uint8_t dim() { 
    static_assert("invalid dimension object");
    return 0; 
}
template <> inline uint8_t dim<float>() { return 1; }
template <> inline uint8_t dim<float2>() { return 2; }
template <> inline uint8_t dim<float3>() { return 3; }
template <> inline uint8_t dim<float4>() { return 4; }
template <> inline uint8_t dim<quat>() { return 4; }

// linear

template <typename T>
inline T interpolateLinear( const Keyframe<T>& a, const Keyframe<T>& b, const float t )
{
    return lerp( a.value, b.value, t );
}
template <>
inline quat interpolateLinear( const KeyframeQ& a, const KeyframeQ& b, const float t )
{
    // quaternion lerp needs to be normalized at least (should really use spherical-lerp)
    return nlerp( a.value, b.value, t );
}

// quaternion SLerp

template <typename T> 
inline T interpolateSLerp( const Keyframe<T>& a, const Keyframe<T>& b, const float t )
{
    static_assert("SLerp quaternion interpolation only works on homogeneous float4 type.");
    return T{ 0 };
}
template <> float4 interpolateSLerp<float4>( const Keyframe<float4>& a, const Keyframe<float4>& b, const float t )
{
    quat qa = { .w = a.value.w, .x = a.value.x, .y = a.value.y, .z = a.value.z, };
    quat qb = { .w = b.value.w, .x = b.value.x, .y = b.value.y, .z = b.value.z, };
    quat qr = otk::slerp(qa, qb, t);
    return float4(qr.x, qr.y, qr.z, qr.w);
}
template <> quat interpolateSLerp<quat>(const Keyframe<quat>& a, const Keyframe<quat>& b, const float t)
{
    return otk::slerp(a.value, b.value, t);
}

// Catmull Rom

template <typename T> 
inline T interpolateCatmullRom( const Keyframe<T>& a, const Keyframe<T>& b, const Keyframe<T>& c, const Keyframe<T>& d, float t )
{
    // https://en.wikipedia.org/wiki/Cubic_Hermite_spline#Interpolation_on_the_unit_interval_with_matched_derivatives_at_endpoints
    // a = p[n-1], b = p[n], c = p[n+1], d = p[n+2]
    T i = -a.value + 3.f * b.value - 3.f * c.value + d.value;
    T j = 2.f * a.value - 5.f * b.value + 4.f * c.value - d.value;
    T k = -a.value + c.value;
    return 0.5f * ((i * t + j) * t + k) * t + b.value;
}

// Hermite

template <typename T> 
inline T interpolateHermite( const Keyframe<T>& a, const Keyframe<T>& b, const Keyframe<T>& c, const Keyframe<T>& d, float t, float dt )
{
    // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-c-spline-interpolation
    const float t2 = t * t;
    const float t3 = t2 * t;
    return (2.f * t3 - 3.f * t2 + 1.f) * b.value
         + (t3 - 2.f * t2 + t) * b.outTangent * dt
         + (-2.f * t3 + 3.f * t2) * c.value
         + (t3 - t2) * c.inTangent * dt;
}


template <typename T> 
T interpolate( Basis interpolation,
    const Keyframe<T>& a, const Keyframe<T>& b, const Keyframe<T>& c, const Keyframe<T>& d, float t, float dt ) {

    switch ( interpolation )
    {
        using enum Basis;
        case Step: return b.value;
        case Linear: return interpolateLinear<T>( b, c, t );
        case CatmullRom: return interpolateCatmullRom<T>( a, b, c, d, t );
        case Hermite: return interpolateHermite<T>( a, b, c, d, t, dt );
        case SLerp: {
            if constexpr ( std::is_same_v<T, quat> || std::is_same_v<T, float4> )
                return interpolateSLerp<T>( b, c, t );
            else
                static_assert( "SLerp can only be applied to quaternions and float4" );
                return T{0};
            }
        default:
            return b.value;
    }
    return T{};
}

template float interpolate( Basis, const Keyframe1&, const Keyframe1&, const Keyframe1&, const Keyframe1&, float, float);
template float2 interpolate( Basis, const Keyframe2&, const Keyframe2&, const Keyframe2&, const Keyframe2&, float, float);
template float3 interpolate( Basis, const Keyframe3&, const Keyframe3&, const Keyframe3&, const Keyframe3&, float, float);
template float4 interpolate( Basis, const Keyframe4&, const Keyframe4&, const Keyframe4&, const Keyframe4&, float, float);
template quat interpolate( Basis, const KeyframeQ&, const KeyframeQ&, const KeyframeQ&, const KeyframeQ&, float, float);

//
// Keyframe
//


template<typename T> inline ValueType Keyframe<T>::valueType() {
    static_assert( "unknown value type" );
    return ValueType::Float;  // make compiler happy
}

template <> ValueType Keyframe<float>::valueType() { return ValueType::Float; }
template <> ValueType Keyframe<float2>::valueType() { return ValueType::Float2; }
template <> ValueType Keyframe<float3>::valueType() { return ValueType::Float3; }
template <> ValueType Keyframe<float4>::valueType() { return ValueType::Float4; }
template <> ValueType Keyframe<quat>::valueType() { return ValueType::Quat; }

//
// Track
//

template<typename T>
std::optional<T> Track<T>::evaluate( float time ) const
{
    if( const uint32_t count = (uint32_t)keyframes.size(); count > 0 )
    {
        if (time <= keyframes.front().time)
            return keyframes.front().value;

        if( ( count == 1 ) || ( time > keyframes.back().time ) )
        {
            if( extrapolate )
                return keyframes.back().value;
            else
                return {};
        }

        // keyframe search optimization : odds are very high that the time for the
        // current evaluation is greater than that of the previous call, so start
        // the search from the index used for the previous call instaed of index 0.
        uint32_t offsetStart = time > keyframes[offsetPrev].time ? offsetPrev : 0;
        
        for( uint32_t offset = offsetStart; offset < count; offset++ )
        {
            const float tb = keyframes[offset].time;
            const float tc = keyframes[offset + 1].time;

            assert( tb < tc );

            if( tb <= time && time < tc )
            {
                const Keyframe<T>& b = keyframes[offset];
                const Keyframe<T>& c = keyframes[offset + 1];
                const Keyframe<T>& a = (offset > 0) ? keyframes[offset - 1] : b;
                const Keyframe<T>& d = (offset < count - 2) ? keyframes[offset + 2] : c;
                const float dt = tc - tb;
                const float u = (time - tb) / dt;

                offsetPrev = offset;

                return interpolate<T>(interpolation, a, b, c, d, u, dt);
            }
        }
    }
    return {};
}

template <typename T>
inline std::optional<float> Track<T>::getStartTime() const
{
    return keyframes.empty() ? std::optional<float>{} : keyframes.front().time;
}

template<typename T>
inline std::optional<float> Track<T>::getEndTime() const
{
    return keyframes.empty() ? std::optional<float>{} : keyframes.back().time;
}

template<typename T>
inline std::optional<float> Track<T>::getDuration() const
{
    auto start = getStartTime();
    auto end = getEndTime();
    if( start && end )
        return ( *end - *start );
    return {};
}

template struct Track<float>;
template struct Track<float2>;
template struct Track<float3>;
template struct Track<float4>;
template struct Track<quat>;

//
// Channel
//

template<typename T>
Keyframe<T>& Channel<T>::operator[]( uint32_t index )
{
    return track.keyframes[index];
}
template<typename T>
const Keyframe<T>& Channel<T>::operator[]( uint32_t index ) const
{
    return track.keyframes[index];
}

template <typename T> inline ValueType _valueType() { static_assert("unknown keyframe ValueType"); return ValueType::Float; }
template <> ValueType _valueType<float>() { return ValueType::Float; }
template <> ValueType _valueType<float2>() { return ValueType::Float2; }
template <> ValueType _valueType<float3>() { return ValueType::Float3; }
template <> ValueType _valueType<float4>() { return ValueType::Float4; }
template <> ValueType _valueType<quat>() { return ValueType::Quat; }

template <typename T>
inline ValueType Channel<T>::valueType() const
{
    return _valueType<T>();
}

template <typename T>
inline void Channel<T>::resize(size_t size)
{
    track.keyframes.resize(size);
}

template <typename T> inline T _fromFloat( const float* b ) { static_assert("invalid T"); }
template <> float _fromFloat( const float* value ) { return value ? value[0] : 0.f; }
template <> float2 _fromFloat( const float* value ) { return value ? float2{ .x = value[0], .y = value[1] } : float2{}; }
template <> float3 _fromFloat( const float* value ) { return value ? float3{ .x = value[0], .y = value[1], .z = value[2] } : float3{}; }
template <> float4 _fromFloat( const float* value ) { return value ? float4{ .x = value[0], .y = value[1], .z = value[2], .w = value[3] } : float4{}; }
template <> quat _fromFloat( const float* value ) { return value ? normalize(quat{ .w = value[0], .x = value[1], .y = value[2], .z = value[3] }) : quat{}; }

template <typename T>
inline bool Channel<T>::setKeyframe( uint32_t index, const KeyframeDesc& desc )
{
    if( desc.valueSize != dim<T>() )
        return false;

    Keyframe<T>& k = track.keyframes[index];

    k.time = desc.time;
    k.value = _fromFloat<T>( desc.value );   
    if( interpolation() == Basis::Hermite )
    {
        k.inTangent = _fromFloat<T>( desc.inTangent );
        k.outTangent = _fromFloat<T>( desc.outTangent );  
    }
    return true;
}

ChannelInterface::~ChannelInterface() {}

template class Channel<float>;
template class Channel<float2>;
template class Channel<float3>;
template class Channel<float4>;
template class Channel<quat>;

} // end namespace anim
 
