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

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/ShaderUtil/Quaternion.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Json 
{
    class Value;
}

namespace anim 
{
    enum class Basis : uint8_t 
    {
        Step = 0,
        Linear,
        SLerp,
        CatmullRom,
        Hermite,
    };

    bool requiresTangents( Basis basis );
    
    enum class ValueType : uint8_t 
    { 
        Float = 0, 
        Float2,
        Float3, 
        Float4,
        Quat 
    };

    static inline uint8_t dim( ValueType vtype );

    template <typename T> struct Keyframe {
        float time = 0.f;
        T value = T{ 0.f };
        T inTangent = T{ 0.f };
        T outTangent = T{ 0.f };

        static ValueType valueType();
    };

    using Keyframe1 = Keyframe<float>;
    using Keyframe2 = Keyframe<float2>;
    using Keyframe3 = Keyframe<float3>;
    using Keyframe4 = Keyframe<float4>;
    using KeyframeQ = Keyframe<otk::quat>;

    template <typename T> T interpolate(
        Basis interpolation,
        const Keyframe<T>& a,
        const Keyframe<T>& b,
        const Keyframe<T>& c,
        const Keyframe<T>& d,
        float t, 
        float dt);

    template <typename T> struct Track 
    {
        std::vector<Keyframe<T>> keyframes;
        Basis interpolation = Basis::Linear;
        bool extrapolate = true;
        
        mutable uint32_t offsetPrev = 0;

        std::optional<float> start() const { return !keyframes.empty() ? keyframes.front().time : float{}; }
        std::optional<float> end() const { return !keyframes.empty() ? keyframes.back().time : float{}; }

        // assumes keyframes are in chronological order : sort if necessary
        std::optional<T> evaluate( float time ) const;

        bool empty() const { return keyframes.empty(); }

        std::optional<float> getStartTime() const;
        std::optional<float> getEndTime() const;
        std::optional<float> getDuration() const;

        void sortKeyframes()
        {
            std::sort( keyframes.begin(), keyframes.end(), 
                []( const Keyframe<T>& a, const Keyframe<T>& b ) { return a.time < b.time; });
        }
    };

    using Track1 = Track<float>;
    using Track2 = Track<float2>;
    using Track3 = Track<float3>;
    using Track4 = Track<float4>;
    using TrackQ = Track<otk::quat>;

    // Run-time virtual interface
    //
    // Entity-component systems often cannot use compile-time instantiation,
    // so the Track functionality is wrapped into a 'Channel' interface, relying
    // on virtual inheritance to resolve keyframe value type at run-time instead.
    //

    struct ChannelInterface
    {
        virtual ~ChannelInterface() = 0;

        virtual void evaluate( float time ) = 0;
        virtual std::optional<float> start() const = 0;
        virtual std::optional<float> end() const = 0;
        inline float duration() const { return start() && end() ?  *end() - *start() : 0.f; }

        virtual ValueType valueType() const = 0;       
        virtual Basis interpolation() const = 0;
        virtual bool extrapolation() const = 0;

        virtual bool empty() const = 0;
        virtual void resize( size_t size ) = 0;

        struct KeyframeDesc {
            float time = 0.f;
            uint8_t valueSize = 0;
            float value[4] = { 0.f, 0.f, 0.f, 0.f };
            float inTangent[4] = { 0.f, 0.f, 0.f, 0.f };
            float outTangent[4] = { 0.f, 0.f, 0.f, 0.f };
        };
        virtual bool setKeyframe( uint32_t index, const KeyframeDesc& desc ) = 0;
        virtual void sortKeyframes() = 0;
        virtual void setInterpolation( Basis interpolation ) = 0;
        virtual void setExtrapolation( bool extrapolate ) = 0;

    };

    template <typename T> class Channel : public ChannelInterface
    {
        T& target;
        Track<T> track;

    public:

        Channel( T& t ) : target( t ) { }

        Keyframe<T>& operator[]( uint32_t index );
        const Keyframe<T>& operator[]( uint32_t index ) const;

        virtual void evaluate( float time ) override 
        {
            if( auto value = track.evaluate(time) )
                target = *value;
        }

        virtual std::optional<float> start() const override { return track.start(); }

        virtual std::optional<float> end() const override { return track.end(); }

        virtual bool empty() const override { return track.keyframes.empty(); }

        virtual Basis interpolation() const override { return track.interpolation; }

        virtual void setInterpolation( Basis basis ) override { track.interpolation = basis; }

        virtual bool extrapolation() const override { return track.extrapolate; }

        virtual void setExtrapolation( bool extrapolate ) override { track.extrapolate = extrapolate; }

        virtual ValueType valueType() const override;

        virtual void resize( size_t size ) override;
        
        virtual bool setKeyframe( uint32_t index, const KeyframeDesc& desc ) override; 

        virtual void sortKeyframes() override { track.sortKeyframes(); }
    };

    using Channel1 = Channel<float>;
    using Channel2 = Channel<float2>;
    using Channel3 = Channel<float3>;
    using Channel4 = Channel<float4>;
    using ChannelQ = Channel<otk::quat>;

    //
    // Implementation
    //

    inline bool requiresTangents(Basis basis)
    {
        if( basis == Basis::Hermite )
            return true;
        return false;
    }

    inline uint8_t dim( ValueType vtype )
    {
        switch( vtype )
        {
            case ValueType::Float: return 1;
            case ValueType::Float2: return 2;
            case ValueType::Float3: return 3;
            case ValueType::Float4: return 4;
            case ValueType::Quat: return 4;
        }
        return 0;
    }


} // end namespace anim
