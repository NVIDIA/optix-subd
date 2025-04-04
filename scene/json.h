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

// clang-format off
#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <cassert>
#include <filesystem>
#include <string>

namespace Json
{
    class Value;
}

Json::Value readFile( const std::filesystem::path& filepath );

template <typename T> T read( const Json::Value& node, const T& defaultValue ) { assert( false ); return T{}; }

template <> std::string read<std::string>( const Json::Value& node, const std::string& defaultValue );

template <> bool read<bool>(const Json::Value& node, const bool& defaultValue);

template <> int8_t read<int8_t>(const Json::Value& node, const int8_t& defaultValue);
template <> int16_t read<int16_t>(const Json::Value& node, const int16_t& defaultValue);
template <> int32_t read<int32_t>( const Json::Value& node, const int32_t& defaultValue );
template <> int2 read<int2>( const Json::Value& node, const int2& defaultValue );
template <> int3 read<int3>( const Json::Value& node, const int3& defaultValue );
template <> int4 read<int4>( const Json::Value& node, const int4& defaultValue );

template <> uint8_t read<uint8_t>( const Json::Value& node, const uint8_t& defaultValue );
template <> uint16_t read<uint16_t>( const Json::Value& node, const uint16_t& defaultValue );
template <> uint32_t read<uint32_t>( const Json::Value& node, const uint32_t& defaultValue );
template <> uint2 read<uint2>( const Json::Value& node, const uint2& defaultValue );
template <> uint3 read<uint3>( const Json::Value& node, const uint3& defaultValue );
template <> uint4 read<uint4>( const Json::Value& node, const uint4& defaultValue );

template <> float read<float>( const Json::Value& node, const float& defaultValue );
template <> float2 read<float2>( const Json::Value& node, const float2& defaultValue );
template <> float3 read<float3>( const Json::Value& node, const float3& defaultValue );
template <> float4 read<float4>( const Json::Value& node, const float4& defaultValue );

template <> double read<double>( const Json::Value& node, const double& defaultValue );
template <> double2 read<double2>( const Json::Value& node, const double2& defaultValue );
template <> double3 read<double3>( const Json::Value& node, const double3& defaultValue );
template <> double4 read<double4>( const Json::Value& node, const double4& defaultValue );

template<typename T> void operator >> ( const Json::Value& node, T& dest )
{
    dest = read<T>(node, dest);
}
// clang-format on
