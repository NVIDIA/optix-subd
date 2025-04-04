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
#include "./json.h"

#include <json/json.h>

#include <fstream>

// clang-format on
namespace fs = std::filesystem;

Json::Value readFile( const fs::path& filepath )
{
    std::string fp = filepath.generic_string();

    std::stringstream ss;

    {
        std::ifstream ifs( fp );

        if( !ifs )
            throw std::runtime_error( std::string( "Cannot find: " ) + fp );

        ss << ifs.rdbuf();
        ifs.close();
    }

    std::basic_string_view data = ss.view();

    if( data.empty() )
        throw std::runtime_error( "error reading '" + fp + "'" );

    Json::CharReaderBuilder builder;
    builder["collectComments"] = false;

    Json::Value root;

    Json::CharReader* reader = builder.newCharReader();

    std::string errors;
    if( !reader->parse( data.data(), data.data() + data.size(), &root, &errors ) )
        throw std::runtime_error( "JSON parse error ('" + fp + "' " + errors );

    delete reader;

    return root;
}

template <>
std::string read<std::string>( const Json::Value& node, const std::string& defaultValue )
{
    if( node.isString() )
        return node.asString();
    return defaultValue;
}

template <>
bool read<bool>( const Json::Value& node, const bool& defaultValue )
{
    if( node.isBool() )
        return node.asBool();
    if( node.isNumeric() )
        return node.asFloat() != 0.f;
    return defaultValue;
}

template <>
int8_t read<int8_t>( const Json::Value& node, const int8_t& defaultValue )
{
    if( node.isNumeric() )
        return (int8_t)node.asInt();
    return defaultValue;
}
template <>
int16_t read<int16_t>( const Json::Value& node, const int16_t& defaultValue )
{
    if( node.isNumeric() )
        return (int16_t)node.asInt();
    return defaultValue;
}
template <>
int32_t read<int32_t>( const Json::Value& node, const int32_t& defaultValue )
{
    if( node.isNumeric() )
        return (int32_t)node.asInt();
    return defaultValue;
}

template <>
int2 read<int2>( const Json::Value& node, const int2& defaultValue )
{
    if( node.isArray() && node.size() == 2 )
        return make_int2( node[0].asInt(), node[1].asInt() );
    return defaultValue;
}

template <>
int3 read<int3>( const Json::Value& node, const int3& defaultValue )
{
    if( node.isArray() && node.size() == 3 )
        return make_int3( node[0].asInt(), node[1].asInt(), node[2].asInt() );
    return defaultValue;
}

template <>
int4 read<int4>( const Json::Value& node, const int4& defaultValue )
{
    if( node.isArray() && node.size() == 4 )
        return make_int4( node[0].asInt(), node[1].asInt(), node[2].asInt(), node[3].asInt() );
    return defaultValue;
}

template <>
uint8_t read<uint8_t>( const Json::Value& node, const uint8_t& defaultValue )
{
    if( node.isNumeric() )
        return (uint8_t)node.asUInt();
    return defaultValue;
}

template <>
uint16_t read<uint16_t>( const Json::Value& node, const uint16_t& defaultValue )
{
    if( node.isNumeric() )
        return (uint16_t)node.asUInt();
    return defaultValue;
}
template <>
uint32_t read<uint32_t>( const Json::Value& node, const uint32_t& defaultValue )
{
    if( node.isNumeric() )
        return (uint32_t)node.asUInt();
    return defaultValue;
}

template <>
uint2 read<uint2>( const Json::Value& node, const uint2& defaultValue )
{
    if( node.isArray() && node.size() == 2 )
        return make_uint2( node[0].asUInt(), node[1].asUInt() );
    return defaultValue;
}

template <>
uint3 read<uint3>( const Json::Value& node, const uint3& defaultValue )
{
    if( node.isArray() && node.size() == 3 )
        return make_uint3( node[0].asUInt(), node[1].asUInt(), node[2].asUInt() );
    return defaultValue;
}

template <>
uint4 read<uint4>( const Json::Value& node, const uint4& defaultValue )
{
    if( node.isArray() && node.size() == 4 )
        return make_uint4( node[0].asUInt(), node[1].asUInt(), node[2].asUInt(), node[3].asUInt() );
    return defaultValue;
}

template <>
float read<float>( const Json::Value& node, const float& defaultValue )
{
    if( node.isNumeric() )
        return node.asFloat();
    return defaultValue;
}

template <>
float2 read<float2>( const Json::Value& node, const float2& defaultValue )
{
    if( node.isArray() && node.size() == 2 )
        return make_float2( node[0].asFloat(), node[1].asFloat() );
    if( node.isNumeric() )
        return make_float2( node.asFloat() );
    return defaultValue;
}

template <>
float3 read<float3>( const Json::Value& node, const float3& defaultValue )
{
    if( node.isArray() && node.size() == 3 )
        return make_float3( node[0].asFloat(), node[1].asFloat(), node[2].asFloat() );
    if( node.isNumeric() )
        return make_float3( node.asFloat() );
    return defaultValue;
}

template <>
float4 read<float4>( const Json::Value& node, const float4& defaultValue )
{
    if( node.isArray() && node.size() == 4 )
        return make_float4( node[0].asFloat(), node[1].asFloat(), node[2].asFloat(), node[3].asFloat() );
    if( node.isNumeric() )
        return make_float4( node.asFloat() );
    return defaultValue;
}
template <>
double read<double>( const Json::Value& node, const double& defaultValue )
{
    if( node.isNumeric() )
        return node.asDouble();
    return defaultValue;
}

template <>
double2 read<double2>( const Json::Value& node, const double2& defaultValue )
{
    if( node.isArray() && node.size() == 2 )
        return make_double2( node[0].asDouble(), node[1].asDouble() );
    if( node.isNumeric() )
        return make_double2( node.asDouble(), node.asDouble() );
    return defaultValue;
}

template <>
double3 read<double3>( const Json::Value& node, const double3& defaultValue )
{
    if( node.isArray() && node.size() == 3 )
        return make_double3( node[0].asDouble(), node[1].asDouble(), node[2].asDouble() );
    if( node.isNumeric() )
        return make_double3( node.asDouble(), node.asDouble(), node.asDouble() );
    return defaultValue;
}

template <>
double4 read<double4>( const Json::Value& node, const double4& defaultValue )
{
    if( node.isArray() && node.size() == 4 )
        return make_double4( node[0].asDouble(), node[1].asDouble(), node[2].asDouble(), node[3].asDouble() );
    if( node.isNumeric() )
        return make_double4( node.asDouble(), node.asDouble(), node.asDouble(), node.asDouble() );
    return defaultValue;
}
