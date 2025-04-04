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

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix_types.h>

#include <stdexcept>
#include <string>

//------------------------------------------------------------------------------
//
// OptiX error-checking
//
//------------------------------------------------------------------------------

#define OPTIX_CHECK( call )                                                    \
    ::otk::optixCheck( call, #call, __FILE__, __LINE__ )

#define OPTIX_CHECK_LOG( call )                                                \
    ::otk::optixCheckLog( call, log, sizeof( log ), sizeof_log, #call, __FILE__, __LINE__ )

// This version of the log-check macro doesn't require the user do setup
// a log buffer and size variable in the surrounding context; rather the
// macro defines a log buffer and log size variable (LOG and LOG_SIZE)
// respectively that should be passed to the message being checked.
// E.g.:
//  OPTIX_CHECK_LOG2( optixProgramGroupCreate( ..., LOG, &LOG_SIZE, ... );
//
#define OPTIX_CHECK_LOG2( call )                                               \
    do                                                                         \
    {                                                                          \
        char   LOG[400];                                                       \
        size_t LOG_SIZE = sizeof( LOG );                                       \
        ::otk::optixCheckLog( call, LOG, sizeof( LOG ), LOG_SIZE, #call,     \
                                __FILE__, __LINE__ );                          \
    } while( false )

#define OPTIX_CHECK_NOTHROW( call )                                            \
    ::otk::optixCheckNoThrow( call, #call, __FILE__, __LINE__ )

//------------------------------------------------------------------------------
//
// CUDA error-checking
//
//------------------------------------------------------------------------------

#define CUDA_CHECK( call ) ::otk::cudaCheck( call, #call, __FILE__, __LINE__ )

#if defined( NDEBUG )

// no-op in Release build
#define CUDA_SYNC_CHECK() do {} while (0)

#else
#define CUDA_SYNC_CHECK() ::otk::cudaSyncCheck( __FILE__, __LINE__ )
#endif


// A non-throwing variant for use in destructors.
// An iostream must be provided for output (e.g. std::cerr).
#define CUDA_CHECK_NOTHROW( call )                                             \
    ::otk::cudaCheckNoThrow( call, #call, __FILE__, __LINE__ )

//------------------------------------------------------------------------------
//
// Assertions
//
//------------------------------------------------------------------------------


#if defined( NDEBUG )

// no-ops in Release build
#define OTK_ASSERT( cond ) do {} while(0)
#define OTK_ASSERT_MSG( cond, msg ) do {} while (0)
#define OTK_ASSERT_FAIL_MSG( msg ) do {} while (0)

#else

#define OTK_ASSERT( cond )                                                   \
    ::otk::assertCond( static_cast<bool>( cond ), #cond, __FILE__, __LINE__ )

#define OTK_ASSERT_MSG( cond, msg )                                          \
    ::otk::assertCondMsg( static_cast<bool>( cond ), #cond, msg, __FILE__, __LINE__ )

#define OTK_ASSERT_FAIL_MSG( msg )                                           \
    ::otk::assertFailMsg( msg, __FILE__, __LINE__ )

#endif


// Macro for requiring conditions. This must never be disabled based on
// build time, i.e. DEBUG builds.
#define OTK_REQUIRE( cond )                                                  \
    ::otk::assertCond( static_cast<bool>( cond ), #cond, __FILE__, __LINE__ )

#define OTK_REQUIRE_MSG( cond, msg )                                          \
    ::otk::assertCondMsg( static_cast<bool>( cond ), #cond, msg, __FILE__, __LINE__ )


namespace otk {

class Exception : public std::runtime_error
{
  public:
    Exception( const char* msg )
        : std::runtime_error( msg )
    {
    }

    Exception( OptixResult res, const char* msg )
        : std::runtime_error( createMessage( res, msg ).c_str() )
    {
    }

  private:
    std::string createMessage( OptixResult res, const char* msg );
};

void optixCheck( OptixResult res, const char* call, const char* file, unsigned int line );

void optixCheckLog( OptixResult res, const char* log, size_t sizeof_log, size_t sizeof_log_returned, const char* call, const char* file, unsigned int line );

void optixCheckNoThrow( OptixResult res, const char* call, const char* file, unsigned int line );

void cudaCheck( cudaError_t error, const char* call, const char* file, unsigned int line );

void cudaCheck( CUresult result, const char* expr, const char* file, unsigned int line );

void cudaSyncCheck( const char* file, unsigned int line );

void cudaCheckNoThrow( cudaError_t error, const char* call, const char* file, unsigned int line ) noexcept;

void cudaCheckNoThrow( CUresult result, const char* expr, const char* file, unsigned int line ) noexcept;

void assertCond( bool result, const char* cond, const char* file, unsigned int line );

void assertCondMsg( bool result, const char* cond, const std::string& msg, const char* file, unsigned int line );

[[noreturn]] void assertFailMsg( const std::string& msg, const char* file, unsigned int line );

}  // end namespace otk
