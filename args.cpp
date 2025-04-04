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

// clang-format off

#include "args.h"

#include <OptiXToolkit/Util/Exception.h>

#include <cstdio>
#include <functional>
#include <map>
#include <stdexcept>
#include <cstring>

// clang-format on

static void printUsageAndExit( const char* argv0, const std::string& token = {} )
{

    // clang-format off

    static char const* msg =
        "Usage  : &s [options]\n"
        "Options: \n"
        "  -h                      | --help                  Print this usage message\n"
        "  -ll <n>                 | --loglevel              Set logCallbackLevel (default: 2)\n"
        "  -i <filepath>           | --input                 Input obj/json file\n"
        "  -p '[eye][at][up]fov'   | --cameraPose            Camera pose\n"
        "  -res <w> <h>            | --resolution            Set image dimensions to <w>x<h>\n"
        "  -o <filename>           | --output                Optional image output file\n"
        "                            --frames <n>            Optional number of frames to use for image output\n";

    // clang-format on

    std::fprintf( stderr, msg, argv0 );

    if( token.find( "Unknown option" ) != std::string::npos )
        std::fprintf( stderr, "\n****** %s ******", token.c_str() );
    else if( !token.empty() )
        std::fprintf( stderr, "\n****** Invalid usage of '%s' ******", token.c_str() );
    exit( 1 );
}


void Args::parse( int argc, char const* const* argv )
{
    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );

        auto parseArgValues = [&argc, &argv, &i, &arg]( int n, std::function<void()> func ) {
            if( i >= argc - n )
                printUsageAndExit( argv[0], argv[i] );
            func();
        };

        if( arg == "--help" || arg == "-h" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( arg == "--output" || arg == "-o" )
        {
            parseArgValues( 1, [&]() { outfile = argv[++i]; } );
        }
        else if( arg == "-p" )
        {
            parseArgValues( 1, [&]() { camString = std::string( argv[++i] ); } );
        }
        else if( arg == "-res" || arg == "--resolution" )
        {
            parseArgValues( 2, [&]() {
                targetResolution.x  = atoi( argv[++i] );
                targetResolution.y = atoi( argv[++i] );
            } );
        }
        else if( arg == "--input" || arg == "-i" )
        {
            parseArgValues( 1, [&]() { meshInputFile = argv[++i]; } );
        }
        else if( arg == "--frames" )
        {
            parseArgValues( 1, [&]() { frames = atoi( argv[++i] ); } );
        }
        else if( arg == "-ll" || arg == "--loglevel" )
        {
            parseArgValues( 1, [&]() { logLevel = atoi( argv[++i] ); } );
        }
        else
            printUsageAndExit( argv[0], std::string( "Unknown option: " ) + argv[i] );
    }
}
