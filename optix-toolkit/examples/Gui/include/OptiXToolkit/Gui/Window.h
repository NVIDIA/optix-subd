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

#include <cuda_runtime.h>
#include <vector_types.h>

#include <cstdlib>
#include <chrono>
#include <vector>

struct GLFWwindow;

namespace otk {

enum BufferImageFormat
{
    UNSIGNED_BYTE4,
    FLOAT4,
    FLOAT3
};

struct ImageBuffer
{
    void* data =      nullptr;
    unsigned int      width = 0;
    unsigned int      height = 0;
    BufferImageFormat pixel_format;
};

size_t pixelFormatSize( BufferImageFormat format );

// Floating point image buffers (see BufferImageFormat above) are assumed to be
// linear and will be converted to sRGB when writing to a file format with 8
// bits per channel.  This can be skipped if disable_srgb is set to true.
// Image buffers with format UNSIGNED_BYTE4 are assumed to be in sRGB already
// and will be written like that.
void        saveImage( const char* filename, const ImageBuffer& buffer, bool disable_srgb );

void displayBufferWindow( const char* title, const ImageBuffer& buffer );

void        initGL();
void        initGLFW();
GLFWwindow* initGLFW( const char* window_title, int width, int height );

// Parse the string of the form <width>x<height> and return numeric values.
void parseDimensions(
        const char* arg,                    // String of form <width>x<height>
        int& width,                         // [out] width
        int& height );                      // [in]  height


// Get current time in seconds for benchmarking/timing purposes.
double currentTime();

// Ensures that width and height have the minimum size to prevent launch errors.
void ensureMinimumSize(
    int& width,                             // Will be assigned the minimum suitable width if too small.
    int& height);                           // Will be assigned the minimum suitable height if too small.

// Ensures that width and height have the minimum size to prevent launch errors.
void ensureMinimumSize(
    unsigned& width,                        // Will be assigned the minimum suitable width if too small.
    unsigned& height);                      // Will be assigned the minimum suitable height if too small.

} // end namespace otk

