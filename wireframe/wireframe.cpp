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

#include "wireframe.h"

#include <GBuffer.h>
#include <scene/scene.h>
#include <subdivision/SubdivisionSurface.h>

#include <cuda_runtime.h>
#include <glad/glad.h>

#include <OptiXToolkit/Gui/Camera.h>
#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>
#include <OptiXToolkit/Gui/GLCheck.h>
#include <OptiXToolkit/Util/Exception.h>

#include <iostream>
#include <vector>

struct WireframePass::SubdInstance
{
    const otk::Matrix4x4 localToWorld = otk::Matrix4x4::identity();
    const uint32_t       meshIndex    = ~uint32_t( 0 );
    const otk::Aabb      aabb;
};

static GLuint createGLShader( const std::string& source, GLuint shader_type )
{
    GLuint shader = glCreateShader( shader_type );
    {
        const GLchar* source_data = reinterpret_cast<const GLchar*>( source.data() );
        glShaderSource( shader, 1, &source_data, nullptr );
        glCompileShader( shader );

        GLint is_compiled = 0;
        glGetShaderiv( shader, GL_COMPILE_STATUS, &is_compiled );
        if( is_compiled == GL_FALSE )
        {
            GLint max_length = 0;
            glGetShaderiv( shader, GL_INFO_LOG_LENGTH, &max_length );

            std::string info_log( max_length, '\0' );
            GLchar*     info_log_data = reinterpret_cast<GLchar*>( &info_log[0] );
            glGetShaderInfoLog( shader, max_length, nullptr, info_log_data );

            glDeleteShader( shader );
            std::cerr << "Compilation of shader failed: " << info_log << std::endl;

            return 0;
        }
    }

    GL_CHECK_ERRORS();

    return shader;
}

static GLuint createGLProgram( const std::string& vert_source, const std::string& frag_source )
{
    GLuint vert_shader = createGLShader( vert_source, GL_VERTEX_SHADER );
    if( vert_shader == 0 )
        return 0;

    GLuint frag_shader = createGLShader( frag_source, GL_FRAGMENT_SHADER );
    if( frag_shader == 0 )
    {
        glDeleteShader( vert_shader );
        return 0;
    }

    GLuint program = glCreateProgram();
    glAttachShader( program, vert_shader );
    glAttachShader( program, frag_shader );
    glLinkProgram( program );

    GLint is_linked = 0;
    glGetProgramiv( program, GL_LINK_STATUS, &is_linked );
    if( is_linked == GL_FALSE )
    {
        GLint max_length = 0;
        glGetProgramiv( program, GL_INFO_LOG_LENGTH, &max_length );

        std::string info_log( max_length, '\0' );
        GLchar*     info_log_data = reinterpret_cast<GLchar*>( &info_log[0] );
        glGetProgramInfoLog( program, max_length, nullptr, info_log_data );
        std::cerr << "Linking of program failed: " << info_log << std::endl;

        glDeleteProgram( program );
        glDeleteShader( vert_shader );
        glDeleteShader( frag_shader );

        return 0;
    }

    // flags for deletion but doesn't delete as long as it's attached
    glDeleteShader( vert_shader );
    glDeleteShader( frag_shader );

    GL_CHECK_ERRORS();

    return program;
}

static GLint getGLUniformLocation( GLuint program, const std::string& name )
{
	GLint loc = glGetUniformLocation( program, name.c_str() );
    OTK_ASSERT_MSG( loc != -1, "Failed to get uniform loc for '" + name + "'" );
    return loc;
}

const std::string s_vert_source = R"(
#version 330 core

layout(location = 0) in vec3 vertexPosition;
uniform mat4 mvpMatrix;

void main()
{
    gl_Position = mvpMatrix * vec4(vertexPosition, 1);
}
)";

const std::string s_frag_source = R"(
#version 330 core

in vec4 gl_FragCoord;
layout(location = 0) out vec4 color;
uniform sampler2D depthTexture;
uniform uvec2 screenDims;
uniform float depthBias;
uniform vec2 jitter;

void main()
{

    float d = texture( depthTexture, vec2( (gl_FragCoord.x - jitter.x)/screenDims.x, (gl_FragCoord.y - jitter.y)/screenDims.y) ).x;

    float zLinear = 1.0f / gl_FragCoord.w;

    color = vec4( 0.0f, 1.0f, 0.0f, 1.0f );

    float alpha = 1.0f - clamp( ( zLinear - d ) / depthBias, 0.0f, 1.0f );
    alpha *= 0.7f;
    color.w = alpha;
}
)";


WireframePass::WireframePass( const Scene& scene )
{

    m_program = createGLProgram( s_vert_source, s_frag_source );
    OTK_REQUIRE( m_program );

    m_uniformMvpMatrixLoc = getGLUniformLocation( m_program, "mvpMatrix" );
    m_uniformScreenDimsLoc = getGLUniformLocation( m_program, "screenDims" );
    m_uniformDepthBiasLoc = getGLUniformLocation( m_program, "depthBias" );
    m_depthTextureLoc = getGLUniformLocation( m_program, "depthTexture" );
    m_jitterLoc = getGLUniformLocation( m_program, "jitter" );

    const auto& instances = scene.getSubdMeshInstances();
    if ( instances.empty() ) return;

    m_instances.clear();
    m_instances.reserve( instances.size() );
    for( size_t i = 0; i < instances.size(); ++i )
    {
        const Instance& inst = instances[i];
        otk::Matrix4x4 l2w = otk::Matrix4x4::makeFrom( inst.localToWorld.matrix3x4() );
        m_instances.push_back(
            SubdInstance{ .localToWorld = l2w, .meshIndex = inst.meshID, .aabb = inst.aabb } );
    }

    const auto& subds = scene.getSubdMeshes();
    OTK_REQUIRE( !subds.empty() );

    m_vertexBuffers.resize( subds.size() );
    m_vertex_arrays.resize( subds.size() );
    m_index_buffers.resize( subds.size() );
    m_index_buffer_sizes.resize( subds.size() );

    GL_CHECK( glGenVertexArrays( (GLsizei)m_vertex_arrays.size(), &m_vertex_arrays[0] ) );
    GL_CHECK( glGenBuffers( (GLsizei)m_index_buffers.size(), &m_index_buffers[0] ) );

    for( size_t i = 0; i < subds.size(); ++i )
    {
        GL_CHECK( glBindVertexArray( m_vertex_arrays[i] ) );

        // Vertex positions
        const uint32_t numVertices = subds[i]->numVertices();
        m_vertexBuffers[i] = std::make_unique<otk::CUDAOutputBuffer<float3>>( otk::CUDAOutputBufferType::GL_INTEROP, numVertices, 1 );
        glBindBuffer( GL_ARRAY_BUFFER, m_vertexBuffers[i]->getPBO() );

        GL_CHECK( glEnableVertexAttribArray( 0 ) );
        GL_CHECK( glVertexAttribPointer( 0,         // must match the layout in the shader.
                                         3,         // size
                                         GL_FLOAT,  // type
                                         GL_FALSE,  // normalized?
                                         0,         // stride
                                         0          // array buffer offset
                                         ) );

        // Vertex indices
        GL_CHECK( glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_index_buffers[i] ) );

        std::vector<uint32_t> edgeIndices = subds[i]->getControlCageEdges();
        m_index_buffer_sizes[i]           = edgeIndices.size();
        GL_CHECK( glBufferData( GL_ELEMENT_ARRAY_BUFFER, edgeIndices.size() * sizeof( GLuint ), edgeIndices.data(), GL_STATIC_DRAW ) );
    }


    // Unbind.  State is saved and can be restored by binding again
    GL_CHECK( glBindVertexArray( 0 ) );

    GL_CHECK_ERRORS();

}

WireframePass::~WireframePass( )
{
    GL_CHECK( glDeleteVertexArrays( (GLsizei)m_vertex_arrays.size(), m_vertex_arrays.data() ) );
    GL_CHECK( glDeleteBuffers( (GLsizei)m_index_buffers.size(), m_index_buffers.data() ) );

    if ( m_program )
        glDeleteProgram( m_program );
}



void WireframePass::run( const Scene& scene, const otk::Camera& cam, float2 jitter, uint32_t width, uint32_t height, const ReadWriteResourceInterop<float>& depth )

{
    if ( m_vertex_arrays.empty() ) return;

    // Copy subd vertices into interop buffers
    const auto& subds = scene.getSubdMeshes();
    OTK_ASSERT( subds.size() == m_vertexBuffers.size() );
    for ( size_t i = 0; i < subds.size(); ++i )
    {
        OTK_ASSERT( m_vertexBuffers[i] );
        float3* gl_verts = m_vertexBuffers[i]->map();
        cudaMemcpy( gl_verts, subds[i]->d_positions.data(), subds[i]->d_positions.size() * sizeof(float3), cudaMemcpyDeviceToDevice );
        m_vertexBuffers[i]->unmap();
    }

    // bind GL

    GL_CHECK( glBindFramebuffer( GL_FRAMEBUFFER, 0 ) );
    GL_CHECK( glViewport( 0, 0, width, height ) );

    glUseProgram( m_program );

    glUniform2ui( m_uniformScreenDimsLoc, width , height );

    // Bind depth texture
    glUniform1i( m_depthTextureLoc, 0 ); 
    glActiveTexture( GL_TEXTURE0 );
    GL_CHECK( glBindTexture( GL_TEXTURE_2D, depth.m_glTexId ) );

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    otk::Matrix4x4 viewProj = cam.getViewProjectionMatrix();

    glUniform2f( m_jitterLoc, jitter.x, jitter.y );

    // Smooth lines can be expensive, leave them off by default
    // glEnable( GL_LINE_SMOOTH );
    // glLineWidth( 0.75f );

    for( size_t i = 0; i < m_instances.size(); ++i )
    {
        const SubdInstance& instance = m_instances[i];

        otk::Matrix4x4 mvp = viewProj * instance.localToWorld;
        glUniformMatrix4fv( m_uniformMvpMatrixLoc, 1, /*transpose*/ GL_TRUE, mvp.getData() );
        
        float depthBias = 0.001f * instance.aabb.maxExtent();
        glUniform1f( m_uniformDepthBiasLoc, depthBias );

        GL_CHECK( glBindVertexArray( m_vertex_arrays[instance.meshIndex] ) );
        GL_CHECK( glDrawElements( GL_LINES, GLsizei( m_index_buffer_sizes[instance.meshIndex] ), GL_UNSIGNED_INT, 0 ) );
    }


    GL_CHECK_ERRORS();

    // unbind GL
    glDisable( GL_BLEND );
    glBindVertexArray(0);
    glUseProgram( 0 );

}
