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

#include <scene/vertex.h>

#include <OptiXToolkit/ShaderUtil/Aabb.h>
#include <OptiXToolkit/Util/CuBuffer.h>

#include <opensubdiv/tmr/surfaceTable.h>
// clang-format on

struct Shape;
struct TessellationCounters;
struct SubdivisionPlanCUDA;
class TopologyCache;
struct TopologyMap;

template <typename U>
struct PatchPointsCUDA;

template <typename T, typename L>
struct SubdLinearCUDA;

// Container for a subdivision mesh

class SubdivisionSurface
{
  public:

    // This (legacy) constructor hashes the mesh topology into a TopologyCache and initializes
    // the device-side data structures corresponding to the Tmr::SurfaceTables for 'vertex' and
    // 'face-vayring' data (position & texcoords).
    SubdivisionSurface( TopologyCache& topologyCache, std::unique_ptr<Shape> shape  );

    bool hasAnimation() const;
    uint32_t numKeyframes() const;
    
    void animate( float t, float frameRate );
    void clearMotionCache();

    uint32_t numVertices() const;
    uint32_t surfaceCount() const;

      // Returns flattened pairs of edge indices suitable for wireframe
    std::vector<uint32_t> getControlCageEdges() const;

    // AABBs in object-space
    std::vector<otk::Aabb> m_aabbKeyframes;
    otk::Aabb m_aabb;

  public:

    template <typename DescriptorT, typename PatchPointT> struct SurfaceTableDeviceData
    {
        CuBuffer<DescriptorT> surface_descriptors;
        CuBuffer<int>         control_point_indices;

        CuBuffer<PatchPointT> patch_points;
        CuBuffer<uint32_t>    patch_points_offsets;
    };

    //
    // 'vertex' limit interpolation surface-table ; see :
    // https://graphics.pixar.com/opensubdiv/docs/subdivision_surfaces.html#vertex-and-varying-data
    //

    SurfaceTableDeviceData<OpenSubdiv::Tmr::SurfaceDescriptor, Vertex> m_vertexDeviceData;

    std::vector<CuBuffer<Vertex>> d_positionKeyframes;
    CuBuffer<Vertex> d_positions;
    CuBuffer<Vertex> d_positionsCached;  // optional, needed for motion vecs for keyframed subds
    
    float m_frameOffset = 0.0f;

    //
    // 'face-varying' (texcoords) limit interpolation surface-table ; see :
    // https://graphics.pixar.com/opensubdiv/docs/subdivision_surfaces.html#face-varying-data-and-topology
    //

    SurfaceTableDeviceData<OpenSubdiv::Tmr::LinearSurfaceDescriptor, TexCoord> m_texcoordDeviceData;

    CuBuffer<TexCoord> d_texcoords;

    CuBuffer<uint16_t> d_materialBindings;

    // Does any bound material have displacements?  Computed on scene load.
    bool m_hasDisplacement = false;

  public:

    // CPU evaluation (legacy)
    Shape const* getShape() const { return m_shape.get(); }

    TopologyMap const* getTopologyMap() const
    {
      return m_topology_map;
    }
  protected:

    TopologyMap const* m_topology_map = nullptr;

    void initDeviceData( const std::unique_ptr<const OpenSubdiv::Tmr::SurfaceTable>& surface_table );

    // CPU evaluation (legacy)

    std::unique_ptr<Shape> m_shape;

    std::unique_ptr<const OpenSubdiv::Tmr::SurfaceTable>       m_surface_table;
    std::unique_ptr<const OpenSubdiv::Tmr::LinearSurfaceTable> m_texcoord_surface_table;
};
