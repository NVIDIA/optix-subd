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
#include "../shadingTypes.h"

#include <OptiXToolkit/ShaderUtil/Matrix.h>
#include <algorithm>
#include <scene/sceneTypes.h>

#include <OptiXToolkit/ShaderUtil/Aabb.h>
#include <OptiXToolkit/ShaderUtil/Affine.h>
#include <OptiXToolkit/Util/CuBuffer.h>
#include <scene/sceneTypes.h>
#include <subdivision/TopologyCache.h>

#include <filesystem>
#include <memory>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <vector>


// clang-format on

struct Args;
struct Shape;
struct TopologyMap;

class MaterialCache;
class SubdivisionSurface;
class TextureCache;
class TextureCuda;

class Scene
{
  public:

    ~Scene();

    struct Attributes
    {
        int2        frameRange = { std::numeric_limits<int>::max(),
                                   std::numeric_limits<int>::min() };
        float       frameRate = 0.f;
                   
        float       averageInstanceScale = 0.f;
        otk::Aabb   aabb;
    };

    static std::unique_ptr<Scene> create( const std::filesystem::path&  filepath,
                                          const std::filesystem::path&  mediapath,
                                          int2                          framerange,
                                          Args&                         args );

    const std::filesystem::path& getFilepath() const { return m_filepath; }

    bool reloadAnimations();

    void animate( const FrameTime& frameTime );

    void clearMotionCache();

    const View* getView() const { return m_defaultView.get(); }

    const std::vector<std::unique_ptr<SubdivisionSurface>>& getSubdMeshes() const { return m_subdMeshes; }
    const std::vector<std::unique_ptr<TopologyMap const>>&  getTopologyMaps() const { return m_topologyMaps; }

    std::span<Instance>       getSubdMeshInstances();
    std::span<Instance const> getSubdMeshInstances() const;

    std::span<OptixInstance const> getOptixInstancesDevice() const
    {
        return d_optixInstances.span(); 
    }

    uint32_t totalSubdPatchCount() const;

    const CuBuffer<Instance>& getDeviceInstances() const { return d_instances; }

    const Attributes& getAttributes() const { return m_attributes; }

    const MaterialCache& getMaterialCache() const { return *m_materialCache; }

  private:

    struct Model;
    class ModelLoader;
    class AnimationLoader;

    Instance* findInstance( const std::string& instanceName );

    void insertModel( Model&& model );

    void loadSceneFile( const std::filesystem::path& filepath, ModelLoader& modelLoader );

    std::filesystem::path m_filepath;

    // animations
    std::vector<std::unique_ptr<Animation>> m_animations;

    // cameras
    std::unique_ptr<View> m_defaultView;

    // materials
    std::unique_ptr<MaterialCache> m_materialCache;


    // subds
    std::vector<std::unique_ptr<TopologyMap const>>  m_topologyMaps;
    std::vector<std::unique_ptr<SubdivisionSurface>> m_subdMeshes;

    // Instances
    // note : for BVH build reasons, instances are sorted at load-time
    // into a single contiguous array.
    // Assumes that there can only be 1 instance per subdMesh
    CuBuffer<Instance>    d_instances;
    CuBuffer<OptixInstance> d_optixInstances;
    std::vector<Instance> m_instances;
    std::set<std::string> m_instanceNames;

    Attributes m_attributes;
};

