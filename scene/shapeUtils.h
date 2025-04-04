//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//
#pragma once

#pragma warning( disable : 4389 )
// clang-format off
#include <scene/vertex.h>

#include <OptiXToolkit/ShaderUtil/Aabb.h>
#include <OptiXToolkit/Util/CuBuffer.h>

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>
// clang-format on

//------------------------------------------------------------------------------
enum Scheme
{
    kBilinear = 0,
    kCatmark,
    kLoop
};

struct ShapeDesc
{
    ShapeDesc( char const* iname, std::string const& idata, Scheme ischeme, bool iIsLeftHanded = false )
        : name( iname )
        , data( idata )
        , scheme( ischeme )
        , isLeftHanded( iIsLeftHanded )
    {
    }

    std::string name;
    std::string data;
    Scheme      scheme;
    bool        isLeftHanded;
};

#define MAX_TAG_STR_SIZE 50

//------------------------------------------------------------------------------
struct Shape
{
    // Note: increment the version after changes in the shapeUtils code to trigger cache rebuild.
    static constexpr std::chrono::system_clock::duration version{6};

    // full(er) spec here: http://paulbourke.net/dataformats/mtl/
    // and here: https://en.wikipedia.org/wiki/Wavefront_.obj_file
    struct material
    {
        std::string name;

        float ka[3]     = {0.f, 0.f, 0.f};  // ambient
        float kd[3]     = {.8f, .8f, .8f};  // diffuse
        float ks[3]     = {0.f, 0.f, 0.f};  // specular
        float ke[3]     = {0.f, 0.f, 0.f};  // emissive
        float ns        = 0.f;              // specular exponent
        float ni        = 0.f;              // optical density (1.0=no refraction, glass=1.5)
        float sharpness = 0.f;              // reflection sharpness
        float tf[3]     = {0.f, 0.f, 0.f};  // transmission filter
        float d         = 0.f;              // dissolve factor (1.0 = opaque)
        int   illum     = 4;                // illumination model
        float bm        = 1.f;              // bump multipler
        float bb        = 0.f;              // bump bias

        std::string map_ka;    // ambient
        std::string map_kd;    // diffuse
        std::string map_ks;    // specular
        std::string map_bump;  // bump

        // MTL extensions (exported by Blender & others)
        float Pr     = 0.f;  // roughness
        float Pm     = 0.f;  // metallic
        float Ps     = 0.f;  // sheen
        float Pc     = 0.f;  // clearcoat thickness
        float Pcr    = 0.f;  // clearcoat roughness
        float Ke     = 0.f;  // emissive
        float aniso  = 0.f;  // anisotropy
        float anisor = 0.f;  // anisotropy rotation

        std::string map_ke;   // emissive
        std::string map_pr;   // roughness
        std::string map_pm;   // metalness
        std::string map_rma;  // roughness / metalness / ambient occlusion
        std::string map_orm;  // alt version of rma

        unsigned int udim = 0;
        unsigned int udimMax = 0;

        bool hasUdims() const;
        std::vector<uint32_t> findUdims( const std::filesystem::path& basepath ) const;
        std::unique_ptr<material> resolveUdimPaths( const std::filesystem::path& basepath, uint32_t udim, uint32_t udimMax ) const;
    };

    int findMaterial( char const* name );

    std::string                            mtllib;
    std::vector<unsigned short>            mtlbind;
    std::vector<std::unique_ptr<material>> mtls;

    struct tag
    {
        std::string              name;
        std::vector<int>         intargs;
        std::vector<float>       floatargs;
        std::vector<std::string> stringargs;

        static bool parseTag( char const* stream, tag* t );
        std::string genTag() const;
    };
    std::vector<tag> tags;

    // read from/write to cache
    void writeShape( const std::string& objFile ) const;
    bool readShape( const std::string& objFile );

    int  getNumVertices() const { return (int)verts.size(); }
    int  getNumFaces() const { return (int)nvertsPerFace.size(); }
    int  getFVarWidth() const { return hasUV() ? 2 : 0; }
    bool hasUV() const { return !( uvs.empty() || faceuvs.empty() ); }

    std::filesystem::path filepath;

    std::vector<int> nvertsPerFace;
    std::vector<int> faceverts;
    std::vector<int> faceuvs;
    std::vector<int> facenormals;

    Scheme scheme       = kCatmark;
    bool   isLeftHanded = false;

    // Vertex attributes
    std::vector<Vertex> verts;
    std::vector<float2> uvs;
    std::vector<float3> normals;  // unused for subd's

    otk::Aabb aabb;

    // hard coded startup shape
    static std::unique_ptr<Shape> defaultShape();

    // load a single obj file
    static std::unique_ptr<Shape> loadObjFile(
        const std::filesystem::path& pathStr, bool parseMaterials = true );

};

std::vector<std::unique_ptr<Shape::material>> parseMtllib( const std::filesystem::path& path );
