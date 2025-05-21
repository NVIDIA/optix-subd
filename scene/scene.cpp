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
#include "./scene.h"

#include "./json.h"
#include "./shapeUtils.h"
#include "./sceneTypes.h"

#include <args.h>
#include <utils.h>

#include <material/materialCache.h>
#include <material/materialCuda.h>
#include <subdivision/SubdivisionSurface.h>
#include <subdivision/TopologyCache.h>
#include <subdivision/TopologyMap.h>
#include <texture/textureCache.h>
#include <OptiXToolkit/ShaderUtil/Matrix.h>
#include <OptiXToolkit/ShaderUtil/Quaternion.h>
#include <OptiXToolkit/Util/Exception.h>
#include <statistics.h>

#include <json/json.h>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <execution>
#include <imgui.h>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <string>
#include <type_traits>
// clang-format on

namespace fs = std::filesystem;


static inline float computeScale( const otk::Aabb& aabb )
{
    const float3 e = make_float3( aabb.extent( 0 ), aabb.extent( 1 ), aabb.extent( 2 ) );
    return ( (float*)&e )[aabb.longestAxis()];
}

// parses filenames of type 'filename[start-stop].ext'
static std::optional<int2> getSequenceRange( const std::string& str )
{
    size_t open = str.find( '[' );
    size_t close = str.find( ']' );
    if( open == std::string::npos || close == std::string::npos )
        return {};

    std::string_view range = { str.data() + open + 1, str.data() + close };

    size_t delim = range.find( '-' );
    if( delim == std::string::npos )
        return {};

    int2 framerange = { 0, 0 };
    std::from_chars( range.data(), range.data() + delim, framerange.x );
    std::from_chars( range.data() + delim + 1, range.data() + range.size(), framerange.y );
    return framerange;
}

static std::string getSequenceFormat( const std::string& str, int2 frameRange )
{
    size_t open = str.find( '[' );
    size_t close = str.find( ']' );
    if( open == std::string::npos || close == std::string::npos )
        return str;

    std::string prefix = str.substr( 0, open );
    std::string suffix = str.substr( close+1, std::string::npos );

    // test some common padding formats
    std::array<const char*, 3> formats = { "%d", "%03d", "%04d" };

    for( const char* format : formats )
    {
        char buf[16];
        std::snprintf( buf, std::size( buf ), format, frameRange.x );

        if( fs::is_regular_file( prefix + buf + suffix ) )
            return prefix + format + suffix;
    }
    return str;
}

static fs::path resolveMediapath( const fs::path& filepath, const fs::path& mediapath )
{
    if( filepath.empty() )
        return {};

    if( fs::is_regular_file( filepath ) )
        return filepath;

    if( !mediapath.empty() && fs::is_regular_file( mediapath / filepath ) )
        return mediapath / filepath;
    
    return {};
}

//
// Args
//

auto parseJsonEnum = []<typename T>( const Json::Value& node, 
    const std::initializer_list<const char*>& enums, T& result ) constexpr {

    assert( size_t( T::COUNT ) == enums.size() );

    uint8_t index = 0;
    for( const char* e : enums )
    {
        if( std::strncmp( node.asString().c_str(), e, std::strlen( e ) ) == 0 )
        {
            result = T(index);
            break;
        }
        ++index;
    }
};


static SceneArgs& operator<<( SceneArgs& args, const Json::Value& node )
{

    // Intentionally empty.  Can add parsing for SceneArgs values here if you want the
    // scene to override a value from the UI

    return args;
}

//
// Scene Attributes
//

static Scene::Attributes& operator << (Scene::Attributes& attrs, const Json::Value& node)
{
    if( const auto& value = node["frame range"]; value.isArray() )
        value >> attrs.frameRange;

    if( const auto& value = node["frame rate"]; value.isDouble() )
        value >> attrs.frameRate;

    return attrs;
}

//
// View
//

static View& operator << ( View& view, const Json::Value& node )
{
    if( const auto& value = node["position"]; !value.isNull() )
        value >> view.position;
    if( const auto& value = node["lookat"]; !value.isNull() )
        value >> view.lookat;
    if( const auto& value = node["up"]; !value.isNull() )
        value >> view.up;
    if (const auto& value = node["rotation"]; !value.isNull())
        value >> *view.rotation;
    if( const auto& value = node["fov"]; !value.isNull() )
        value >> view.fov;
    return view;
}

//
// Instance
//

static Instance& operator << ( Instance& instance, const Json::Value& node )
{
    if( const auto& value = node["translation"]; !value.isNull() )
        value >> instance.translation;

    if( const auto& value = node["rotation"]; !value.isNull() )
    {
        if( node.isArray() && node.size() == 4 )
            throw std::runtime_error("expecting 4-component quaternion for node's 'rotation' (use 'euler' otherwise)");
        value >> instance.rotation;
    }
    else if( const auto& value = node["euler"]; !value.isNull() )
    {
        float3 euler = { 0.0, 0.0, 0.0 };
        value >> euler;
        euler *= float( M_PI ) / 180.f;
        instance.rotation = otk::rotationEuler<float>( euler );
    }

    if( const auto& value = node["scaling"]; !value.isNull() )
        value >> instance.scaling;

    instance.updateLocalTransform();

    return instance;
}

void Instance::updateLocalTransform()
{
    localToWorld = otk::Affine::scale( scaling );

    localToWorld = otk::toAffine<float, float3>( rotation ) * localToWorld;

    localToWorld = otk::Affine::translate( translation ) * localToWorld;
}


//
// Track & Channel
//

inline anim::Basis parseChannelModeEnum( const Json::Value& node )
{
    if( node.empty() )
        throw std::runtime_error( "track interpolation mode expects a string value" );

    std::string mode = node.asString();

    using enum anim::Basis;
         if( mode == "step" ) return Step;
    else if( mode == "linear" ) return Linear;
    else if( mode == "slerp" ) return  SLerp;
    else if( mode == "catmull-rom" ) return  CatmullRom;
    else if( mode == "hermite" ) return Hermite;
    else
        throw std::runtime_error( "unknown track interpolation mode : " + mode );
}

//
// Sequence
//

void Sequence::animate( const FrameTime& frameTime )
{
    for( const auto& channel : channels )
        channel->evaluate( frameTime.currentTime );
}

void Animation::animate( const FrameTime & frameTime )
{
    float time = frameTime.currentTime;

    for( uint32_t sequenceIndex = 0; sequenceIndex < (uint32_t) sequences.size(); ++sequenceIndex )
    {
        Sequence& sequence = *sequences[sequenceIndex];
        
        if( time >  sequence.end )
            continue;

        if( ( time >= sequence.start ) && ( time < sequence.end ) )
        {
            sequence.animate( frameTime );
            return;
        }
    }
}

//
// ModelLoader
//

struct Scene::Model
{
    std::vector<Instance>               instances;
    std::unique_ptr<SubdivisionSurface> subd;

    int2      frameRange = { std::numeric_limits<int>::max(),
                             std::numeric_limits<int>::min() };
};

class Scene::ModelLoader
{

    Model loadObjFile( const fs::path& filepath, int2 frameRange, float frameOffset, const Instance& parent )
    {
        Model model;

        int nframes = frameRange.y - frameRange.x + 1;

        auto start = std::chrono::steady_clock::now();
        
        std::unique_ptr<SubdivisionSurface> subd;
        
        {
            std::unique_ptr<Shape> shape;
            if( filepath.empty() )
                shape = Shape::defaultShape();
            else
            {
                if( nframes == 1 )
                    shape = Shape::loadObjFile( filepath.generic_string() );
                else
                {
                    char buf[1024];
                    std::snprintf( buf, std::size( buf ), filepath.generic_string().c_str(), frameRange.x );
                    shape = Shape::loadObjFile( buf );
                }

                if ( shape->uvs.empty() || shape->faceuvs.empty() )
                {
                    // Add some placeholder uvs to simplify tessellation later
                    shape->faceuvs = shape->faceverts;
                    shape->uvs.resize( shape->verts.size(), float2{0} );
                }
            }
            subd = std::make_unique<SubdivisionSurface>( topologyCache, std::move( shape ) );
        }
        
        // upload keyframe data to device
        if( nframes > 1 )
        {
            subd->d_positionKeyframes.resize( nframes );
            subd->m_aabbKeyframes.resize( nframes );

            subd->d_positionKeyframes[0].uploadAsync( subd->getShape()->verts );
            subd->m_aabbKeyframes[0] = subd->getShape()->aabb;

            std::vector<int> frames(nframes - 1);  // excluding the 1st frame.
            std::iota( frames.begin(), frames.end(), 1 );

            std::for_each( std::execution::par_unseq, frames.begin(), frames.end(), [&]( int frame ) {
                        
                char buf[1024];
                std::snprintf( buf, 1024, filepath.generic_string().c_str(), frame + frameRange.x );

                std::unique_ptr<Shape> s = Shape::loadObjFile( buf, false );

                subd->d_positionKeyframes[frame].uploadAsync( s ? s->verts : subd->getShape()->verts );

                subd->m_aabbKeyframes[frame] = s ? s->aabb : subd->getShape()->aabb;
            } );

            if ( frameOffset > 0 )
            {
                subd->m_frameOffset = frameOffset;
                subd->animate( 0.0f, 1.0f );
            }
        }

        // materials & textures
        auto bindings = materialCache.cacheMaterials( *subd->getShape(), subd->surfaceCount() );
        
        if( !bindings.empty() )
        {
            subd->d_materialBindings.upload( bindings );

            // Mark if any materials have displacement maps
            const Shape* shape = subd->getShape();
            for ( const auto& mtl : shape->mtls )
            {
                if ( mtl->map_bump.size() && mtl->bm > 0.0f )
                {
                    subd->m_hasDisplacement = true;
                    break;
                }
            }
        }

        // OBJ have no transforms : local2world = identity * parent.local2world
        Instance instance = parent; 
        instance.aabb = subd->m_aabb.transform( instance.localToWorld );

        model.subd = std::move( subd );
        model.instances.emplace_back( instance );
        model.frameRange = frameRange;

        auto stop = std::chrono::steady_clock::now();
    
        printf( "loaded %s (%f s)\n", filepath.generic_string().c_str(),
            std::chrono::duration<float, std::milli>( stop - start ).count() / 1000.f );

        return model;
    }

  public:
    SceneArgs& args;
    TopologyCache& topologyCache;
    MaterialCache& materialCache;

    fs::path modelpath;
    const fs::path& mediapath;

    Model loadModel( const fs::path& filepath, const Instance& parent, int2 frameRange = { 0, 0 }, float frameOffset = 0.0f )
    {   
        if( filepath.empty() )
            return loadObjFile( filepath, frameRange, frameOffset, parent );

        fs::path fp = modelpath.empty() ? filepath : modelpath / filepath;

        if( !fs::is_regular_file( fp ) && fs::is_regular_file( mediapath / fp ) )
            fp = mediapath / fp;

        if( fp.extension() == ".obj" )
        {
            int2 range = frameRange;
            if( auto r = getSequenceRange( fp.generic_string()); r && (*r).y > (*r).x )
            {
                fp = getSequenceFormat( fp.generic_string(), *r );
                range = *r;
            }            
            return loadObjFile( fp, range, frameOffset, parent );
        } else
            throw std::runtime_error( "unsupported type of model: " + fp.generic_string() );
    }
};


class Scene::AnimationLoader {

    Scene& m_scene;

    uint32_t readFloatArray( const Json::Value& node, float* dest, uint32_t size ) {
        if( node.isArray() && ( node.size() == size ) )
        {
            for( uint8_t i = 0; i < size; ++i )
                dest[i] = node[i].asFloat();
            return size;
        }
        return 0;
    };

    std::unique_ptr<anim::ChannelInterface> resolveChannel( const std::string& target, Instance& instance )
    {
        if( target == ".translation" )
            return std::make_unique<anim::Channel3>( instance.translation );
        else if( target == ".rotation" )
            return std::make_unique<anim::ChannelQ>( instance.rotation );
        else if( target == ".euler" )
            throw std::runtime_error( "animation of 'euler' angle rotations not supported (yet) - use quaternion 'rotation' channel instead" );
        else if( target == ".scaling" )
            return std::make_unique<anim::Channel3>( instance.scaling );        
        else
            throw std::runtime_error( "animation channel '." +  target + "' not supported for a graph instance" );
        return nullptr;
    }

    std::unique_ptr<anim::ChannelInterface> resolveChannel( const std::string& target, View& view )
    {
        if( target == ".position" )
            return std::make_unique<anim::Channel3>( view.position );
        else if( target == ".rotation" )
            return std::make_unique<anim::ChannelQ>( *(view.rotation=otk::quat{}) );
        else if( target == ".lookat" )
            return std::make_unique<anim::Channel3>( view.lookat );
        else if( target == ".up" )
            return std::make_unique<anim::Channel3>( view.up );
        else if( target == ".fov" )
            return std::make_unique<anim::Channel<float>>( view.fov );
        else
            throw std::runtime_error( "animation channel '." +  target + "' not supported for a view" );
        
        return nullptr;
    }

    std::unique_ptr<anim::ChannelInterface> resolveChannelTarget( const fs::path& targetPath )
    {
        fs::path::const_iterator targetType = ++targetPath.begin();

        std::string targetAttribute = targetPath.extension().generic_string();
    
        if( *targetType == "Instance" )
        {
            std::string targetName = targetPath.stem().generic_string();
            if( Instance* instance = m_scene.findInstance( targetName ) )
                return resolveChannel( targetAttribute, *instance );
        }
        else if( *targetType == "View" )
        {
            if( !m_scene.m_defaultView )
                m_scene.m_defaultView = std::make_unique<View>();
            
            m_scene.m_defaultView->isAnimated = true;
            return resolveChannel( targetAttribute, *m_scene.m_defaultView );
        }
        return nullptr;
    }

    anim::ChannelInterface::KeyframeDesc loadKeyframe( const Json::Value& keyframeNode, uint8_t valueSize, bool readTangents )
    {
        anim::ChannelInterface::KeyframeDesc keyframeDesc;

        assert( valueSize <= (uint8_t)std::size( keyframeDesc.value ) );

        if( Json::Value const& timeNode = keyframeNode["time"]; timeNode.isNumeric() )
            keyframeDesc.time = timeNode.asFloat();
        else
            throw std::runtime_error("invalid keyframe time token : expected a numeric value");

        Json::Value const& valueNode = keyframeNode["value"]; 

        if( valueNode.isArray() && ( valueNode.size() == valueSize ) )
        {
            keyframeDesc.valueSize = uint8_t( readFloatArray( valueNode, keyframeDesc.value, valueSize ) );

            if( readTangents )
            {
                if( Json::Value const& inTangentNode = keyframeNode["in-tangent"]; inTangentNode.isArray() )
                    readFloatArray( inTangentNode, keyframeDesc.inTangent, valueSize );
                if( Json::Value const& outTangentNode = keyframeNode["out-tangent"]; outTangentNode.isArray() )
                    readFloatArray( outTangentNode, keyframeDesc.outTangent, valueSize );
            }
        }
        else
            throw std::runtime_error( "invalid keyframe value token : expected a numeric array" );
        return keyframeDesc;
    }

    std::unique_ptr<anim::ChannelInterface> loadChannel( const Json::Value& channelNode )
    {
        std::unique_ptr<anim::ChannelInterface> channel;

        if( const Json::Value& targetNode = channelNode["target"]; targetNode.isString() && !targetNode.empty() )
            channel = resolveChannelTarget( targetNode.asString() );
        
        if( !channel )
            return nullptr;

        if( const Json::Value& modeNode = channelNode["mode"]; modeNode.isString() )
            channel->setInterpolation( parseChannelModeEnum( modeNode ) );
        else
            throw std::runtime_error( "channel interpolation mode expects a string typed value" );

        const Json::Value& dataNode = channelNode["data"];
        if( dataNode.isArray() && !dataNode.empty() )
        {
            channel->resize( dataNode.size() );

            bool requiresTangents = anim::requiresTangents( channel->interpolation() );

            uint8_t valueSize = anim::dim( channel->valueType() );

            for( uint32_t keyframeIndex = 0; keyframeIndex < dataNode.size(); ++keyframeIndex )
            {
                auto keyframeDesc = loadKeyframe( dataNode[keyframeIndex], valueSize, requiresTangents );
                
                if( !channel->setKeyframe( keyframeIndex, keyframeDesc ) )
                    throw std::runtime_error( "incorrect array size for keyframe value" );
            }

            channel->sortKeyframes();
        }
        return channel;
    }
    std::unique_ptr<Sequence> loadSequence( const Json::Value& sequenceNode )
    {
        auto sequence = std::make_unique<Sequence>();

        if( const Json::Value& node = sequenceNode["name"]; node.isString() && !node.empty() )
            sequence->name = node.asString();

        if( const Json::Value& channelsNode = sequenceNode["channels"]; channelsNode.isArray() && !channelsNode.empty() )
        {
            sequence->channels.reserve( channelsNode.size() );

            for( uint32_t i = 0; i < channelsNode.size(); ++i )
            {
                const Json::Value& channelNode = channelsNode[i];

                if( const Json::Value& targetNode = channelNode["target"]; targetNode.isString() && !targetNode.empty() )
                {
                    if( auto channel = loadChannel( channelNode ); channel && !channel->empty() )
                    {
                        // automatic detection of sequence start/end
                        sequence->start = std::min( sequence->start, *channel->start() );
                        sequence->end = std::max( sequence->end, *channel->end() );
                        sequence->channels.emplace_back( std::move( channel ) );
                    }
                }
            }

        }

        // override sequence start/end if the user specified either
        if( const Json::Value& node = sequenceNode["start"]; node.isNumeric() )
            sequence->start = node.asFloat();
        if( const Json::Value& node = sequenceNode["end"]; node.isNumeric() )
            sequence->end = node.asFloat();

        if (sequence->end < sequence->start)
            std::swap( sequence->start, sequence->end );

        return sequence;
    }

    std::unique_ptr<Animation> loadAnimation( const Json::Value& animationNode )
    {
        auto animation = std::make_unique<Animation>();

        if (const Json::Value& nameNode = animationNode["name"]; nameNode.isString() && !nameNode.empty())
            animation->name = nameNode.asString();

        if( const Json::Value& sequencesNode = animationNode["sequences"]; sequencesNode.isArray() && !sequencesNode.empty() )
        {
            animation->sequences.resize(sequencesNode.size());

            for( uint32_t i = 0 ; i < sequencesNode.size(); ++i )
            {
                if( auto sequence = loadSequence( sequencesNode[i] ) )
                {
                    animation->start = std::min( animation->start , sequence->start );
                    animation->end = std::min( animation->end , sequence->end );
                    animation->sequences[i] = std::move(sequence);
                }
            }
        }

        // force sequences to be in chronological order
        std::sort(animation->sequences.begin(), animation->sequences.end(),
            []( const std::unique_ptr<Sequence>& a, const std::unique_ptr<Sequence>& b ) { return a->start < b->start; });

        assert( animation->end >= animation->start );
        return animation;
    }

public:

    AnimationLoader( Scene& s ) : m_scene(s) {}

    bool loadAnimations( const Json::Value& rootNode )
    {
        if( const Json::Value& animationsNode = rootNode["animations"]; !animationsNode.isNull() )
        {
            if( !animationsNode.isArray() || animationsNode.empty() )
                return false;

            m_scene.m_animations.clear();

            m_scene.m_animations.resize( animationsNode.size() );

            for( uint32_t i = 0; i < animationsNode.size(); ++i )
                m_scene.m_animations[i] = loadAnimation( animationsNode[i] );        
            
            return true;
        }
        return false;
    }
};

//
// Scene
//

Scene::~Scene()
{ }

std::unique_ptr<Scene> Scene::create(
    const fs::path&               filepath,
    const fs::path&               mediapath,
    int2                          framerange,
    Args&                         args )
{
    stats::evaluatorSamplers = {};

    auto scene                 = std::make_unique<Scene>();

    scene->m_materialCache    = std::make_unique<MaterialCache>();

    // eventually hash topology off-line & initialize directly from file

    TopologyCache topologyCache( TopologyCache::Options{} );

    ModelLoader loader = {
        .args = args.sceneArgs(),
        .topologyCache = topologyCache,
        .materialCache = *scene->m_materialCache,
        .mediapath = mediapath,
    };

    if( filepath.extension() == ".json" )
    {
        scene->loadSceneFile( filepath, loader );
    }
    else
    {
        Model model = loader.loadModel( filepath, Instance{}, framerange );

        scene->insertModel( std::move( model ) );
    }

    {
        // finalize scene attributes
        Attributes& attrs = scene->m_attributes;

        attrs.averageInstanceScale = 0.0f;
        for( const Instance& instance : scene->m_instances )
        {
            attrs.averageInstanceScale += computeScale( instance.aabb );
            attrs.aabb.include( instance.aabb );
        }

        attrs.averageInstanceScale /= float(scene->m_instances.size());

        if( ( attrs.frameRange.y > attrs.frameRange.x ) && ( attrs.frameRate == 0.f ) )
            attrs.frameRate = 24.f;
    }

    // upload to device

    scene->m_topologyMaps = topologyCache.initDeviceData();

    if( scene->m_materialCache )
    {
        scene->m_materialCache->initDeviceData();

        stats::memUsageSamplers.bcSize = scene->m_materialCache->getTextureCache().memoryUse();
    }

    scene->d_instances.upload( scene->m_instances );

    // Also cache OptiX instances since they won't change
    {
        std::vector<OptixInstance> optixInstances;
        optixInstances.reserve( scene->m_instances.size() ); 

        for( size_t i = 0; i < scene->m_instances.size(); ++i )
        {
            OptixInstance optixInstance = {};
            optixInstance.instanceId                  = (unsigned int)i;
            optixInstance.visibilityMask              = 1,
            optixInstance.flags                       = OPTIX_INSTANCE_FLAG_NONE,
            optixInstance.traversableHandle           = 0,
            optixInstance.sbtOffset                   = 0,
            copy( scene->m_instances[i].localToWorld, optixInstance.transform );
            optixInstances.push_back( optixInstance );
        }
        scene->d_optixInstances.upload( optixInstances );
    }
    

    args.meshInputFile = filepath.lexically_normal().generic_string();


    return scene;
}

Instance* Scene::findInstance( const std::string& instanceName )
{
    for( Instance& instance : m_instances )
        if( instance.name && ( instanceName == instance.name ) )
            return &instance;
    return nullptr;
}

bool Scene::reloadAnimations()
{
    if( fs::is_regular_file( m_filepath ) )
    {
        if( auto json_root = readFile( m_filepath ); json_root.isObject() )
            return AnimationLoader( *this ).loadAnimations( json_root );
    }
    return false;
}

void Scene::insertModel( Model&& model )
{
    if( model.subd )
    {
        OTK_REQUIRE( model.instances.size() == 1 );

        m_attributes.frameRange.x = min( m_attributes.frameRange.x, model.frameRange.x );
        m_attributes.frameRange.y = max( m_attributes.frameRange.y, model.frameRange.y );

        model.instances.front().meshID = uint32_t( m_subdMeshes.size() );  
        
        m_instances.emplace_back( model.instances.front() );

        model.instances.clear();

        m_subdMeshes.emplace_back( std::move( model.subd ) );
    }
    
}

void Scene::loadSceneFile( const fs::path& filepath, Scene::ModelLoader& modeLoader )
{
    fs::path fp = filepath;

    if( !fs::is_regular_file( fp ) && fs::is_regular_file( modeLoader.mediapath / fp )  )
        fp = modeLoader.mediapath / fp;

    if( auto json_root = readFile( fp ); json_root.isObject() )
    {
        modeLoader.modelpath = fp.parent_path();

        const Json::Value& models = json_root["models"];
        const Json::Value& graph = json_root["graph"];

        if( !models.isArray() || !graph.isArray() )
            throw std::runtime_error("need valid 'models' and 'graph' arrays in '" + fp.generic_string() + "'");

        uint32_t nmodels = models.size();

        for( uint32_t i = 0 ; i < graph.size() ; ++i )
        {
            const Json::Value& node = graph[i];

            Instance instance;

            instance << node;

            std::string nodeName = "<unnamed instance>";
            if( const auto& name = node["name"]; name.isString() )
                instance.name = m_instanceNames.emplace( name.asString() ).first->c_str();

            if( const auto& modelNode = node["model"]; !modelNode.isNull() )
            {
                if( !modelNode.isIntegral() )
                    throw std::runtime_error( "'model' value for graph node '" + nodeName + "' must be an index" );

                int modelIndex = modelNode.asInt();
                if( modelIndex < 0 || modelIndex >= nmodels )
                    throw std::runtime_error( "out of bounds 'model' index for graph node '" + nodeName + "'" );

                const Json::Value& modelName = models[modelIndex];

                if( !modelName.isString() )
                    throw std::runtime_error( "invalid model path in 'models' section" );

                float frameOffset = 0.0f;
                if( const auto& offset = node["frameoffset"]; offset.isDouble() )
                {
                    frameOffset = offset.asFloat();
                }

                Model model = modeLoader.loadModel( modelName.asString(), instance, { 0, 0 }, frameOffset );

                insertModel( std::move( model ) );
            }
            
            if( const auto& type = node["type"]; type.isString() )
                throw std::runtime_error( "'type' token for graph node '" + nodeName + "' not supported" );

            if( const auto& parent = node["parent"]; !parent.isNull() )
                throw std::runtime_error( "'parent' token for graph node '" + nodeName + "' not supported" );

            if( const auto& children = node["children"]; !children.isNull() )
                throw std::runtime_error( "'children' token for graph node '" + nodeName + "' not supported" );
        }

        if( const Json::Value& view = json_root["view"]; view.isObject() )
        {
            if( !m_defaultView )
                m_defaultView = std::make_unique<View>();
            *m_defaultView << view;
        }
        
        AnimationLoader( *this  ).loadAnimations( json_root );

        if( Json::Value& settings = json_root["settings"]; settings.isObject() )
        {
            modeLoader.args << settings;
            m_attributes << settings;
        }

        m_filepath = fp;
    }
}



std::span<Instance const> Scene::getSubdMeshInstances() const
{
    return const_cast<Scene*>(this)->getSubdMeshInstances();
}
std::span<Instance> Scene::getSubdMeshInstances()
{
    if (!m_subdMeshes.empty())
        return std::span<Instance>( m_instances );
    return {};
}

uint32_t Scene::totalSubdPatchCount() const
{
    const auto& instances = getSubdMeshInstances();
    const auto& subds = getSubdMeshes();
    uint32_t sum{0};
    for( auto i = instances.begin(); i != instances.end(); ++i)
        sum += subds[i->meshID]->surfaceCount();
    return sum;
}

void Scene::animate( const FrameTime& frameTime )
{
    // pose all animated meshes
    for( auto& subdMesh : m_subdMeshes )
        subdMesh->animate( frameTime.currentTime, frameTime.frameRate );

    // update animation channels
    if( !m_animations.empty() )
    {
        // XXXX hard-wire animation for now - we can extend in the future if
        // we need to select between multiple animations
        m_animations.front()->animate( frameTime );    
    }
}

void Scene::clearMotionCache()
{
    for( auto& subdMesh : m_subdMeshes )
        subdMesh->clearMotionCache();
}


