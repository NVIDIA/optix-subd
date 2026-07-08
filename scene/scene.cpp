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

// Fills the single printf integer field of a sequence path template (e.g. "%d",
// "%03d") with a frame index, zero-padded to the parsed field width via "%0*d".
static std::string formatFrameNumber( const std::string& templ, int frame )
{
    size_t pct = templ.find( '%' );
    if( pct == std::string::npos )
        return templ;

    const char* begin = templ.data() + pct + 1;
    const char* end   = templ.data() + templ.size();

    int width = 0;
    const char* next = std::from_chars( begin, end, width ).ptr;

    if( next < end && *next == 'd' )
    {
        char field[32];
        std::snprintf( field, std::size( field ), "%0*d", width, frame );
        return templ.substr( 0, pct ) + field + std::string( next + 1, end );
    }
    return templ;
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

auto parseJsonEnum = []<typename T>( const json::json& node,
    const std::initializer_list<const char*>& enums, T& result ) constexpr {

    assert( size_t( T::COUNT ) == enums.size() );

    uint8_t index = 0;
    for( const char* e : enums )
    {
        if( std::strncmp( node.get<std::string>().c_str(), e, std::strlen( e ) ) == 0 )
        {
            result = T(index);
            break;
        }
        ++index;
    }
};


static SceneArgs& operator<<( SceneArgs& args, const json::json& node )
{

    // Intentionally empty.  Can add parsing for SceneArgs values here if you want the
    // scene to override a value from the UI

    return args;
}

//
// Scene Attributes
//

static Scene::Attributes& operator<<( Scene::Attributes& attrs, const json::json& node )
{
    attrs.frameRange = node.value( "frame range", attrs.frameRange );
    attrs.frameRate  = node.value( "frame rate", attrs.frameRate );
    return attrs;
}

//
// View
//

static View& operator<<( View& view, const json::json& node )
{
    view.position = node.value( "position", view.position );
    view.lookat   = node.value( "lookat", view.lookat );
    view.up       = node.value( "up", view.up );
    if( view.rotation )
        *view.rotation = node.value( "rotation", *view.rotation );
    view.fov = node.value( "fov", view.fov );
    return view;
}

//
// Instance
//

static Instance& operator<<( Instance& instance, const json::json& node )
{
    instance.translation = node.value( "translation", instance.translation );

    if( node.contains( "rotation" ) )
    {
        if( node["rotation"].is_array() && node["rotation"].size() != 4 )
            throw std::runtime_error( "expecting 4-component quaternion for node's 'rotation' (use 'euler' otherwise)" );
        instance.rotation = node.value( "rotation", instance.rotation );
    }
    else if( node.contains( "euler" ) )
    {
        float3 euler      = node.value( "euler", float3{0, 0, 0} );
        euler *= float( M_PI ) / 180.f;
        instance.rotation = otk::rotationEuler<float>( euler );
    }

    instance.scaling = node.value( "scaling", instance.scaling );

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

inline anim::Basis parseChannelModeEnum( const json::json& node )
{
    if( node.empty() )
        throw std::runtime_error( "track interpolation mode expects a string value" );

    std::string mode = node.get<std::string>();

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
                    std::string path = formatFrameNumber( filepath.generic_string(), frameRange.x );
                    shape = Shape::loadObjFile( path );
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
                        
                std::string path = formatFrameNumber( filepath.generic_string(), frame + frameRange.x );

                std::unique_ptr<Shape> s = Shape::loadObjFile( path, false );

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

    uint32_t readFloatArray( const json::json& node, float* dest, uint32_t size ) {
        if( node.is_array() && ( node.size() == size ) )
        {
            for( uint8_t i = 0; i < size; ++i )
                dest[i] = node[i].get<float>();
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

    anim::ChannelInterface::KeyframeDesc loadKeyframe( const json::json& keyframeNode, uint8_t valueSize, bool readTangents )
    {
        anim::ChannelInterface::KeyframeDesc keyframeDesc;

        assert( valueSize <= (uint8_t)std::size( keyframeDesc.value ) );

        keyframeDesc.time = keyframeNode.value( "time", 0.f );

        if( auto it = keyframeNode.find( "value" ); it != keyframeNode.end() && it->is_array() && ( it->size() == valueSize ) )
        {
            keyframeDesc.valueSize = uint8_t( readFloatArray( *it, keyframeDesc.value, valueSize ) );

            if( readTangents )
            {
                if( auto inTangentIt = keyframeNode.find( "in-tangent" );
                    inTangentIt != keyframeNode.end() && inTangentIt->is_array() )
                    readFloatArray( *inTangentIt, keyframeDesc.inTangent, valueSize );
                if( auto outTangentIt = keyframeNode.find( "out-tangent" );
                    outTangentIt != keyframeNode.end() && outTangentIt->is_array() )
                    readFloatArray( *outTangentIt, keyframeDesc.outTangent, valueSize );
            }
        }
        else
            throw std::runtime_error( "invalid keyframe value token : expected a numeric array" );
        return keyframeDesc;
    }

    std::unique_ptr<anim::ChannelInterface> loadChannel( const json::json& channelNode )
    {
        std::unique_ptr<anim::ChannelInterface> channel;

        if( auto it = channelNode.find( "target" ); it != channelNode.end() && it->is_string() && !it->empty() )
            channel = resolveChannelTarget( it->get<std::string>() );

        if( !channel )
            return nullptr;

        if( auto it = channelNode.find( "mode" ); it != channelNode.end() && it->is_string() )
            channel->setInterpolation( parseChannelModeEnum( *it ) );
        else
            throw std::runtime_error( "channel interpolation mode expects a string typed value" );

        if( auto it = channelNode.find( "data" ); it != channelNode.end() && it->is_array() && !it->empty() )
        {
            channel->resize( it->size() );

            bool requiresTangents = anim::requiresTangents( channel->interpolation() );

            uint8_t valueSize = anim::dim( channel->valueType() );

            for( uint32_t keyframeIndex = 0; keyframeIndex < it->size(); ++keyframeIndex )
            {
                auto keyframeDesc = loadKeyframe( (*it)[keyframeIndex], valueSize, requiresTangents );

                if( !channel->setKeyframe( keyframeIndex, keyframeDesc ) )
                    throw std::runtime_error( "incorrect array size for keyframe value" );
            }

            channel->sortKeyframes();
        }
        return channel;
    }
    std::unique_ptr<Sequence> loadSequence( const json::json& sequenceNode )
    {
        auto sequence = std::make_unique<Sequence>();

        sequence->name = sequenceNode.value( "name", "" );

        if( auto it = sequenceNode.find( "channels" ); it != sequenceNode.end() && it->is_array() && !it->empty() )
        {
            const auto& channelsNode = *it;
            sequence->channels.reserve( channelsNode.size() );

            for( const auto& channelNode : channelsNode )
            {
                if( auto channelIt = channelNode.find( "target" );
                    channelIt != channelNode.end() && channelIt->is_string() && !channelIt->empty() )
                {
                    if( auto channel = loadChannel( channelNode ); channel && !channel->empty() )
                    {
                        // automatic detection of sequence start/end
                        sequence->start = std::min( sequence->start, *channel->start() );
                        sequence->end   = std::max( sequence->end, *channel->end() );
                        sequence->channels.emplace_back( std::move( channel ) );
                    }
                }
            }
        }

        // override sequence start/end if the user specified either
        sequence->start = sequenceNode.value( "start", sequence->start );
        sequence->end   = sequenceNode.value( "end", sequence->end );

        if( sequence->end < sequence->start )
            std::swap( sequence->start, sequence->end );

        return sequence;
    }

    std::unique_ptr<Animation> loadAnimation( const json::json& animationNode )
    {
        auto animation = std::make_unique<Animation>();

        animation->name = animationNode.value( "name", "" );

        if( auto it = animationNode.find( "sequences" ); it != animationNode.end() && it->is_array() && !it->empty() )
        {
            const auto& sequencesNode = *it;
            animation->sequences.resize( sequencesNode.size() );

            for( uint32_t i = 0; i < sequencesNode.size(); ++i )
            {
                if( auto sequence = loadSequence( sequencesNode[i] ) )
                {
                    animation->start       = std::min( animation->start, sequence->start );
                    animation->end         = std::min( animation->end, sequence->end );
                    animation->sequences[i] = std::move( sequence );
                }
            }
        }

        // force sequences to be in chronological order
        std::sort( animation->sequences.begin(), animation->sequences.end(),
                   []( const std::unique_ptr<Sequence>& a, const std::unique_ptr<Sequence>& b ) { return a->start < b->start; } );

        assert( animation->end >= animation->start );
        return animation;
    }

public:

    AnimationLoader( Scene& s ) : m_scene(s) {}

    bool loadAnimations( const json::json& rootNode )
    {
        if( auto it = rootNode.find( "animations" ); it != rootNode.end() && it->is_array() && !it->empty() )
        {
            const auto& animationsNode = *it;
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
        if( auto json_root = readFile( m_filepath ); json_root.is_object() )
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

    if( !fs::is_regular_file( fp ) && fs::is_regular_file( modeLoader.mediapath / fp ) )
        fp = modeLoader.mediapath / fp;

    if( auto json_root = readFile( fp ); json_root.is_object() )
    {
        modeLoader.modelpath = fp.parent_path();

        const auto& models = json_root.value("models", json::json::array());
        const auto& graph = json_root.value("graph", json::json::array());

        if( !models.is_array() || !graph.is_array() )
            throw std::runtime_error( "need valid 'models' and 'graph' arrays in '" + fp.generic_string() + "'" );

        uint32_t nmodels = static_cast<uint32_t>( models.size() );

        for( uint32_t i = 0; i < graph.size(); ++i )
        {
            const json::json& node = graph[i];

            Instance instance;

            instance << node;

            std::string nodeName = "<unnamed instance>";
            if( auto it = node.find( "name" ); it != node.end() && it->is_string() )
                instance.name = m_instanceNames.emplace( it->get<std::string>() ).first->c_str();

            if( auto it = node.find( "model" ); it != node.end() && it->is_number_integer() )
            {
                int modelIndex = it->get<int>();
                if( modelIndex < 0 || modelIndex >= nmodels )
                    throw std::runtime_error( "out of bounds 'model' index for graph node '" + nodeName + "'" );

                const json::json& modelName = models[modelIndex];

                if( !modelName.is_string() )
                    throw std::runtime_error( "invalid model path in 'models' section" );

                float frameOffset = node.value( "frameoffset", 0.0f );

                Model model = modeLoader.loadModel( modelName.get<std::string>(), instance, {0, 0}, frameOffset );

                insertModel( std::move( model ) );
            }

            if( node.contains( "type" ) )
                throw std::runtime_error( "'type' token for graph node '" + nodeName + "' not supported" );

            if( node.contains( "parent" ) )
                throw std::runtime_error( "'parent' token for graph node '" + nodeName + "' not supported" );

            if( node.contains( "children" ) )
                throw std::runtime_error( "'children' token for graph node '" + nodeName + "' not supported" );
        }

        if( auto it = json_root.find( "view" ); it != json_root.end() && it->is_object() )
        {
            if( !m_defaultView )
                m_defaultView = std::make_unique<View>();
            *m_defaultView << *it;
        }

        AnimationLoader( *this ).loadAnimations( json_root );

        if( auto it = json_root.find( "settings" ); it != json_root.end() && it->is_object() )
        {
            modeLoader.args << *it;
            m_attributes << *it;
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

bool Scene::animate( const FrameTime& frameTime, bool& isDiscontinuous )
{
    isDiscontinuous = false;
    if( frameTime.currentTime == m_lastAnimationTime )
    {
        clearMotionCache();
        return false;
    }

    if( m_lastAnimationTime >= 0.0f )
    {
        const float timeDelta = frameTime.currentTime - m_lastAnimationTime;
        // Discontinuity is a jump of more than 10 frames.
        if( frameTime.frameRate > 0 && fabsf( timeDelta ) > 10.0f / frameTime.frameRate )
        {
            isDiscontinuous = true;
        }
    }
    m_lastAnimationTime = frameTime.currentTime;

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
    return true;
}

void Scene::clearMotionCache()
{
    for( auto& subdMesh : m_subdMeshes )
        subdMesh->clearMotionCache();
}


