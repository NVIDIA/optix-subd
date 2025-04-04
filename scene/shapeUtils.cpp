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

// clang-format off
#include "./shapeUtils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <map>
#include <tuple>
#include <type_traits>

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/Util/CuBuffer.h>

namespace fs = std::filesystem;

// clang-format on

static std::string readAsciiFile( char const* filepath )
{
    std::ifstream ifs( filepath );

    if( !ifs )
        throw std::runtime_error( std::string( "Cannot find: " ) + filepath );

    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();

    std::string s = ss.str();
    if( s.empty() )
        throw std::runtime_error( std::string( "Read error: " ) + filepath );

    return std::move( s );
}


static char const* sgets( char* s, int size, char** stream )
{
    for( int i = 0; i < size; ++i )
    {
        if( ( *stream )[i] == '\n' || ( *stream )[i] == '\0' )
        {

            memcpy( s, *stream, i );
            s[i] = '\0';

            if( ( *stream )[i] == '\0' )
                return 0;
            else
            {
                ( *stream ) += i + 1;
                return s;
            }
        }
    }
    return 0;
}

static std::vector<std::unique_ptr<Shape::material>> parseMtllib( char const* mtlstr )
{
    std::vector<std::unique_ptr<Shape::material>> mtls;

    char *str = const_cast<char*>( mtlstr ), line[256];

    Shape::material* mtl = nullptr;

    bool  done = false;
    float r, g, b, a;
    while( !done )
    {
        done      = sgets( line, sizeof( line ), &str ) == 0;
        char* end = &line[strlen( line ) - 1];
        if( *end == '\n' )
            *end = '\0';  // strip trailing nl
        switch( line[0] )
        {
            case 'n':
            {
                char name[256] = {""};
                if( sscanf( line, "newmtl %s", name ) == 1 )
                {
                    mtl       = mtls.emplace_back( std::make_unique<Shape::material>() ).get();
                    mtl->name = name;
                }
            }
            break;
            case 'K':
                if( sscanf( line + 2, " %f %f %f", &r, &g, &b ) == 3 )
                {
                    switch( line[1] )
                    {
                        case 'a':
                            mtl->ka[0] = r;
                            mtl->ka[1] = g;
                            mtl->ka[2] = b;
                            break;
                        case 'd':
                            mtl->kd[0] = r;
                            mtl->kd[1] = g;
                            mtl->kd[2] = b;
                            break;
                        case 's':
                            mtl->ks[0] = r;
                            mtl->ks[1] = g;
                            mtl->ks[2] = b;
                            break;
                        case 'e':
                            mtl->ke[0] = r;
                            mtl->ke[1] = g;
                            mtl->ke[2] = b;
                            break;
                    }
                }
                break;
            case 'N':
                if( sscanf( line + 2, " %f", &a ) == 1 )
                {
                    switch( line[1] )
                    {
                        case 's':
                            mtl->ns = a;
                            break;
                        case 'i':
                            mtl->ni = a;
                            break;
                    }
                }
                break;
            case 'd':
                if( sscanf( line, "d %f", &a ) == 1 )
                    mtl->d = a;
                break;
            case 'T':
                if( line[1] == 'f' )
                {
                    if( sscanf( line, "Tf %f %f %f", &r, &g, &b ) == 3 )
                    {
                        mtl->tf[0] = r;
                        mtl->tf[1] = g;
                        mtl->tf[2] = b;
                    }
                    break;
                }
                break;
            case 'i':
                int illum;
                if( sscanf( line, "illum %d", &illum ) == 1 )
                    mtl->illum = illum;
                break;
            case 's':
                if( sscanf( line, "sharpness %f", &a ) == 1 )
                    mtl->sharpness = a;
                break;
            case 'm':
                if( strncmp( line, "map_", 4 ) == 0 )
                {
                    char buf[1024];
                    buf[1023] = '\0';

                    switch( line[4] )
                    {
                        case 'K':
                            if( sscanf( line + 6, " %1023s", buf ) == 1 )
                                switch( line[5] )
                                {
                                    case 'a':
                                        mtl->map_ka = buf;
                                        break;
                                    case 'd':
                                        mtl->map_kd = buf;
                                        break;
                                    case 'e':
                                        mtl->map_ke = buf;
                                        break;
                                    case 's':
                                        mtl->map_ks = buf;
                                        break;
                                }
                            break;
                        case 'B':
                                 if( sscanf( line + 5, "ump -bm %f -bb %f %1023s", &mtl->bm, &mtl->bb, buf ) == 3 )
                                mtl->map_bump = buf;
                            else if( sscanf( line + 5, "ump -bm %f %1023s", &mtl->bm, buf ) == 2 )
                                mtl->map_bump = buf;
                            else if( sscanf( line + 5, "ump %1023s", buf ) == 1 )
                                mtl->map_bump = buf;
                            break;

                        case 'P':
                            if( sscanf( line + 6, " %1023s", buf ) == 1 )
                                switch( line[5])
                                {
                                    case 'r':
                                        mtl->map_pr = buf;
                                        break;
                                    case 'm':
                                        mtl->map_pm = buf;
                                        break;
                                }
                            break;
                        case 'R':
                            if( sscanf( line + 5, "ma %1023s", buf ) == 1 )
                                mtl->map_pr = buf;
                            break;
                        case 'O':
                            if( sscanf( line + 5, "rm %1023s", buf ) == 1 )
                                mtl->map_pr = buf;
                            break;
                        
                    }
                }
                break;
            case 'P':
                switch( line[1] )
                {
                    case 'r':
                        if( sscanf( line + 2, " %f", &mtl->Pr ) == 1 )
                            break;
                    case 'm':
                        if( sscanf( line + 2, " %f", &mtl->Pm ) == 1 )
                            break;
                    case 's':
                        if( sscanf( line + 2, " %f", &mtl->Ps ) == 1 )
                            break;
                    case 'c':
                        switch( line[2] )
                        {
                            case ' ':
                                if( sscanf( line + 2, " %f", &mtl->Pc ) == 1 )
                                    break;
                            case 'r':
                                if( sscanf( line + 3, " %f", &mtl->Pcr ) == 1 )
                                    break;
                        }
                }
                break;
            case 'a':
            {
                float a = 0.f;
                if( sscanf( line, "aniso %f", &a ) == 1 )
                    mtl->aniso = a;
                else if( sscanf( line, "anisor %f", &a ) == 1 )
                    mtl->anisor = a;
            }
            break;
        }
    }
    return mtls;
}

std::unique_ptr<Shape> parseObj( char const* filepath, Scheme shapescheme, bool isLeftHanded, bool parsemtl )
{
    std::string shapestr = readAsciiFile( filepath );

    std::unique_ptr<Shape> s = std::make_unique<Shape>();

    s->scheme       = shapescheme;
    s->isLeftHanded = isLeftHanded;

    char *str    = const_cast<char*>( shapestr.c_str() ), line[1024], buf[256];
    short usemtl = -1;
    bool  done   = false;
    
    uint32_t faceId = 0;
    char groupName[512] = { 0 };

    s->aabb.invalidate();
    while( !done )
    {
        done = sgets( line, sizeof( line ), &str ) == 0;
        if( line[0] )
        {
            char* end = &line[strlen( line ) - 1];
            if( *end == '\n' )
                *end = '\0';  // strip trailing nl
        }
        float x, y, z, u, v;
        switch( line[0] )
        {
            case 'v':
                switch( line[1] )
                {
                    case ' ':
                        if( sscanf( line, "v %f %f %f", &x, &y, &z ) == 3 )
                        {
                            s->verts.push_back( {x, y, z} );
                            s->aabb.include( make_float3( x, y, z ) );
                        }
                        break;
                    case 't':
                        if( sscanf( line, "vt %f %f", &u, &v ) == 2 )
                        {
                            s->uvs.push_back( {u, v} );
                        }
                        break;
                    case 'n':
                        if( sscanf( line, "vn %f %f %f", &x, &y, &z ) == 3 )
                        {
                            s->normals.push_back( {x, y, z} );
                        }
                        break;  // skip normals for now
                }
                break;
            case 'f':
                if( line[1] == ' ' )
                {
                    int         vi, ti, ni;
                    const char* cp = &line[2];
                    while( *cp == ' ' )
                        cp++;
                    int nverts = 0, nitems = 0;
                    while( ( nitems = sscanf( cp, "%d/%d/%d", &vi, &ti, &ni ) ) > 0 )
                    {
                        nverts++;
                        s->faceverts.push_back( vi - 1 );
                        if( nitems > 1 )
                            s->faceuvs.push_back( std::max( 0, ti - 1 ) );
                        if( nitems > 2 )
                            s->facenormals.push_back( std::max( 0, ni - 1 ) );
                        while( *cp && *cp != ' ' )
                            cp++;
                        while( *cp == ' ' )
                            cp++;
                    }
                    s->nvertsPerFace.push_back( nverts );
                    if( !s->mtls.empty() )
                    {
                        s->mtlbind.push_back( usemtl );
                    }
                    ++faceId;
                }
                break;
            case 't':
                if( line[1] == ' ' )
                {
                    Shape::tag t;
                    if( Shape::tag::parseTag( line, &t ) )
                        s->tags.emplace_back( std::move( t ) );
                }
                break;
            case 'g':
                if( line[1] == ' ' )
                {
                    sscanf( line, "g %s", groupName );
                    faceId = 0;
                }
            case 'u':
                if( parsemtl && sscanf( line, "usemtl %s", buf ) == 1 )
                {
                    usemtl = static_cast<short>( s->findMaterial( buf ) );
                }
                break;
            case 'm':
                if( parsemtl && sscanf( line, "mtllib %s", buf ) == 1 )
                {
                    fs::path p = buf;
                    if( ! fs::is_regular_file( p ) )
                        p = fs::path( filepath ).parent_path() / buf;
                    if( fs::is_regular_file(p) )
                    {
                        s->mtls = parseMtllib(readAsciiFile(p.generic_string().c_str()).c_str());
                        s->mtllib = buf;
                    }
                }
                break;
            case 'c':
                if( parsemtl && sscanf( line, "capslib %s", buf ) == 1 )
                {
                    // ignore
                }
                break;
            case 'o':  // check for lfs-standin files
                if( sscanf( line, "oid sha256:%s", buf ) == 1 )
                {
                    return nullptr;
                }
                break;
        }
    }
    return s;
}

bool Shape::tag::parseTag( char const* line, tag* t )
{

    const char* cp = &line[2];

    char buf[256];

    while( *cp == ' ' )
        cp++;
    if( std::sscanf( cp, "%255s", buf ) != 1 )
        return false;
    while( *cp && *cp != ' ' )
        cp++;
    t->name = buf;

    int nints = 0, nfloats = 0, nstrings = 0;
    while( *cp == ' ' )
        cp++;
    if( sscanf( cp, "%d/%d/%d", &nints, &nfloats, &nstrings ) != 3 )
        return false;
    while( *cp && *cp != ' ' )
        cp++;

    t->intargs.reserve( nints );
    for( int i = 0; i < nints; ++i )
    {
        int val;
        while( *cp == ' ' )
            cp++;
        if( sscanf( cp, "%d", &val ) != 1 )
            return false;
        t->intargs.push_back( val );
        while( *cp && *cp != ' ' )
            cp++;
    }

    t->floatargs.reserve( nfloats );
    for( int i = 0; i < nfloats; ++i )
    {
        float val;
        while( *cp == ' ' )
            cp++;
        if( sscanf( cp, "%f", &val ) != 1 )
            return false;
        t->floatargs.push_back( val );
        while( *cp && *cp != ' ' )
            cp++;
    }

    t->stringargs.reserve( nstrings );
    for( int i = 0; i < nstrings; ++i )
    {
        char val[512];
        while( *cp == ' ' )
            cp++;
        if( sscanf( cp, "%512s", val ) != 1 )
            return false;
        t->stringargs.push_back( std::string( val ) );
        while( *cp && *cp != ' ' )
            cp++;
    }
    return true;
}

std::string Shape::tag::genTag() const
{

    std::stringstream t;

    t << "t " << name << " ";

    t << intargs.size() << "/" << floatargs.size() << "/" << stringargs.size() << " ";

    std::copy(intargs.begin(), intargs.end(), std::ostream_iterator<int>(t, " "));
    //t<<" ";

    t << std::fixed;
    std::copy(floatargs.begin(), floatargs.end(), std::ostream_iterator<float>(t, " "));
    //t<<" ";

    std::copy(stringargs.begin(), stringargs.end(), std::ostream_iterator<std::string>(t, " "));
    t << "\n";

    return t.str();
}


//
// udim
//

static std::array<char const*, 2> udim_patterns = {".<UDIM>", ".<UVTILE>"};

static std::string udimPath( std::string const& filename, char const* udimID = nullptr )
{
    for( char const* pattern : udim_patterns )
        if( size_t p = filename.find( pattern ); p != std::string::npos )
            return filename.substr( 0, p + 1 ) + ( udimID ? udimID : "%04d" )
                   + filename.substr( p + strlen( pattern ), std::string::npos );
    return {};
};

template <typename... Ts>
constexpr auto materialMaps( Shape::material& m )
{
    return std::forward_as_tuple( m.map_ka, m.map_kd, m.map_ks, m.map_bump, m.map_ke, m.map_pr, m.map_pm, m.map_rma, m.map_orm );
}
template <typename... Ts>
constexpr auto materialMaps( Shape::material const& m )
{
    return std::forward_as_tuple( m.map_ka, m.map_kd, m.map_ks, m.map_bump, m.map_ke, m.map_pr, m.map_pm, m.map_rma, m.map_orm );
}
static bool hasUdims( Shape::material const& mtl )
{
    bool result = false;

    auto hasUdim = [&result]( std::string const& texpath ) {
        if( result || texpath.empty() )
            return;
        for( auto pattern : udim_patterns )
            if( size_t p = texpath.find( pattern ); p != std::string::npos )
            {
                result = true;
                return;
            }
    };

    std::apply( [&result, &hasUdim]( auto const&... maps ) { ( hasUdim( maps ), ... ); }, materialMaps( mtl ) );
    return result;
}

static bool hasUdims( Shape const& shape )
{
    for( auto& mtl : shape.mtls )
        if( hasUdims( *mtl ) )
            return true;
    return false;
}

static std::unique_ptr<Shape::material> resolveUdim( fs::path const& basepath, Shape::material const& mtl, uint32_t udim )
{
    auto new_mtl = std::make_unique<Shape::material>( mtl );

    auto resolve = [&basepath, &udim]( std::string& texpath ) {
        if( texpath.empty() )
            return;

        texpath = udimPath( texpath, std::to_string( udim ).c_str() );

        if( !fs::is_regular_file( basepath / texpath ) )
            throw std::runtime_error( std::string( "cannot find udim: " ) + ( basepath / texpath ).generic_string().c_str() );
    };

    std::apply( [&resolve]( auto&... maps ) { ( resolve( maps ), ... ); }, materialMaps( *new_mtl ) );

    new_mtl->udim = udim;

    return new_mtl;
}

static void resolveUdims( Shape& shape )
{
    if( !hasUdims( shape ) || shape.mtlbind.empty() || shape.faceuvs.empty() )
        return;

    fs::path basepath = shape.filepath.parent_path();

    // generate new library where materials with udims are duplicated

    std::vector<std::unique_ptr<Shape::material>> mtls;
    mtls.reserve( shape.mtls.size() * 2 );

    std::map<uint64_t, uint32_t> mtlsMap;

    auto makeKey = []( uint32_t mtlid, uint32_t udim ) { return uint64_t( mtlid ) << 32 | uint64_t( udim ); };

    for( uint32_t i = 0; i < shape.mtls.size(); ++i )
    {
        auto mtl = std::move( shape.mtls[i] );

        if( hasUdims( *mtl ) )
        {
            std::vector<uint32_t> udims = mtl->findUdims( basepath );

            for( uint32_t udim : udims )
            {
                //printf("material %s : %d udim: %d -> %d\n", mtl->name.c_str(), i, udim, (uint32_t)mtls.size());
                mtlsMap[makeKey( i, udim )] = static_cast<uint32_t>( mtls.size() );
                mtls.emplace_back( mtl->resolveUdimPaths( basepath, udim, udims.back() - 1001 ) );
            }
        }
        else
        {
            //printf("material %s : %d -> %d\n", mtl->name.c_str(), i, (uint32_t)mtls.size());
            mtlsMap[makeKey( i, 0 )] = static_cast<uint32_t>( mtls.size() );
            mtls.emplace_back( std::move( mtl ) );
        }
    }

    mtls.shrink_to_fit();

    // re-assign material bindings

    assert( shape.mtlbind.size() == shape.getNumFaces() );

    // see: https://learn.foundry.com/katana/Content/ug/checking_uvs/multi_tile_textures.html
    auto makeUdim = []( float2 uv ) -> uint32_t {
        return 1001 + uint32_t( std::trunc( uv.x ) + 10 * std::trunc( uv.y ) );
    };

    std::vector<unsigned short> mtlbind( shape.mtlbind.size() );

    for( uint32_t face = 0, vertCount = 0; face < shape.getNumFaces(); ++face )
    {
        uint32_t nverts = shape.nvertsPerFace[face];
        uint32_t mtlId  = shape.mtlbind[face];

        auto it = mtlsMap.find( makeKey( mtlId, 0 ) );

        if( it == mtlsMap.end() )
        {
            float2 texcoord = shape.uvs[shape.faceuvs[vertCount]];

            uint32_t udim = makeUdim( texcoord );

            it = mtlsMap.find( makeKey( mtlId, udim ) );

            if( it == mtlsMap.end() )
                throw std::runtime_error( std::string( "texcoord references missing udim texture " ) + std::to_string( udim ) );
                
            assert( mtls[it->second]->udim == udim );

            for( int vert = 1; vert < nverts; ++vert )
            {
                texcoord = shape.uvs[shape.faceuvs[vertCount + vert]];

                if( makeUdim( texcoord ) != udim )
                    throw std::runtime_error( std::string( "texcoord crosses udim bounds for face " + std::to_string( face ) ) );
            }
        }
        else
            assert( mtls[it->second]->udim == 0 );

        mtlbind[face] = static_cast<short>( it->second );

        vertCount += nverts;
    }

    shape.mtls    = std::move( mtls );
    shape.mtlbind = std::move( mtlbind );
}

//
// serialization / deserialization
//

auto writeTrivial = []<typename T>( std::ofstream & os, T const& v ) -> std::ofstream&
{
    static_assert( std::is_trivial_v<T> && std::is_standard_layout_v<T> );
    os.write( reinterpret_cast<char const*>( &v ), sizeof( T ) );
    return os;
};

auto readTrivial = []<typename T>( std::ifstream & is, T& v ) -> std::ifstream&
{
    static_assert( std::is_trivial_v<T> && std::is_standard_layout_v<T> );
    is.read( reinterpret_cast<char*>( &v ), sizeof( T ) );
    return is;
};

std::ofstream& operator<<( std::ofstream& os, Vertex const& v )
{
    return writeTrivial( os, v.point );
}

std::ifstream& operator>>( std::ifstream& is, Vertex& v )
{
    return readTrivial( is, v.point );
}

template <unsigned int M, unsigned int N>
std::ofstream& operator<<( std::ofstream& os, otk::Matrix<M, N> const& m )
{
    constexpr size_t size = M * N * sizeof( typename std::remove_pointer<decltype( m.getData() )>::type );
    os.write( reinterpret_cast<char const*>( m.getData() ), size );
    return os;
}

template <unsigned int M, unsigned int N>
std::ifstream& operator>>( std::ifstream& is, otk::Matrix<M, N>& m )
{
    constexpr size_t size = M * N * sizeof( typename std::remove_pointer<decltype( m.getData() )>::type );
    is.read( reinterpret_cast<char*>( m.getData() ), size );
    return is;
}

std::ofstream& operator<<( std::ofstream& os, otk::Aabb const& aabb )
{
    writeTrivial( os, aabb.m_min );
    writeTrivial( os, aabb.m_max );
    return os;
}

std::ifstream& operator>>( std::ifstream& is, otk::Aabb& aabb )
{
    readTrivial( is, aabb.m_min );
    readTrivial( is, aabb.m_max );
    return is;
}

std::ofstream& operator<<( std::ofstream& os, std::string const& s )
{
    if( writeTrivial( os, s.size() ); !s.empty() )
        os.write( s.data(), s.size() );
    return os;
}

std::ifstream& operator>>( std::ifstream& is, std::string& s )
{
    size_t size;
    if( readTrivial( is, size ); size > 0 )
    {
        s.resize( size );
        is.read( s.data(), size );
    }
    return is;
}

template <typename T>
std::ofstream& operator<<( std::ofstream& os, std::vector<T> const& v )
{
    if( writeTrivial( os, v.size() ); !v.empty() )
    {
        if constexpr( std::is_trivial_v<T> && std::is_standard_layout_v<T> )
            os.write( reinterpret_cast<char const*>( v.data() ), v.size() * sizeof( T ) );
        else
            for( size_t i = 0; i < v.size(); ++i )
                os << v[i];
    }
    return os;
}

template <typename T>
std::ifstream& operator>>( std::ifstream& is, std::vector<T>& v )
{
    v.clear();
    size_t size = 0;
    if( readTrivial( is, size ); size > 0 )
    {
        v.resize( size );
        if constexpr( std::is_trivial_v<T> && std::is_standard_layout_v<T> )
            is.read( reinterpret_cast<char*>( v.data() ), size * sizeof( T ) );
        else
            for( size_t i = 0; i < v.size(); ++i )
                is >> v[i];
    }
    return is;
}

std::ofstream& operator<<( std::ofstream& os, Shape::tag const& t )
{
    os << t.name;
    os << t.intargs;
    os << t.floatargs;
    os << t.stringargs;
    return os;
}

std::ifstream& operator>>( std::ifstream& is, Shape::tag& t )
{
    is >> t.name;
    is >> t.intargs;
    is >> t.floatargs;
    is >> t.stringargs;
    return is;
}

void Shape::writeShape( const std::string& objFile ) const
{
    using namespace std::chrono;
    fs::path cacheFile = fs::path( objFile ).replace_extension( ".bin" );

    if( std::ofstream os( cacheFile, std::ios::out | std::ofstream::binary ); os.is_open() )
    {
        system_clock::duration::rep objFileTimeStamp = ( fs::last_write_time( objFile ).time_since_epoch() + version ).count();

        writeTrivial( os, objFileTimeStamp );

        os << verts;
        os << normals;
        os << uvs;
        os << faceverts;
        os << faceuvs;
        os << facenormals;
        os << nvertsPerFace;
        os << tags;

        os << mtllib;
        os << mtlbind;

        os << aabb;
    }
}

bool Shape::readShape( const std::string& objFile )
{
    using namespace std::chrono;

    system_clock::duration::rep objFileTimeStamp = ( fs::last_write_time( objFile ).time_since_epoch() + version ).count();

    fs::path cacheFile = fs::path( objFile ).replace_extension( ".bin" );

    if( std::ifstream is( cacheFile, std::ios::in | std::ofstream::binary ); is.is_open() )
    {
        system_clock::duration::rep binFileTimStamp;

        readTrivial( is, binFileTimStamp );

        // if timestamp stored in .bin doesn't match the .obj's timestamp return false
        // i.e. read/load the .obj file instead
        if( binFileTimStamp == objFileTimeStamp )
        {
            is >> verts;
            is >> normals;
            is >> uvs;
            is >> faceverts;
            is >> faceuvs;
            is >> facenormals;
            is >> nvertsPerFace;
            is >> tags;

            is >> mtllib;
            is >> mtlbind;

            is >> aabb;
            
            return true;
        }
    }
    return false;
}


// Create shape with default geometry
std::unique_ptr<Shape> Shape::defaultShape()
{
    auto shape = std::unique_ptr<Shape>( new Shape );

    // clang-format on
    shape->verts = std::vector<Vertex>{
        {0.5f, 0.5f, 0.5f}, {1.5f, 0.5f, 0.5f}, {0.5f, 1.5f, 0.5f}, {1.5f, 1.5f, 0.5f}, {0.5f, 1.5f, -0.5f},
        {1.5f, 1.5f, -0.5f}, {0.5f, 0.5f, -0.5f}, {1.5f, 0.5f, -0.5f}, {-1.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f},
        {-1.5f, 1.5f, 0.5f}, {-0.5f, 1.5f, 0.5f}, {-1.5f, 1.5f, -0.5f}, {-0.5f, 1.5f, -0.5f}, {-1.5f, 0.5f, -0.5f},
        {-0.5f, 0.5f, -0.5f}, {0.5f, -1.5f, 0.5f}, {1.5f, -1.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {1.5f, -0.5f, 0.5f},
        {0.5f, -0.5f, -0.5f}, {1.5f, -0.5f, -0.5f}, {0.5f, -1.5f, -0.5f}, {1.5f, -1.5f, -0.5f}, {-1.5f, -1.5f, 0.5f},
        {-0.5f, -1.5f, 0.5f}, {-1.5f, -0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}, {-1.5f, -0.5f, -0.5f}, {-0.5f, -0.5f, -0.5f},
        {-1.5f, -1.5f, -0.5f}, {-0.5f, -1.5f, -0.5f}, {0.5f, -0.5f, 1.5f}, {1.5f, -0.5f, 1.5f}, {0.5f, 0.5f, 1.5f},
        {1.5f, 0.5f, 1.5f}, {-1.5f, -0.5f, 1.5f}, {-0.5f, -0.5f, 1.5f}, {-1.5f, 0.5f, 1.5f}, {-0.5f, 0.5f, 1.5f},
        {-0.5f, 1.5f, 1.5f}, {0.5f, 1.5f, 1.5f}, {1.5f, 1.5f, 1.5f}, {-1.5f, 1.5f, 1.5f}, {-0.5f, -1.5f, 1.5f},
        {0.5f, -1.5f, 1.5f}, {1.5f, -1.5f, 1.5f}, {-1.5f, -1.5f, 1.5f}, {0.5f, 0.5f, -1.5f}, {1.5f, 0.5f, -1.5f},
        {0.5f, -0.5f, -1.5f}, {1.5f, -0.5f, -1.5f}, {-1.5f, 0.5f, -1.5f}, {-0.5f, 0.5f, -1.5f}, {-1.5f, -0.5f, -1.5f},
        {-0.5f, -0.5f, -1.5f}, {-0.5f, 1.5f, -1.5f}, {0.5f, 1.5f, -1.5f}, {1.5f, 1.5f, -1.5f}, {-1.5f, 1.5f, -1.5f},
        {-0.5f, -1.5f, -1.5f}, {0.5f, -1.5f, -1.5f}, {1.5f, -1.5f, -1.5f}, {-1.5f, -1.5f, -1.5f}, 
    };

    shape->faceverts = std::vector<int>{ 2, 3, 5, 4, 6, 7, 1, 0, 1, 7, 5, 3, 6, 0, 2, 4, 10, 11, 13, 12, 14, 15, 9, 8,
        9, 15, 13, 11, 14, 8, 10, 12, 18, 19, 21, 20, 22, 23, 17, 16, 17, 23, 21, 19, 22, 16, 18, 20, 26, 27, 29, 28,
        30, 31, 25, 24, 25, 31, 29, 27, 30, 24, 26, 28, 32, 33, 35, 34, 0, 1, 19, 18, 33, 19, 1, 35, 18, 32, 34, 0, 36,
        37, 39, 38, 8, 9, 27, 26, 37, 27, 9, 39, 26, 36, 38, 8, 39, 34, 41, 40, 40, 41, 2, 11, 11, 2, 0, 9, 9, 0, 34,
        39, 34, 35, 42, 41, 41, 42, 3, 2, 35, 1, 3, 42, 38, 39, 40, 43, 43, 40, 11, 10, 8, 38, 43, 10, 44, 45, 32, 37,
        37, 32, 18, 27, 27, 18, 16, 25, 25, 16, 45, 44, 45, 46, 33, 32, 16, 17, 46, 45, 46, 17, 19, 33, 47, 44, 37, 36,
        24, 25, 44, 47, 24, 47, 36, 26, 20, 21, 7, 6, 48, 49, 51, 50, 21, 51, 49, 7, 50, 20, 6, 48, 28, 29, 15, 14, 52,
        53, 55, 54, 29, 55, 53, 15, 54, 28, 14, 52, 15, 6, 4, 13, 13, 4, 57, 56, 56, 57, 48, 53, 53, 48, 6, 15, 4, 5,
        58, 57, 57, 58, 49, 48, 7, 49, 58, 5, 12, 13, 56, 59, 59, 56, 53, 52, 52, 14, 12, 59, 31, 22, 20, 29, 29, 20,
        50, 55, 55, 50, 61, 60, 60, 61, 22, 31, 50, 51, 62, 61, 61, 62, 23, 22, 23, 62, 51, 21, 54, 55, 60, 63, 63, 60,
        31, 30, 63, 30, 28, 54, };

    // clang-format on

    shape->nvertsPerFace.resize( shape->faceverts.size()/4, 4 );
    
    // dummy texcoords
    shape->uvs.resize( shape->verts.size(), float2{0} );
    shape->faceuvs = shape->faceverts;

    shape->aabb = otk::Aabb( make_float3( -1.5f ), make_float3( 1.5f ) );

    return shape;
}

std::unique_ptr<Shape> Shape::loadObjFile( const fs::path& filepath, bool parseMaterials )
{
    if( filepath.empty() )
        return nullptr;

    constexpr bool isLeftHanded  = false;

    std::unique_ptr<Shape> shape = std::make_unique<Shape>();
    
    std::string filepathStr = filepath.lexically_normal().generic_string();

    // readShape queries the original objFile's last mod time
    // compares it to the one stored in the bin/cache file and loads the cache file if they match.  Otherwise, it
    // loads the mesh from the objFile and writes it in the bin/cache file along the last modified time of the objFile.
    if( !shape->readShape( filepathStr ) )
    {
        if (shape = parseObj( filepathStr.c_str(), kCatmark, isLeftHanded, parseMaterials); !shape)
            throw std::runtime_error( "OBJ parsing error: " + filepathStr );

        shape->writeShape( filepathStr );
    }

    shape->filepath = filepathStr;

    if( parseMaterials && !shape->mtllib.empty() )
    {
        // binary files do not encode materials, so we have to read them here since they
        // weren't read during OBJ parsring
        if( shape->mtls.empty() )
        {
            fs::path p = shape->mtllib;
            if( !fs::is_regular_file( p ) )
                p = shape->filepath.parent_path() / shape->mtllib;
            if( fs::is_regular_file( p ) )
                shape->mtls = parseMtllib( readAsciiFile( p.generic_string().c_str() ).c_str());
        }
        resolveUdims(*shape);    
    }

    return shape;
}

int Shape::findMaterial( char const* name )
{
    for( int i = 0; i < (int)mtls.size(); ++i )
        if( mtls[i]->name == name )
            return i;
    return -1;
}

bool Shape::material::hasUdims() const
{
    return ::hasUdims(*this);
}

std::vector<uint32_t> Shape::material::findUdims( fs::path const& basepath ) const
{
    std::vector<uint32_t> udims;

    auto search = [&basepath, &udims]( fs::path const& texpath ) {
        if( udims.empty() && !texpath.empty() )
        {
            if( std::string pattern = udimPath( texpath.filename().generic_string() ); !pattern.empty() )
            {
                fs::path dir = ( basepath / texpath.parent_path() );

                if( !fs::is_directory( dir ) )
                    return;

                for( auto const& entry : fs::directory_iterator( dir ) )
                {
                    int id;
                    if( sscanf( entry.path().filename().generic_string().c_str(), pattern.c_str(), &id ) == 1 )
                        udims.push_back( id );
                }

                // remove duplicates (ex. caused by dds version of texture)
                std::sort( udims.begin(), udims.end() );
                udims.erase( std::unique( udims.begin(), udims.end() ), udims.end() );

                if( udims.empty() )
                    throw std::runtime_error( std::string( "cannot find udims for: " ) + texpath.generic_string() );
            }
        }
    };

    std::apply( [&search]( auto const&... maps ) { ( search( maps ), ... ); }, materialMaps( *this ) );

    return udims;
}

std::unique_ptr<Shape::material> Shape::material::resolveUdimPaths( fs::path const& basepath, uint32_t udim, uint32_t udimMax ) const
{
    auto new_mtl = std::make_unique<Shape::material>( *this );

    auto resolve = [&basepath, &udim]( std::string& texpath ) {
        if( texpath.empty() )
            return;

        texpath = udimPath( texpath, std::to_string( udim ).c_str() );

#if defined( _WIN32 )
        // Workaround for windows 260 character path limit. See README.md
        fs::path normalizedPath{ basepath / texpath };
        normalizedPath = normalizedPath.lexically_normal().wstring();
        normalizedPath.make_preferred();
#else
        fs::path normalizedPath{ basepath / texpath };
#endif

        if( !fs::is_regular_file( normalizedPath ) )
            throw std::runtime_error( std::string( "cannot find udim: " ) + ( normalizedPath ).generic_string().c_str() );
    };

    std::apply( [&resolve]( auto&... maps ) { ( resolve( maps ), ... ); }, materialMaps( *new_mtl ) );

    new_mtl->udim = udim;
    new_mtl->udimMax = udimMax;

    return new_mtl;
}

std::vector<std::unique_ptr<Shape::material>> parseMtllib( const std::filesystem::path& path )
{
    return parseMtllib( readAsciiFile( path.string().c_str() ).c_str() );
}
