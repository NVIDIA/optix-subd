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
#include "optixSubdGUI.h"
#include "optixSubdApp.h"
#include "optixRenderer.h"
#include "statistics.h"

#include <profiler/profilerGUI.h>
#include <scene/scene.h>

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_internal.h>

#include <charconv>
#include <filesystem>
#include <string_view>


#ifndef _WIN32
    #include <unistd.h>
    #include <cstdio>
    #include <climits>
#else
    #include <Windows.h>
    #include <libloaderapi.h>
#endif // _WIN32

// clang-format on

namespace fs = std::filesystem;

// null-terminated array of filter strings

std::array<char const*, 3> UIData::formatFilters() const
{
    std::array<char const*, 3> filters = {".json", ".obj", nullptr };

    return filters;
}


void UIData::togglePlayPause()
{
    auto& state = timeLineEditorState;

    using enum otk::ImGuiRenderer::TimeLineEditorState::Playback;
    if( state.isPaused() )
    { 
        if( state.playCallback )
            state.playCallback(state);
        state.mode = Play; 
    }
    else 
    { 
        if( state.pauseCallback )
            state.pauseCallback(state);
        state.mode = Pause; 
    }
}

static inline bool isFolderFiltered( fs::path const& p, char const* const* filters )
{
    for( char const* const* filter = filters; *filter != nullptr; ++filter )
        if( p.generic_string().find( *filter ) != std::string::npos )
            return true;
    return false;
}

static inline bool isFormatFiltered( fs::path const& ext, char const* const* filters )
{
    for( char const* const* filter = filters; *filter != nullptr; ++filter )
        if( ext == *filter )
            return true;
    return false;
}

MediaAssetsMap findMediaAssets( fs::path const& mediapath, char const* const* format_filters )
{
    auto to_int = []( std::string_view str ) -> std::optional<int> {
        int value = 0;
        if( std::from_chars( str.data(), str.data() + str.size(), value ).ec == std::errc{} )
            return value;
        return {};
    };

    auto is_padded = []( std::string_view digits ) { return digits[0] == '0'; };

    auto get_sequence_str = []( std::string const& str ) -> std::string_view {
        if( auto last = std::find_if( str.rbegin(), str.rend(), ::isdigit ); last != str.rend() )
            if( auto first = std::find_if( last, str.rend(), []( char c ) { return !std::isdigit( c ); } ); first != str.rend() )
                return {first.base(), last.base()};
        return {};
    };


    if( !fs::is_directory( mediapath ) )
        return {};

    MediaAssetsMap assets;

    auto insert_asset = [&mediapath, &assets]( fs::path const& rp, std::string const& name = {} ) {
        auto [it, success] = assets.insert( {name.empty() ? rp.generic_string() : name, {}} );
        assert( success );
        it->second.name = it->first.c_str();
        return it;
    };

    auto opts = std::filesystem::directory_options::follow_directory_symlink;
    for( auto it = fs::recursive_directory_iterator( mediapath, opts ); it != fs::recursive_directory_iterator(); ++it )
    {
        if( !isFormatFiltered( it->path().extension(), format_filters ) )
            continue;

        fs::path rp = fs::relative( it->path(), mediapath ).lexically_normal();

        std::string stem = rp.stem().generic_string();

        if( std::string_view seq = get_sequence_str( stem ); !seq.empty() )
        {
            auto number = to_int( seq );
            if( !number )
                continue;

            std::string name = ( rp.parent_path() / std::string_view( stem.data(), seq.data() ) ).generic_string();

            auto it = assets.find( name );

            if( it == assets.end() )
            {
                it = insert_asset( rp, name );
            }

            if( is_padded( seq ) )
                it->second.padding = std::max( it->second.padding, (int)seq.size() );
            it->second.type = MediaAsset::Type::OBJ_SEQUENCE;
            it->second.growFrameRange( *number );
        }
        else
        {
            insert_asset( rp );
        }
    }

    for( auto it = assets.begin(); it != assets.end(); )
    {
        auto& asset = *it;

        if( asset.second.isSequence() )
        {
            char buf[1024];
            if( asset.second.frameRange.x < asset.second.frameRange.y )
            {
                std::snprintf( buf, std::size( buf ), "%s[%d-%d].obj", asset.first.c_str(), asset.second.frameRange.x,
                               asset.second.frameRange.y );
                asset.second.sequenceName = buf;

                if( asset.second.padding > 0 )
                    std::snprintf( buf, std::size( buf ), "%s%%0%dd.obj", asset.first.c_str(), asset.second.padding );
                else
                    std::snprintf( buf, std::size( buf ), "%s%%d.obj", asset.first.c_str() );
                asset.second.sequenceFormat = buf;

                asset.second.frameRate = 24.f;

                it = std::next( it );
            }
            else  // WAR for a single obj file whose name ends in a number being mistaken for an animation keyframe
            {
                std::snprintf( buf, std::size( buf ), "%s%d.obj", asset.first.c_str(), asset.second.frameRange.x );
                MediaAsset asset = { .frameRange = { 0, 0 }, .frameRate = 0.f };
                std::swap( assets[buf], asset );
                it = assets.erase( it );
            }
        }
        else
        {
            it->second.frameRange = { 0, 0 };
            it->second.frameRate = 0.f;
            it = std::next( it );
        }
    }

    return assets;
}

OptixSubdGUI::OptixSubdGUI( OptixSubdApp& app, UIData& ui )
    : m_app( app )
    , m_ui( ui )
{
    m_ui.showUI = m_app.getArgs().showUIonStart;
    m_ui.showOverlay = m_app.getArgs().showOverlayOnStart;

    const fs::path& mediaPath = m_app.getMediaPath();

    if( const fs::path& binaryPath = m_app.getBinaryPath(); !binaryPath.empty() )
        m_ui.iniFilepath = ( binaryPath / "imgui.ini" ).generic_string();

    const Scene::Attributes& attrs = m_app.getScene().getAttributes();

    setAnimationRange( attrs.frameRange, attrs.frameRate );

}

OptixSubdGUI::~OptixSubdGUI() = default;

void OptixSubdGUI::setAnimationRange( int2 frameRange, float frameRate )
{
    float startTime = 0.f;
    float endTime = 0.f;

    if( frameRange.y > frameRange.x )
    {
        startTime = float( frameRange.x ) / frameRate;
        endTime   = float( frameRange.y ) / frameRate;
    }
    else
    {
        assert( frameRate == 0.f );
        startTime = endTime = frameRate = 0.f;
    }
    auto& editor       = m_ui.timeLineEditorState;
    editor.frameRange  = frameRange;
    editor.startTime   = startTime;
    editor.endTime     = endTime;
    editor.currentTime = startTime;
    editor.frameRate   = frameRate;
}

void OptixSubdGUI::init( GLFWwindow* window )
{
    this->ImGuiRenderer::init( window );

    ImGui::SetCurrentContext( m_imgui );

    ImGuiIO& io = ImGui::GetIO();

    // custom ini file settings

    if( !m_ui.iniFilepath.empty() )
    {
        io.IniFilename = m_ui.iniFilepath.c_str();
    }

    io.IniSavingRate = 60.f;  // save every minute only or on quit

    // custom ini settings handler for the app - extend as needed

    static struct Settings
    {
        int  camera_roam_mode      = 0;

        bool display_stats         = false;
        bool display_help_window   = false;

        bool wantApply = false;
    } settings;

    ImGuiSettingsHandler ini_handler;
    ini_handler.TypeName = "ClusterBenchApp";
    ini_handler.TypeHash = ImHashStr( ini_handler.TypeName );
    ini_handler.UserData = this;

    ini_handler.ReadOpenFn = []( ImGuiContext*, ImGuiSettingsHandler*, const char* name ) -> void* {
        settings.wantApply = true;
        return &settings;
    };

    ini_handler.ApplyAllFn = []( ImGuiContext* ctx, ImGuiSettingsHandler* handler ) {
        if( settings.wantApply )
        {
            auto* gui = reinterpret_cast<OptixSubdGUI*>( handler->UserData );
            auto* app = &gui->m_app;

            gui->getProfilerGUI().m_displayGraphWindow = settings.display_stats;

            app->getTrackBall().setRoamMode( (bool)settings.camera_roam_mode );

            UIData& ui = gui->getUIData();

            ui.showHelpWindow = settings.display_help_window;

            const fs::path& mediaPath = app->getMediaPath();
            if( !mediaPath.empty() )
            {
                assert( ui.mediaAssets.empty() );
                auto format_filters = ui.formatFilters( );
                ui.mediaAssets = findMediaAssets( mediaPath, format_filters.data() );
            }

            if( std::string const& currentShape = app->getCurrentShape(); !currentShape.empty() )
            {
                // attempt to locate the asset in the media store ; the path may or may not
                // be relative to the media folder root path
                if( fs::path shapePath = fs::relative( currentShape, mediaPath ); !shapePath.empty() )
                ui.selectCurrentAsset( shapePath.generic_string() );
                else
                    ui.selectCurrentAsset( currentShape );
            }

            settings.wantApply = false;
        }
    };

    ini_handler.ReadLineFn = []( ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line ) {

        int roamMode = 0;
        if( std::sscanf( line, "RoamMode=%d", &roamMode ) == 1 )
            settings.camera_roam_mode = roamMode;

        int display_stats = false;
        if( std::sscanf( line, "DisplayStatistics=%d", &display_stats ) == 1 )
            settings.display_stats = display_stats != 0;

        int display_help_window = false;
        if( std::sscanf( line, "DisplayHelpWindow=%d", &display_help_window) == 1 )
            settings.display_help_window = display_help_window != 0;

    };

    ini_handler.WriteAllFn = []( ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf ) {

        auto* gui = reinterpret_cast<OptixSubdGUI*>( handler->UserData );
        auto* app = &gui->m_app;

        settings.camera_roam_mode      = app->getTrackBall().getRoamMode();
        settings.display_stats         = gui->getProfilerGUI().m_displayGraphWindow;
        settings.display_help_window   = gui->m_ui.showHelpWindow;

        buf->reserve( buf->size() + 2 );  // ballpark reserve
        buf->appendf( "[%s][%s]\n", handler->TypeName, "Settings" );
        buf->appendf( "RoamMode=%d\n", settings.camera_roam_mode );

        buf->appendf( "DisplayStatistics=%d\n", settings.display_stats );
        buf->appendf( "DisplayHelpWindow=%d\n", settings.display_help_window );
        buf->append( "\n" );
    };

    ImGui::AddSettingsHandler( &ini_handler );

}

void OptixSubdGUI::animate( float elapsedTimeSeconds )
{
    ImGuiRenderer::animate( elapsedTimeSeconds );

    if( m_ui.timeLineEditorState.isPlaying() )
        m_ui.timeLineEditorState.update( elapsedTimeSeconds );
}

void OptixSubdGUI::loadAsset( MediaAsset const& asset, std::string const& name )
{
    bool is_sequence = asset.isSequence();

    const fs::path& mediapath = m_app.getMediaPath();

    std::string shapePath =
        ( mediapath / ( is_sequence ? fs::path( asset.sequenceFormat ) : fs::path( name ) ) ).generic_string();

    m_app.loadScene( shapePath, mediapath.generic_string(), asset.frameRange);

    const Scene::Attributes& attrs = m_app.getScene().getAttributes();

    setAnimationRange( attrs.frameRange, attrs.frameRate );

    m_ui.showRecommendationWindow = true;

    m_ui.currentAsset = &asset;
}

void OptixSubdGUI::buildUI_internal( int2 window_size )
{
    ImGuiIO& io = ImGui::GetIO();

    int width = window_size.x, height = window_size.y;

    float profiler_width = m_profiler.controllerWindow.size.x;
    float timeline_width = float(width) - 30.f - profiler_width;

    ImVec2 itemSize = ImGui::GetItemRectSize();

    auto& renderer = m_app.getOptixRenderer();

    UIData& ui = getUIData();

    // Settings
    ImGui::SetNextWindowPos( ImVec2( 10.f, 10.f ) );
    ImGui::SetNextWindowBgAlpha( .65f );
    ImGui::Begin( "Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize );

    ImGui::PushFont(m_iconicFont);
    if( ImGui::Button( (char const*)(u8"\ue02c" "## screenshot"), { 0.f, itemSize.y } ) )
    {
        renderer.saveScreenshot();
    }
    ImGui::PopFont();
    if( ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f )
        ImGui::SetTooltip("Capture a screenshot.");

    ImGui::SameLine();
    if( ImGui::Checkbox( "VSync", &m_ui.vsync ) )
        glfwSwapInterval( m_ui.vsync );

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    fs::path mediapath = m_app.getMediaPath();

    // media folder
    bool objFilesNeedUpdate = false;
    {
        ImGui::PushFont( m_iconicFont );
        if( ImGui::Button( (char const*)( u8"\ue06b" "## media path" ), { 0.f, itemSize.y } ) )
        {
            std::string folderpath = mediapath.generic_string();
            if( ImGuiRenderer::folderDialog( folderpath ) )
            {
                mediapath = fs::path( folderpath ).lexically_normal();
                objFilesNeedUpdate = true;
            }
        }
        ImGui::PopFont();
        ImGui::SameLine();

        char buf[1024] = { 0 };
        std::strncpy( buf, mediapath.generic_string().c_str(), std::size( buf ) );
        if( ImGui::InputText( "Data Folder", buf, std::size( buf ), ImGuiInputTextFlags_EnterReturnsTrue ) )
        {
            mediapath = buf;
            objFilesNeedUpdate = true;
        }
        if( ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f && !mediapath.empty() )
            ImGui::SetTooltip( "%s", mediapath.generic_string().c_str());
    }

    m_app.setMediaPath( mediapath );

    if( objFilesNeedUpdate || m_ui.mediaAssets.empty() )
    {
        std::string assetName = m_ui.currentAsset ? m_ui.currentAsset->getName() : std::string();

        auto format_filters = m_ui.formatFilters( );
    
        m_ui.mediaAssets = findMediaAssets( mediapath, format_filters.data() );

        m_ui.selectCurrentAsset( assetName );
    }

    char const* currentAssetName = m_ui.currentAsset ? m_ui.currentAsset->getName() : nullptr;
    if( ImGui::BeginCombo( "Scene",  currentAssetName, ImGuiComboFlags_HeightLargest ) )
    {
        for( auto const& it : m_ui.mediaAssets )
        {
            auto& asset = it.second;

            bool is_sequence = asset.isSequence();
            
            std::string const& name = is_sequence ? asset.sequenceName : it.first;          
            
            bool is_selected = currentAssetName && (name == currentAssetName);

            if( ImGui::Selectable( name.c_str(), is_selected ) )
            {
                loadAsset( asset, name );
            }
            if( is_selected )
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    if( ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f && m_ui.currentAsset && m_ui.currentAsset->name )
        ImGui::SetTooltip( "%s", m_ui.currentAsset->name );

    ImGui::Spacing();

    buildUI_cameraSettings();

    ImGui::Spacing();

    ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( .3f, .4f, .35f, 1.f ) );
    if( ImGui::CollapsingHeader( "Rendering", ImGuiTreeNodeFlags_DefaultOpen ) )
    {

        bool wireframe = renderer.getWireframe();
        if( ImGui::Checkbox( "Triangle Edges", &wireframe ) )
        {
            renderer.setWireframe( wireframe );
        }
        if (ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Display tessellated triangle wireframe");
        ImGui::SameLine();

        bool wireframeGL = m_app.getGLWireframeEnabled();
        if( ImGui::Checkbox( "Subd Cage", &wireframeGL ) )
        {
            m_app.setGLWireframeEnabled( wireframeGL );
        }
        if (ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f)
            ImGui::SetTooltip("Display subd mesh wireframe");

        ImGui::SameLine();

        
        ImGui::PushItemWidth(65);

        {
            int n = static_cast<int>( std::sqrt( renderer.getAOSamples() ) - 1 );
            if( ImGui::Combo( "AO samples", &n, " 1x\0 4x\0 9x\0 16x\0 25x\0 36x\0 49x\0 64x\0" ) )
            {
                renderer.setAOSamples( (n+1) * (n+1) );
            }
        }
        ImGui::PopItemWidth();

        int colorMode = (int)renderer.getColorMode();
        if( ImGui::Combo( "Color Mode", &colorMode,
                          "Base Color\0Triangle\0Surface Normal\0Tex Coord\0Material\0"
                          "Cluster\0Patch UV\0Triangle Area\0" ) )
        {
            renderer.setColorMode( ColorMode( colorMode ) );
        }

        int channel = (int)renderer.getDisplayChannel();
        if( ImGui::Combo( "Display Channel", &channel, "Albedo\0Normals\0MotionVecs\0Color\0Depth\0Hires Depth\0Accum/Denoised\0" ) )
        {
            renderer.setDisplayChannel( GBuffer::Channel( channel ) );
        }

        float3 missColor = renderer.getMissColor();
        if( ImGui::ColorEdit3( "Miss Color", &missColor.x, ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel ) )
        {
            renderer.setMissColor( missColor );
        }

        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();

    }
    ImGui::PopStyleColor();
    ImGui::Spacing();

    ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( .4f, .3f, .35f, 1.f ) );
    if( ImGui::CollapsingHeader( "Tessellation", ImGuiTreeNodeFlags_DefaultOpen ) )
    {
        int clusterPattern = static_cast<int>( m_app.getClusterTessellationPattern() );
        float comboBoxWidth = ImGui::GetTextLineHeightWithSpacing() + ImGui::CalcTextSize( "Slanted  " ).x + ImGui::GetStyle().FramePadding.x * 2.0f;
        ImGui::SetNextItemWidth( comboBoxWidth );
        if( ImGui::Combo( "Tess Pattern", &clusterPattern, "Regular\0Slanted\0" ) )
        {
            m_app.setClusterTessellationPattern( ClusterPattern( clusterPattern ) );
        }
        if( ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f )
            ImGui::SetTooltip( "Toggles the 'slanted' grid pattern on. 'Slanted' grids\n"
                "allow for a smoother transition when the number of edge segments on\n" 
                "opposite sides of a quad don't match.\n\n" );

        bool adaptive = m_app.getAdaptiveTessellation();
        if( ImGui::Checkbox( "Adaptive", &adaptive ) )
        {
            m_app.setAdaptiveTessellation( adaptive );
        }
        if (ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f)
            ImGui::SetTooltip( "Keep tessellation camera in sync. Uncheck to lock the camera." );

        ImGui::SameLine();
        bool enableVertexNormals = m_app.getVertexNormalsEnabled();
        if( ImGui::Checkbox( "Vertex Normals", &enableVertexNormals ) )
        {
            m_app.setVertexNormalsEnabled( enableVertexNormals );
        }
        if (ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f)
            ImGui::SetTooltip( "Shade using limit vertex normals from tessellation.\n"
                               "Uncheck to use facet normals.\n\n" );

        float fineTessRate = m_app.getFineTessellationRate();
        if( ImGui::SliderFloat( "Fine Tessellation Rate", &fineTessRate, 0.001f, 2.f ) )
        {
            if ( fineTessRate > 0.0f )
            {
                m_app.setFineTessellationRate( fineTessRate );
            }
        }
        if (ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f)
            ImGui::SetTooltip( "Tessellation rate for geometry considered to be visible" );

        float coarseTessRate = m_app.getCoarseTessellationRate();
        if( ImGui::SliderFloat( "Coarse Tessellation Rate", &coarseTessRate, 0.001f, 2.f ) )
        {
            if ( coarseTessRate > 0.0f )
                m_app.setCoarseTessellationRate( coarseTessRate );
        }
        if( ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f )
            ImGui::SetTooltip( "Tessellation rate for geometry considered to be not visible" );
        
        bool enableFrustumVisibility = m_app.getFrustumVisibilityEnabled();
        if( ImGui::Checkbox( "Frustum", &enableFrustumVisibility ) )
            m_app.setFrustumVisibilityEnabled( enableFrustumVisibility );
        if( ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f )
            ImGui::SetTooltip( "Use frustum test to adjust tessellation rate" );
        
        ImGui::SameLine();
        bool enableBackfaceVisibility = m_app.getBackfaceVisibilityEnabled();
        if( ImGui::Checkbox( "Backface", &enableBackfaceVisibility) )
            m_app.setBackfaceVisibilityEnabled( enableBackfaceVisibility );
        if( ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f )
            ImGui::SetTooltip( "Use backface test to adjust tessellation rate.\n"
                               "Turn this off to fix errors with single-sided geometry.\n\n" );

    }
    ImGui::PopStyleColor();

    ImGui::Spacing();

    ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( .1f, .3f, .85f, 1.f ) );
    if( ImGui::CollapsingHeader( "Displacement", ImGuiTreeNodeFlags_None ) )
    {
        // displacement textures

        float displacementScale = m_app.getDisplacementScale();
        if( ImGui::SliderFloat( "Displacement Scale", &displacementScale, 0.0f, 3.0f ) )
        {
            m_app.setDisplacementScale( displacementScale );
        }

        float displacementBias = m_app.getDisplacementBias();
        if( ImGui::SliderFloat( "Displacement Bias", &displacementBias, -1.0f, 1.0f ) )
        {
            m_app.setDisplacementBias(displacementBias);
        }

        // Note: displacement filter scale and mip bias not currently exposed in UI

    }
    ImGui::PopStyleColor();

    ImGui::Spacing();

    ImGui::End();

    // profiler

    m_profiler.fps = (int)(1000.f / (float)m_app.getCPUFrameTime());

    m_profiler.ntris = computeTriangleCount();

    float fontScale = io.FontGlobalScale;
    m_profiler.controllerWindow = {
        .pos = ImVec2( float( width ) - 10.f, float( height ) - 10.f ),
        .pivot = ImVec2( 1.f, 1.f ),
        .size = ImVec2( 115*fontScale, 0 )
    };

    m_profiler.profilingWindow = {
        .pos = ImVec2( float( width ) - 10.f, 10.f ),
        .cond = ImGuiCond_FirstUseEver,
        .pivot = ImVec2( 1.f, 0.f ),
        .size = ImVec2( 800.f, 450.f ),
    };

    m_profiler.buildUI<stats::FrameSamplers, stats::ClusterAccelSamplers, stats::EvaluatorSamplers, stats::MemUsageSamplers>(
        *this, stats::frameSamplers, stats::clusterAccelSamplers, stats::evaluatorSamplers, stats::memUsageSamplers );

    if( m_ui.showTimeLineEditor )
        buildUI_timeline( { width, height }, timeline_width );

}

void OptixSubdGUI::buildUI()
{
    int2 size;
    glfwGetWindowSize( m_window, &size.x, &size.y );

    if( m_ui.showOverlay )
        buildUI_overlay( size );

    if( !m_ui.showUI )
        return;

    // meat of the UI
    buildUI_internal( size );

    if( m_ui.showHelpWindow )
        buildUI_help( size, &m_ui.showHelpWindow );
}

// clang-format off

size_t OptixSubdGUI::computeTriangleCount() const
{
    size_t subdTriCount = stats::clusterAccelSamplers.numTriangles.latest;
    return  subdTriCount;
}

void OptixSubdGUI::buildUI_cameraSettings()
{    
     ImGui::PushStyleColor( ImGuiCol_Header, ImVec4( .2f, .2f, .55f, 1.f ) );
     if( ImGui::CollapsingHeader( "Camera", ImGuiTreeNodeFlags_None ) )
     {
        int cameraMode = (int)m_app.getTrackBall().getRoamMode();
        if( ImGui::Combo( "Camera Control", &cameraMode, "Orbit\0Roam\0" ) )
        {
            m_app.getTrackBall().setRoamMode( (bool)cameraMode );
        }
        if( ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f )
            ImGui::SetTooltip(
                "- Roam mode: the camera operates in first-person 'wasd' navigation mode.\n"
                "  Third-person (orbit) can be accessed by holding the 'alt' key.\n"
                "- Orbit mode: the camera only operates in third-person mode, and no longer\n"
                "  requires holding the 'alt' key. First person navigation is disabled." );
            float fovY = m_app.getCamera().getFovY();
            if( ImGui::SliderFloat( "Camera FoV", &fovY, 10.f, 60.f, "%.2f" ) )
            {
                m_app.getCamera().setFovY( fovY );
                m_app.unlockCamera();
            }
            if( ImGui::IsItemHovered() && m_imgui->HoveredIdTimer > .5f )
                ImGui::SetTooltip( "Camera vertical FoV (in mm()" );
            
     }
     ImGui::PopStyleColor();
}

void OptixSubdGUI::buildUI_help( int2 window_size, bool* display_help_window )
{
    int width = window_size.x, height = window_size.y;

    ImGui::SetNextWindowPos( ImVec2( float( width ) * .25f, 10.f ), ImGuiCond_FirstUseEver );       
    ImGui::SetNextWindowBgAlpha( .65f );
    if( ImGui::Begin( "Controls", display_help_window, ImGuiWindowFlags_None ) )
    {
        auto buildRow = []( const char* key, const char* function ) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex( 0 );
            ImGui::Text( "%s", key );
            ImGui::TableSetColumnIndex( 1 );
            ImGui::Text( "%s", function );
        };

        {
            // camera controls
            ImGui::Spacing();
            ImGui::Text( "Camera Controls" );
            ImGui::Spacing();

            ImGui::BeginTable( "##Input1", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX );

            ImGui::TableSetupColumn( "Input", ImGuiTableColumnFlags_WidthFixed, 100 );
            ImGui::TableSetupColumn( "Function", ImGuiTableColumnFlags_WidthFixed, 300 );
            ImGui::TableHeadersRow();

            buildRow( "F", "reset the camera" );
            buildRow( "Ctrl + LMB", "latch focus point to object" );
            buildRow( "Alt + LMB", "orbit" );
            buildRow( "Alt + MMB", "pan" );
            buildRow( "Alt + RMB", "dolly" );

            ImGui::TableNextRow( );
            ImGui::TableSetColumnIndex( 0 );
            ImGui::TableHeader( "FPS (roam) mode" );

            buildRow( "LMB", "look up/down/left/right (roam mode only)" );
            buildRow( "W", "move forward (roam mode only)" );
            buildRow( "A", "strafe left (roam mode only)" );
            buildRow( "S", "move backward (roam mode only)" );
            buildRow( "D", "strafe right (roam mode only)" );

            ImGui::EndTable();

            ImGui::Spacing();
            ImGui::Separator();
        }
        {
            // keyboard controlS
            ImGui::Spacing();
            ImGui::Text( "Keyboard Controls" );
            ImGui::Spacing();

            ImGui::BeginTable("##Input2", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_NoHostExtendX);

            ImGui::TableSetupColumn( "Hotkey", ImGuiTableColumnFlags_WidthFixed, 100 );
            ImGui::TableSetupColumn( "Function", ImGuiTableColumnFlags_WidthFixed, 300 );
            ImGui::TableHeadersRow();

            buildRow( "Ctrl + Q", "exit the application" );

            buildRow( "~", "toggle UI" );
            buildRow( "`", "toggle overlay" );

            buildRow( "C", "print the camera string to console" );
            buildRow( "H", "print usage/help to console" );
            buildRow( "Ctrl + H", "toggle this help window" );

            ImGui::EndTable();
        }
    }
    ImGui::End();
}


void OptixSubdGUI::buildUI_timeline( int2 window_size, float timeline_width )
{
    int width = window_size.x, height = window_size.y;

        // attach to the left of the profiler UI and scale to fit screen
        float tw = timeline_width;
        ImGui::SetNextWindowPos( ImVec2( tw + 10.f, float( height ) - 10.f ), 0, ImVec2( 1.f, 1.f ) );
        ImGui::SetNextWindowSize( ImVec2( tw, 0.f ) );
        ImGui::SetNextWindowBgAlpha( .65f );
        if( ImGui::Begin( "TimeLine Editor", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoTitleBar ) )
        {
            if( ImGuiRenderer::buildTimeLineEditor( m_ui.timeLineEditorState, float2( tw, 0.f ) ) )
            {
            }
        }
        ImGui::End();
}

void OptixSubdGUI::buildUI_overlay( int2 canvas_size )
{
    int width = canvas_size.x, height = canvas_size.y;

    ImGui::PushFont( m_nvidiaBldFont );
    
    const float fontScale = .75f / ImGui::GetIO().FontGlobalScale;
    const ImVec2 charSize = ImGui::CalcTextSize("A");
    const ImVec2 fontSize = ImVec2( fontScale * charSize.x, fontScale * charSize.y );
    
    auto const& renderer = m_app.getOptixRenderer();
    const uint8_t numLines = 4;

    const float col0_width = 8 * fontSize.x;
    const float col1_width = 7 * fontSize.x;

    ImVec2 windowOfs = ImVec2( 25.f, 25.f );
    ImVec2 windowSize( col0_width + col1_width + 50.f, numLines * ( fontSize.y + 5.f ) );
    ImVec2 windowPos = ImVec2( windowOfs.x, ( float( height ) - windowSize.y - windowOfs.y) );

    ImGui::SetNextWindowPos( windowPos, ImGuiCond_Always );
    ImGui::SetNextWindowSize( windowSize, ImGuiCond_Always);
    ImGui::SetNextWindowBgAlpha( 0.f );
    ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.f );

    ImGui::Begin( "overlay", 0, ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoScrollbar );

    ImGui::SetWindowFontScale( fontScale );

    ImGui::PushStyleColor( ImGuiCol_Text, ImVec4( 1.f, 1.f, 1.f, 1.f ) );

    char buf[50];

    ImGui::BeginTable( "overlay_table", 2, ImGuiTableFlags_NoHostExtendX );
    {       
        ImGui::TableSetupColumn( "overlay_Name", ImGuiTableColumnFlags_WidthFixed, col0_width );
        ImGui::TableSetupColumn( "overlay_Value", ImGuiTableColumnFlags_WidthFixed, col1_width );

        auto displayRow = [&col1_width]( std::string_view label, std::string_view value ) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex( 0 );
            ImGui::TextUnformatted( &label.front(), &label.back() + 1 );
            ImGui::TableSetColumnIndex( 1 );
            // text right-adjust
            float textPos = ImGui::GetCursorPosX() + col1_width - ImGui::CalcTextSize( value.data() ).x;
            ImGui::SetCursorPosX( textPos );
            ImGui::TextUnformatted( &value.front(), &value.back() + 1 );
        };

        // Display Mode
        {
            static std::array<std::pair<ColorMode, const char*>, 7> modeNames = {
                std::pair{ ColorMode::BASE_COLOR, "Color" },
                std::pair{ ColorMode::COLOR_BY_TRIANGLE, "Triangle" },
                std::pair{ ColorMode::COLOR_BY_NORMAL, "Normal" },
                std::pair{ ColorMode::COLOR_BY_TEXCOORD, "Tex Coord" },
                std::pair{ ColorMode::COLOR_BY_MATERIAL, "Material ID" },
                std::pair{ ColorMode::COLOR_BY_MICROTRI_AREA, "Area" },
                std::pair{ ColorMode::COLOR_BY_CLUSTER_ID, "Cluster ID" },
            };

            ColorMode colorMode = renderer.getColorMode();

            auto it = std::find_if( modeNames.begin(), modeNames.end(), 
                [&colorMode]( auto modeName ) { return (int)modeName.first == (int)colorMode; } );

            displayRow( "Display", it != modeNames.end() ? it->second : "----" );
        }

        // Control mesh
        size_t surfaceCount = stats::evaluatorSamplers.surfaceCountTotal;
        ProfilerGUI::humanFormatter( static_cast<double>(surfaceCount), buf, std::size(buf) );
        displayRow( "SubD Faces", buf );

        // Triangles
        size_t ntris = computeTriangleCount();
        ProfilerGUI::humanFormatter( static_cast<double>( ntris ), buf, std::size( buf ) );
        displayRow( "Triangles", buf );

        // FPS
        int fps = (int)( 1000.f / stats::frameSamplers.cpuFrameTime.runningAverage() );
        snprintf( buf, std::size( buf ), "%d", fps );
        displayRow( "FPS", buf );

        ImGui::TableNextRow();
    }
    ImGui::EndTable();

    ImGui::PopStyleColor();

    ImGui::End();
    ImGui::PopStyleVar();
    ImGui::PopFont();
}

// clang-format on
