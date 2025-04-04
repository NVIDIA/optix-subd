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

#include "optixSubdApp.h"
#include "optixSubdGUI.h"
#include "optixRenderer.h"

#include <scene/scene.h>

#include <OptiXToolkit/Gui/GLDisplay.h>

#include <GLFW/glfw3.h>

#include <cassert>
#include <charconv>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

static const std::map<ColorMode, std::string> ColorModeNames = {
    { ColorMode::BASE_COLOR,               "BASE_COLOR" },
    { ColorMode::COLOR_BY_TRIANGLE,        "COLOR_BY_TRIANGLE"},
    { ColorMode::COLOR_BY_NORMAL,          "COLOR_BY_NORMAL" },
    { ColorMode::COLOR_BY_TEXCOORD,        "COLOR_BY_TEXCOORD" },
    { ColorMode::COLOR_BY_MATERIAL,        "COLOR_BY_MATERIAL" },
    { ColorMode::COLOR_BY_CLUSTER_UV,      "COLOR_BY_CLUSTER_UV" },
    { ColorMode::COLOR_BY_CLUSTER_ID,      "COLOR_BY_CLUSTER_ID" },
    { ColorMode::COLOR_BY_MICROTRI_AREA,   "COLOR_BY_MICROTRI_AREA" },
};

// clang-format on

void fatalError( const char* fmt, ... )
{
    static std::mutex mtx;

    std::lock_guard<std::mutex> lockGuard( mtx );

    char buf[4096];
    va_list args;
    va_start( args, fmt );
    vsnprintf( buf, std::size( buf ), fmt, args );

#if _WIN32
    OutputDebugStringA( buf );
    OutputDebugStringA( "\n" );

    MessageBoxA( 0, buf, "Error", MB_ICONERROR );
#else
    fprintf(stderr, "Error: %s\n", buf);
#endif

    abort();
}

static constexpr size_t const gigabyte = 1024 * 1024 * 1024;

static constexpr size_t const min_memory_gb = 10;

static bool hasCapableDevice()
{
    int count = 0;
    CUDA_CHECK( cudaGetDeviceCount( &count ) );

    OTK_REQUIRE_MSG( count > 0, "No compute-capable device found." );

    for( int i = 0; i < count; ++i )
    {
        cudaDeviceProp props{};
        CUDA_CHECK( cudaGetDeviceProperties( &props, i ) );

        // Modern proto boards should all identify with the same default vendor string,
        // and Geforce & Quadro SKUs have an 'RTX' in their vendor ID
        static constexpr char const protoName[] = "NVIDIA Graphics Device";

        if( !( std::strncmp( props.name, protoName, std::size( protoName ) ) == 0 ) &&
            ( std::string( props.name ).find( "RTX" ) == std::string::npos ) )
            continue;

        // Google says Ampere compute is 8.6
        if( ( props.major < 8 ) || ( ( props.major == 8 ) && ( props.minor < 6 ) ) )
            continue;

        if (props.totalGlobalMem < ( min_memory_gb * gigabyte ) )
            continue;

        return true;
    }
    return false;
}

void printInteractiveHelp();

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

bool minimized = false;

static void charModsCallback( GLFWwindow* window, unsigned int unicode, int mods )
{
    auto* gui = reinterpret_cast<OptixSubdGUI*>(glfwGetWindowUserPointer(window));
    if (gui->keyboardCharInput(unicode, mods))
        return;
}

static void scrollCallback( GLFWwindow* window, double xscroll, double yscroll )
{
    auto* gui = reinterpret_cast<OptixSubdGUI*>(glfwGetWindowUserPointer(window));

    if (gui->mouseScrollUpdate(xscroll, yscroll))
        return;

    auto& app = gui->getApp();

    if( yscroll != 0 )
        app.getTrackBall().mouseWheelUpdate((int)yscroll);
}

static void mouseButtonCallback( GLFWwindow* window, int button, int action, int mods )
{
    auto* gui = reinterpret_cast<OptixSubdGUI*>(glfwGetWindowUserPointer(window));

    if (gui->mouseButtonUpdate(button, action, mods))
        return;

    auto& app = gui->getApp();

    otk::Trackball& trackball = app.getTrackBall();
    otk::Camera&    camera    = app.getCamera();

    trackball.mouseButtonUpdate( button, action, mods );

    if( action == GLFW_PRESS )
    {
        double xpos, ypos;
        glfwGetCursorPos( window, &xpos, &ypos );

        uint2 pos = { static_cast<uint32_t>( xpos ), static_cast<uint32_t>( ypos ) };

        if( mods & GLFW_MOD_CONTROL )
        {
            if( auto pick = app.getOptixRenderer().pick( pos ); pick->w > 0.f && std::isfinite( pick->w ) )
            {
                camera.setLookat( make_float3( *pick ) );
                const float d = length( camera.getLookat() - camera.getEye() );
                trackball.setMoveSpeed( d );
                trackball.reinitOrientationFromCamera();
            }
        }

    }
}

static void cursorPosCallback( GLFWwindow* window, double xpos, double ypos )
{
    auto* gui = reinterpret_cast<OptixSubdGUI*>(glfwGetWindowUserPointer(window));

    if (gui->mousePosUpdate(xpos, ypos))
        return;

    auto& app = gui->getApp();

    int2 pos    = { static_cast<int>( xpos ), static_cast<int>( ypos ) };
    int2 canvas = app.getOutputBufferTargetSize();

    app.getTrackBall().mouseTrackingUpdate( pos, canvas );
}

static void windowSizeCallback( GLFWwindow* window, int32_t width, int32_t height )
{
    // Keep rendering at the current resolution when the window is minimized.
    if( minimized )
        return;

    auto& app = reinterpret_cast<OptixSubdGUI*>(glfwGetWindowUserPointer(window))->getApp();

    app.setOutputBufferTargetSize( make_uint2( std::max( 1, width ), std::max( 1, height ) ) );
}

static void windowIconifyCallback( GLFWwindow* window, int32_t iconified )
{
    minimized = ( iconified > 0 );
}

static void toggleFullscreen( GLFWwindow* window )
{
    if( !window )
        return;

    static int width = 1920, height = 1080, xpos = 150, ypos = 150;

    if( GLFWmonitor* monitor = glfwGetWindowMonitor( window ) )
    {
        // fullscreen -> windowed
        glfwSetWindowMonitor( window, NULL, xpos, ypos, width, height, 0 );
    }
    else
    {
        // windowed -> fullscreen
        glfwGetWindowSize( window, &width, &height );
        glfwGetWindowPos( window, &xpos, &ypos );

        int monitorCount = 0;
        GLFWmonitor** monitors = glfwGetMonitors( &monitorCount );
        if( monitorCount == 0 )
            return;

        const GLFWvidmode* mode = glfwGetVideoMode( monitors[0] );

        glfwSetWindowMonitor( window, monitors[0], 0, 0, mode->width, mode->height, mode->refreshRate );
    }
}

static void keyCallback( GLFWwindow* window, int32_t key, int32_t code, int32_t action, int32_t mods )
{
    auto* gui = reinterpret_cast<OptixSubdGUI*>(glfwGetWindowUserPointer(window));

    if (gui->keyboardUpdate(key, code, action, mods))
        return;

    auto& app = gui->getApp();

    app.getTrackBall().keyboardUpdate( key, code, action, mods );

    otk::Camera& camera = app.getCamera();

    OptixRenderer& renderer = app.getOptixRenderer();

    if( action == GLFW_PRESS )
    {
        switch( key )
        {
            case GLFW_KEY_Q:
                if( ( mods & GLFW_MOD_CONTROL ) == 0 )
                    break;

            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose( window, true );
                break;

            case GLFW_KEY_ENTER: {
                if( ( mods & GLFW_MOD_ALT ) == GLFW_MOD_ALT )
                    toggleFullscreen( window );
                } break;

            case GLFW_KEY_SPACE: {
                gui->getUIData().togglePlayPause();
            } break;

            case GLFW_KEY_F:
                app.resetCamera();
                break;
            
            case GLFW_KEY_GRAVE_ACCENT:
                if( ( mods & GLFW_MOD_SHIFT ) == GLFW_MOD_SHIFT )
                    gui->getUIData().showOverlay = !gui->getUIData().showOverlay;
                else
                    gui->getUIData().showUI = !gui->getUIData().showUI;
                break;

            case GLFW_KEY_H:
                    if( ( mods & GLFW_MOD_CONTROL ) == GLFW_MOD_CONTROL )
                        gui->getUIData().showHelpWindow = !gui->getUIData().showHelpWindow;
                    else
                        printInteractiveHelp();                
                break;

            case GLFW_KEY_C: {
                std::string cli = camera.getCliArgs();
                std::string pr = camera.getPositionRotation();
                fprintf(stdout, "CLI:%s JSON: %s\n", cli.c_str(), pr.c_str());
                fflush(stdout);
                glfwSetClipboardString(window, (cli + " " + pr).c_str());
                break;
            }

        }
    }
}


void displaySubframe( otk::CUDAOutputBuffer<uchar4>& output_buffer, otk::GLDisplay& gl_display, GLFWwindow* window )
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize( window, &framebuf_res_x, &framebuf_res_y );
    gl_display.display( output_buffer.width(), output_buffer.height(), framebuf_res_x, framebuf_res_y, output_buffer.getPBO() );
}

void printInteractiveHelp()
{
    // clang-format off
    std::cerr 
        << "Interactive controls:\n"
        << "Camera     : WASD+left mouse button (roam mode)\n"
        << "           : ALT+left/middle/right mouse (orbit mode)\n"
        << "           : CTRL+left mouse to center & orbit clicked point\n"
        << "ESC/CTRL+q : quit\n"
        << "`          : toggle GUI\n"
        << "h          : help (print this message)\n"
        << "CTRL+h     : toggle help window in UI\n"
        << "f          : reset camera\n"
        << "c          : print camera (for use with -p flag or json scene files)\n"
        << std::endl;
    // clang-format on
}

void exitHandler()
{
    Profiler::get().terminate();
}

int main( int argc, char* argv[] )
{
    std::atexit( exitHandler );

    try
    {
        UIData    uiData;

        auto app = std::make_unique<OptixSubdApp>( argc, argv );
        bool interactive = app->interactiveMode();
        bool glInterop = true; //app->GLinterop();

        if( interactive )
        {
            auto gui = std::make_unique<OptixSubdGUI>( *app, uiData );

            int2 size = app->getOutputBufferTargetSize();

            extern char const* windowName;
            GLFWwindow* window = otk::initGLFW( windowName, size.x, size.y);
            OTK_ASSERT_MSG( window, "GLFW failed to open a window");

            otk::initGL();
            otk::GLDisplay glDisplay;

            app->setupGL();
            
            glfwSetWindowUserPointer( window, gui.get() );

            glfwSetMouseButtonCallback( window, mouseButtonCallback );
            glfwSetCursorPosCallback( window, cursorPosCallback );
            glfwSetWindowSizeCallback( window, windowSizeCallback );
            glfwSetWindowIconifyCallback( window, windowIconifyCallback );
            glfwSetKeyCallback( window, keyCallback );
            glfwSetCharModsCallback( window, charModsCallback );
            glfwSetScrollCallback( window, scrollCallback );
            
            // initialize imgui after the callbacks have been set
            gui->init( window );

            printInteractiveHelp();

            auto& output_buffer = app->getOptixRenderer().getOutputBuffer();

            auto const& animState = uiData.timeLineEditorState;

            do
            {
                glfwPollEvents();

                app->renderInteractiveSubframe(animState.animationTime(), animState.frameRate);

                displaySubframe( output_buffer, glDisplay, window );

                app->drawGL();

                gui->animate(app->getCPUFrameTime() / 1000.f);

                gui->render();

                glfwSwapBuffers( window );

            } while( !glfwWindowShouldClose( window ) );

            CUDA_SYNC_CHECK();

            // cleanup
            gui.reset();
        }
        else
        {
            if( glInterop )
            {
                otk::initGLFW();
                otk::initGL();
            }

            app->renderBatchSubframes();
        }

        // cleanup

        app.reset();

        Profiler::get().terminate();

        if( !interactive && glInterop )
            glfwTerminate();
    }
    catch( std::exception& e )
    {
        fatalError( e.what() ) ;
    }
    return 0;
}
