project(Fluid)
cmake_minimum_required(VERSION 3.9)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# glfw
# link against: glfw
# Adding glfw as a submodule since I couldn't find a properly written conan
# package for glfw which would handle MSVC runtime settings correctly.
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_VULKAN_STATIC OFF CACHE BOOL "" FORCE)
set(GLFW_DOCUMENT_INTERNALS OFF CACHE BOOL "" FORCE)
if ((CONAN_LINK_RUNTIME STREQUAL "/MT") OR (CONAN_LINK_RUNTIME STREQUAL "/MTd"))
    set(USE_MSVC_RUNTIME_LIBRARY_DLL OFF CACHE BOOL "" FORCE)
else ()
    set(USE_MSVC_RUNTIME_LIBRARY_DLL ON CACHE BOOL "" FORCE)
endif ()
add_subdirectory(external/glfw)

# imgui
file(GLOB IMGUI_SRC external/imgui/*.cpp)
add_library(imgui STATIC ${IMGUI_SRC})
target_include_directories(imgui PUBLIC external/imgui)

# remotery
# link against: remotery
add_library(remotery STATIC external/remotery/lib/Remotery.c)
target_include_directories(remotery PUBLIC external/remotery/lib)
target_compile_definitions(remotery PUBLIC RMT_USE_OPENGL)

# variant
# link against: variant
add_library(variant INTERFACE)
target_include_directories(variant INTERFACE
    ${PROJECT_SOURCE_DIR}/external/variant/include/mpark)

# OpenGL
find_package(OpenGL REQUIRED)

# Main executable.
set(LIBS variant imgui remotery glfw ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
file(GLOB SRC *.cpp)
add_executable(fluid ${SRC})
target_link_libraries(fluid
    ${CONAN_TARGETS} ${LIBS})

# Unit tests.
file(GLOB TEST_SRC test/*.cpp)
add_executable(unittest ${TEST_SRC})
target_link_libraries(unittest ${CONAN_TARGETS} ${LIBS})
include(GoogleTest)
gtest_add_tests(TARGET unittest)
enable_testing()