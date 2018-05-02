project(Fluid)
cmake_minimum_required(VERSION 3.9)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# glad
# link against: glad
find_package(OpenGL REQUIRED)
add_library(glad STATIC external/glad/src/glad.c)
target_include_directories(glad PUBLIC external/glad/include)
target_link_libraries(glad ${OPENGL_glu_LIBRARY} ${OPENGL_gl_LIBRARY} ${CMAKE_DL_LIBS})

# glm
# link against: glm
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE
    ${PROJECT_SOURCE_DIR}/external/glm)

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

# eigen
# link against: eigen
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE
    ${PROJECT_SOURCE_DIR}/external/eigen)

add_library(fluid_sim INTERFACE)
target_link_libraries(fluid_sim INTERFACE glm eigen)

# Main executable.
set(LIBS fluid_sim variant imgui remotery glfw glm glad)
file(GLOB SRC *.cpp *.h)
add_executable(fluid ${SRC})
target_link_libraries(fluid ${CONAN_TARGETS} ${LIBS})

# Unit tests.
file(GLOB TEST_SRC test/*.cpp *.h)
add_executable(unittest ${TEST_SRC})
target_link_libraries(unittest fluid_sim CONAN_PKG::gtest)
include(GoogleTest)
gtest_add_tests(TARGET unittest)
enable_testing()
