#pragma once
#ifndef _COMMON_H
#define _COMMON_H

#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"

// spdlog includes windows.h on Windows, which should be included before glfw.
#include <spdlog/spdlog.h>

// To prevent glad from including windows.h.
#if defined(_WIN32) && !defined(APIENTRY)
#define APIENTRY __stdcall
#endif
#include <glad/glad.h>
#undef APIENTRY

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <variant.hpp>

#include <array>
#include <bitset>
#include <iostream>
#include <limits>
#include <memory>
#include <cstdio>

#if _MSC_VER
// Disable warning about strcpy, sprintf and sscanf may be unsafe.
#pragma warning(disable : 4996)
#endif

#endif
