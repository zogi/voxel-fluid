#pragma once
#ifndef _COMMON_H
#define _COMMON_H

#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"

// spdlog includes windows.h on Windows, which should be included before glfw.
#include <spdlog/spdlog.h>

#include <GL/glew.h>
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

#endif
