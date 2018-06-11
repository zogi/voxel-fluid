#pragma once
#ifndef _COMMON_H
#define _COMMON_H

#include <imgui.h>
#include "imgui_impl_glfw_gl3.h"

// spdlog includes windows.h on Windows, which should be included before glfw.
#include <spdlog/spdlog.h>

#include <PerlinNoise.hpp>

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
#include <glm/gtx/polar_coordinates.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <variant.hpp>
#include <json/json.h>

#include <array>
#include <bitset>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <cstdio>

#if _MSC_VER
// Disable warning about strcpy, sprintf and sscanf may be unsafe.
#pragma warning(disable : 4996)
#endif

template <typename T>
struct SphericalCoords {
    T radius;
    T polar;
    T azimuthal;
};
template <typename T>
inline glm::tvec3<T> euclideanFromSpherical(const SphericalCoords<T> &spherical)
{
    const auto latitude = glm::half_pi<T>() - spherical.polar;
    return spherical.radius * glm::euclidean(glm::tvec2<T>(latitude, spherical.azimuthal));
}
template <typename T>
inline SphericalCoords<T> sphericalFromEuclidean(const glm::tvec3<T> &euclidean)
{
    const auto glm_polar = glm::polar(euclidean);
    SphericalCoords<T> res;
    res.radius = glm::length(euclidean);
    res.polar = glm::half_pi<T>() - glm_polar.x;
    res.azimuthal = glm_polar.y;
    return res;
}

#endif
