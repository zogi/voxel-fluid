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
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#include <array>
#include <bitset>
#include <iostream>
#include <limits>
#include <memory>
#include <cstdio>


// === Fluid sim ===

namespace sim {

typedef double Float;
typedef glm::ivec3 GridSize3;
typedef GridSize3 GridIndex3;
typedef glm::vec3 Velocity;

template <typename T>
class Grid {
public:
    Grid(const GridSize3 &grid_size)
        : m_size(grid_size), m_cell_count(m_size.x * m_size.y * m_size.z)
    {
        initArray();
    }

    T &cell(int i, int j, int k) { return m_cells[cellIndex(i, j, k)]; }
    const T &cell(int i, int j, int k) const { return m_cells[cellIndex(i, j, k)]; }
    T cellSafe(int i, int j, int k) const
    {
        if (isValid(i, j, k)) {
            return cell(i, j, k);
        } else {
            return {};
        }
    }

    T interpolate(Float x, Float y, Float z) const
    {
        const int i = int(x);
        const int j = int(y);
        const int k = int(z);
        const auto alpha_x = x - i;
        const auto v_yz = glm::mix(cellSafe(i, j, k), cellSafe(i + 1, j, k), alpha_x);
        const auto v_Yz = glm::mix(cellSafe(i, j + 1, k), cellSafe(i + 1, j + 1, k), alpha_x);
        const auto v_yZ = glm::mix(cellSafe(i, j, k + 1), cellSafe(i + 1, j, k + 1), alpha_x);
        const auto v_YZ = glm::mix(cellSafe(i, j + 1, k + 1), cellSafe(i + 1, j + 1, k + 1), alpha_x);
        const auto alpha_y = y - j;
        const auto v_z = glm::mix(v_yz, v_Yz, alpha_y);
        const auto v_Z = glm::mix(v_yZ, v_YZ, alpha_y);
        const auto alpha_z = z - k;
        return glm::mix(v_z, v_Z, alpha_z);
    }

    GridSize3 size() const { return m_size; }
    size_t cellCount() const { return m_cell_count; }
    bool isValid(int i, int j, int k) const
    {
        return 0 <= i && i < m_size.x && 0 <= j && j < m_size.y && 0 <= k && k < m_size.z;
    }

    inline size_t cellIndex(int i, int j, int k) const { return i + m_size.x * (j + m_size.y * k); }

private:
    const GridSize3 m_size;
    const size_t m_cell_count;
    std::vector<T> m_cells;
    void initArray() { m_cells.resize(m_cell_count); }
};

template <typename DataInCell>
class MACGrid : public Grid<DataInCell> {
public:
    typedef Grid<DataInCell> Super;
    typedef Float FloatType;

    MACGrid(const GridSize3 &grid_size, Float dx)
        : Super(grid_size)
        , m_dx(dx)
        , m_scale(1.0f / dx)
        , m_u(grid_size + GridSize3(1, 0, 0))
        , m_v(grid_size + GridSize3(0, 1, 0))
        , m_w(grid_size + GridSize3(0, 0, 1))
    {
    }

    // Subtract half from i to get the logical MAC grid coordinate for the u component.
    // E.g. uIndex(i, j, k) returns the linear index (into m_velocities) of the u velocity
    // component at (i-1/2, j, k).  Similarly subtract half from j and k when querying the
    // v and w components respectively to get the logical index.
    Float &u(int i, int j, int k) { return m_u.cell(i, j, k); }
    Float &v(int i, int j, int k) { return m_v.cell(i, j, k); }
    Float &w(int i, int j, int k) { return m_w.cell(i, j, k); }
    Float u(int i, int j, int k) const { return m_u.cell(i, j, k); }
    Float v(int i, int j, int k) const { return m_v.cell(i, j, k); }
    Float w(int i, int j, int k) const { return m_w.cell(i, j, k); }

    Float interpolateU(Float x, Float y, Float z) const { return m_u.interpolate(x, y, z); }
    Float interpolateV(Float x, Float y, Float z) const { return m_v.interpolate(x, y, z); }
    Float interpolateW(Float x, Float y, Float z) const { return m_w.interpolate(x, y, z); }

    glm::tvec3<Float> velocity(int i, int j, int k) const
    {
        return { Float(0.5) * (m_u.cellSafe(i, j, k) + m_u.cellSafe(i + 1, j, k)),
                 Float(0.5) * (m_v.cellSafe(i, j, k) + m_v.cellSafe(i, j + 1, k)),
                 Float(0.5) * (m_w.cellSafe(i, j, k) + m_w.cellSafe(i, j, k + 1)) };
    }

    glm::tvec3<Float> interpolateVelocity(Float x, Float y, Float z) const
    {
        return { interpolateU(x + Float(0.5), y, z), interpolateV(x, y + Float(0.5), z),
                 interpolateW(x, y, z + Float(0.5)) };
    }

    Float divergence(int i, int j, int k) const
    {
        return m_scale * (u(i + 1, j, k) - u(i, j, k) + v(i, j + 1, k) - v(i, j, k) +
                          w(i, j, k + 1) - w(i, j, k));
    }

private:
    const Float m_dx, m_scale;
    Grid<Float> m_u, m_v, m_w;
};

struct SmokeData {
    float temperature;
    float concentration;
};

class FluidSim {
public:
    typedef MACGrid<SmokeData> FluidGrid;

    static constexpr int MAX_FLUID_CELL_COUNT = 8192;

    FluidSim(int n, Float dx, Float dt, Float rho)
        : m_size({ n, n, n })
        , m_dx(dx)
        , m_dt(dt)
        , m_rho(rho)
        , m_grid(m_size, dx)
        , m_fluid_cell_linear_index(m_size)
        , m_pressure_rhs(MAX_FLUID_CELL_COUNT)
        , m_pressure_mtx(MAX_FLUID_CELL_COUNT, MAX_FLUID_CELL_COUNT)
    {
        // Preallocate temporary storage for pressure solver.
        m_cell_neighbors.reserve(MAX_FLUID_CELL_COUNT);
        m_fluid_cell_grid_index.reserve(MAX_FLUID_CELL_COUNT);
        // In each column there can be at most 9 nonzero coeffs:
        // one for the cell and one for each neighbor containing fluid.
        m_pressure_mtx.reserve(9 * MAX_FLUID_CELL_COUNT);
    }

    FluidGrid &grid() { return m_grid; }
    Float density() { return m_rho; }
    void pressureSolve();
    void pressureUpdate();

    enum class CellType : uint8_t {
        Empty,
        Fluid
        // TODO: solid wall
    };

    CellType cellType(int i, int j, int k) const
    {
        constexpr Float EMPTY_THRESHOLD = Float(1e-6);
        const auto &cell = m_grid.cell(i, j, k);
        if (cell.concentration < EMPTY_THRESHOLD) {
            return CellType::Empty;
        } else {
            return CellType::Fluid;
        }
    }

    CellType cellTypeBoundsChecked(int i, int j, int k) const
    {
        if (!m_grid.isValid(i, j, k)) {
            return CellType::Empty;
        }

        return cellType(i, j, k);
    };

private:
    const GridSize3 m_size;
    const Float m_dx, m_dt, m_rho;
    FluidGrid m_grid;

    // New typedef to have the option of running the solver at a higher precision.
    typedef double SimFloat;
    typedef Eigen::SparseMatrix<SimFloat> SparseMatrix;
    typedef Eigen::Matrix<SimFloat, Eigen::Dynamic, 1> Vector;
    typedef Eigen::IncompleteCholesky<SimFloat> Preconditioner;
    typedef Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper, Preconditioner> Solver;

    // Temporary storage for pressure solve.
    typedef uint16_t FluidCellIndex;
    std::vector<GridIndex3> m_fluid_cell_grid_index;
    Grid<FluidCellIndex> m_fluid_cell_linear_index;

    struct CellNeighbors {
        CellType iplus : 2;
        CellType iminus : 2;
        CellType jplus : 2;
        CellType jminus : 2;
        CellType kplus : 2;
        CellType kminus : 2;
    };
    std::vector<CellNeighbors> m_cell_neighbors;
    Vector m_pressure_rhs, m_pressure;
    SparseMatrix m_pressure_mtx;
};

} // namespace sim

#endif
