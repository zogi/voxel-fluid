#ifndef FLS_FLUID_SIM_H_
#define FLS_FLUID_SIM_H_

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <cassert>

#ifndef FLS_API
#ifdef FLS_PRIVATE
#define FLS_API static
#else
#define FLS_API extern
#endif
#endif

namespace sim {

#ifdef FLS_USE_DOUBLE
typedef double Float;
#else
typedef float Float;
#endif

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

    T &cell(int i, int j, int k)
    {
        assert(isValid(i, j, k));
        return m_cells[cellIndex(i, j, k)];
    }
    const T &cell(int i, int j, int k) const
    {
        assert(isValid(i, j, k));
        return m_cells[cellIndex(i, j, k)];
    }
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

template <typename CellType>
FLS_API void advect(const MACGrid<CellType> &source_grid, Float dt, MACGrid<CellType> &dest_grid);

class FluidSim {
public:
    typedef MACGrid<SmokeData> FluidGrid;

    static constexpr int MAX_FLUID_CELL_COUNT = 8192;

    FluidSim(const GridSize3 &size, Float dx, Float dt, Float rho)
        : m_size(size)
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


#endif // FLS_FLUID_SIM_H_

#ifdef FLS_IMPLEMENTATION

template <>
GLM_FUNC_QUALIFIER sim::SmokeData
glm::mix<sim::SmokeData, sim::Float>(sim::SmokeData x, sim::SmokeData y, sim::Float alpha)
{
    sim::SmokeData res;
    res.concentration = glm::mix(x.concentration, y.concentration, alpha);
    res.temperature = glm::mix(x.temperature, y.temperature, alpha);
    return res;
}

namespace sim {

#ifdef __GNUG__
int popcnt(unsigned int a) { return __builtin_popcount(a); }
#elif defined _MSC_VER
int popcnt(unsigned int a) { return __popcnt(a); }
#else
int popcnt(unsigned int a)
{
    return std::bitset<std::numeric_limits<unsigned int>::digits>(a).count();
}
#endif

template <typename CellType>
void advect(const MACGrid<CellType> &source_grid, Float dt, MACGrid<CellType> &dest_grid)
{
    // RK2
    // xmid = x - 0.5 dt u(x)
    // xnew = x - dt u(xmid)

    const auto size = source_grid.size();

    // Advect cell contents.
    for (int i = 0; i < size.x; ++i)
        for (int j = 0; j < size.y; ++j)
            for (int k = 0; k < size.z; ++k) {
                const auto p = glm::tvec3<Float>(i, j, k);
                const auto mid_p = p - Float(0.5) * dt * source_grid.velocity(i, j, k);
                const auto new_p = p - dt * source_grid.interpolateVelocity(mid_p.x, mid_p.y, mid_p.z);
                dest_grid.cell(i, j, k) = source_grid.interpolate(new_p.x, new_p.y, new_p.z);
            }
    // Advect the U velocity component.
    for (int i = 0; i <= size.x; ++i)
        for (int j = 0; j < size.y; ++j)
            for (int k = 0; k < size.z; ++k) {
                const auto p = glm::tvec3<Float>(i - 0.5, j, k);
                const auto mid_p =
                    p - Float(0.5) * dt * source_grid.interpolateVelocity(p.x, p.y, p.z);
                const auto new_p = p - dt * source_grid.interpolateVelocity(mid_p.x, mid_p.y, mid_p.z);
                dest_grid.u(i, j, k) = source_grid.interpolateU(new_p.x, new_p.y, new_p.z);
            }
    // Advect the V velocity component.
    for (int i = 0; i < size.x; ++i)
        for (int j = 0; j <= size.y; ++j)
            for (int k = 0; k < size.z; ++k) {
                const auto p = glm::tvec3<Float>(i, j - 0.5, k);
                const auto mid_p =
                    p - Float(0.5) * dt * source_grid.interpolateVelocity(p.x, p.y, p.z);
                const auto new_p = p - dt * source_grid.interpolateVelocity(mid_p.x, mid_p.y, mid_p.z);
                dest_grid.v(i, j, k) = source_grid.interpolateV(new_p.x, new_p.y, new_p.z);
            }
    // Advect the W velocity component.
    for (int i = 0; i < size.x; ++i)
        for (int j = 0; j < size.y; ++j)
            for (int k = 0; k <= size.z; ++k) {
                const auto p = glm::tvec3<Float>(i, j, k - 0.5);
                const auto mid_p =
                    p - Float(0.5) * dt * source_grid.interpolateVelocity(p.x, p.y, p.z);
                const auto new_p = p - dt * source_grid.interpolateVelocity(mid_p.x, mid_p.y, mid_p.z);
                dest_grid.w(i, j, k) = source_grid.interpolateW(new_p.x, new_p.y, new_p.z);
            }
}

#define INSTANTIATE_ADVECT(T) \
    template void advect<T>(const MACGrid<T> &source_grid, Float dt, MACGrid<T> &dest_grid);

INSTANTIATE_ADVECT(float)
INSTANTIATE_ADVECT(double)
INSTANTIATE_ADVECT(SmokeData)

void FluidSim::pressureSolve()
{
    // Set up right hand side.

    const auto setupRHS = [this]() {
        FluidCellIndex idx = 0;
        m_cell_neighbors.clear();
        m_fluid_cell_grid_index.clear();
        m_pressure_rhs.resize(MAX_FLUID_CELL_COUNT);
        for (int i = 0; i < m_size.x; ++i)
            for (int j = 0; j < m_size.y; ++j)
                for (int k = 0; k < m_size.z; ++k) {
                    if (idx == MAX_FLUID_CELL_COUNT)
                        return;

                    if (cellType(i, j, k) != CellType::Fluid) {
                        continue;
                    }
                    m_fluid_cell_linear_index.cell(i, j, k) = idx;
                    m_fluid_cell_grid_index.push_back({ i, j, k });

                    m_pressure_rhs[idx] = -m_grid.divergence(i, j, k);
                    CellNeighbors cell_neighbors;
                    cell_neighbors.iplus = cellTypeBoundsChecked(i + 1, j, k);
                    cell_neighbors.iminus = cellTypeBoundsChecked(i - 1, j, k);
                    cell_neighbors.jplus = cellTypeBoundsChecked(i, j + 1, k);
                    cell_neighbors.jminus = cellTypeBoundsChecked(i, j - 1, k);
                    cell_neighbors.kplus = cellTypeBoundsChecked(i, j, k + 1);
                    cell_neighbors.kminus = cellTypeBoundsChecked(i, j, k - 1);
                    m_cell_neighbors.push_back(cell_neighbors);

                    ++idx;
                }
    };
    setupRHS();

    // Set up the matrix.

    // Fluid density (TODO: implement variable density solve).
    const Float rho = density();
    const Float scale = m_dt / (rho * m_dx * m_dx);

    const auto fluid_cell_count = m_cell_neighbors.size();
    m_pressure_rhs.conservativeResize(fluid_cell_count);
    m_pressure_mtx.resize(fluid_cell_count, fluid_cell_count);
    m_pressure_mtx.setZero();

    // TODO: pre-allocate and fill sparse matrix buffers directly, so this can be parallelized.
    FluidCellIndex row = 0;
    for (const auto grid_index : m_fluid_cell_grid_index) {
        const int i = grid_index.x;
        const int j = grid_index.y;
        const int k = grid_index.z;
        if (cellType(i, j, k) != CellType::Fluid) {
            continue;
        }
        // Fill this nonzero coeffs of this column in order:
        // k minus, j minus, i minus, current cell, i plus, j plus, k plus
        const auto cell_neighbors = m_cell_neighbors[row];
        if (cell_neighbors.kminus == CellType::Fluid) {
            const FluidCellIndex col = m_fluid_cell_linear_index.cell(i, j, k - 1);
            m_pressure_mtx.insert(col, row) = -scale;
        }
        if (cell_neighbors.jminus == CellType::Fluid) {
            const FluidCellIndex col = m_fluid_cell_linear_index.cell(i, j - 1, k);
            m_pressure_mtx.insert(col, row) = -scale;
        }
        if (cell_neighbors.iminus == CellType::Fluid) {
            const FluidCellIndex col = m_fluid_cell_linear_index.cell(i - 1, j, k);
            m_pressure_mtx.insert(col, row) = -scale;
        }
        {
            const auto nonsolid_count = 6; // TODO: calculate actual count
            m_pressure_mtx.insert(row, row) = nonsolid_count * scale;
        }
        if (cell_neighbors.iplus == CellType::Fluid) {
            const FluidCellIndex col = m_fluid_cell_linear_index.cell(i + 1, j, k);
            m_pressure_mtx.insert(col, row) = -scale;
        }
        if (cell_neighbors.jplus == CellType::Fluid) {
            const FluidCellIndex col = m_fluid_cell_linear_index.cell(i, j + 1, k);
            m_pressure_mtx.insert(col, row) = -scale;
        }
        if (cell_neighbors.kplus == CellType::Fluid) {
            const FluidCellIndex col = m_fluid_cell_linear_index.cell(i, j, k + 1);
            m_pressure_mtx.insert(col, row) = -scale;
        }

        row += 1;
    }

    // Solve for pressure.
    Solver solver;
    solver.compute(m_pressure_mtx);
    m_pressure = solver.solve(m_pressure_rhs);
    assert(solver.info() == Eigen::ComputationInfo::Success);
}

void FluidSim::pressureUpdate()
{
    // Pressure update.
    // TODO: variable density.
    const Float rho = density();
    const Float scale = m_dt / (rho * m_dx);
#pragma omp parallel for
    for (int idx = 0; idx < m_fluid_cell_grid_index.size(); ++idx) {
        const auto &grid_index = m_fluid_cell_grid_index[idx];
        const int i = grid_index.x;
        const int j = grid_index.y;
        const int k = grid_index.z;
        const Float p = scale * Float(m_pressure[idx]);
        m_grid.u(i, j, k) -= p;
        m_grid.u(i + 1, j, k) += p;
        m_grid.v(i, j, k) -= p;
        m_grid.v(i, j + 1, k) += p;
        m_grid.w(i, j, k) -= p;
        m_grid.w(i, j, k + 1) += p;
    }
}

} // namespace sim

#endif // FLS_IMPLEMENTATION
