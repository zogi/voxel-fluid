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

#ifdef FLS_USE_REMOTERY
#include <remotery.h>
#else
#define rmt_ScopedCPUSample(name, flags)
#endif

namespace sim {

#ifdef FLS_USE_DOUBLE
typedef double Float;
#else
typedef float Float;
#endif

typedef glm::ivec3 GridSize3;
typedef GridSize3 GridIndex3;
typedef glm::tvec3<Float> Vector3;
typedef Vector3 Velocity;

template <typename T>
class Grid {
public:
    Grid(const GridSize3 &grid_size)
        : m_size(grid_size), m_cell_count(m_size.x * m_size.y * m_size.z)
    {
        initArray();
    }

    T &cell(size_t linear_index) { return m_cells[linear_index]; }
    const T &cell(size_t linear_index) const { return m_cells[linear_index]; }
    T &cell(const GridIndex3 &idx)
    {
        assert(isValid(idx));
        return cell(indexLinearFromGrid(idx));
    }
    const T &cell(const GridIndex3 &idx) const
    {
        assert(isValid(idx));
        return cell(indexLinearFromGrid(idx));
    }
    T cellSafe(const GridIndex3 &idx) const
    {
        if (isValid(idx)) {
            return cell(idx);
        } else {
            return {};
        }
    }
    T &cell(int i, int j, int k) { return cell({ i, j, k }); }
    const T &cell(int i, int j, int k) const { return cell({ i, j, k }); }
    T cellSafe(int i, int j, int k) const { return cellSafe({ i, j, k }); }

    T interpolate(Float x, Float y, Float z) const
    {
        const int i = int(floor(x));
        const int j = int(floor(y));
        const int k = int(floor(z));
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
    T interpolate(const Vector3 &v) const { return interpolate(v.x, v.y, v.z); }

    GridSize3 size() const { return m_size; }
    size_t cellCount() const { return m_cell_count; }
    bool isValid(int i, int j, int k) const
    {
        return 0 <= i && i < m_size.x && 0 <= j && j < m_size.y && 0 <= k && k < m_size.z;
    }
    bool isValid(const GridIndex3 &idx) const { return isValid(idx.x, idx.y, idx.z); }

    void clear() { m_cells.assign(m_cells.size(), T()); }

    typedef typename std::vector<T>::const_iterator const_iterator;
    const_iterator begin() const { return m_cells.begin(); }
    const_iterator end() const { return m_cells.end(); }
    typedef typename std::vector<T>::iterator iterator;
    iterator begin() { return m_cells.begin(); }
    iterator end() { return m_cells.end(); }

    size_t indexLinearFromGrid(const GridIndex3 &idx) const
    {
        return idx.x + m_size.x * (idx.y + m_size.y * idx.z);
    }

    GridIndex3 indexGridFromLinear(size_t idx) const
    {
        return { idx % m_size.x, (idx / m_size.x) % m_size.y, idx / (m_size.x * m_size.y) };
    }

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
    Float &u(const GridIndex3 &idx) { return m_u.cell(idx); }
    Float &v(const GridIndex3 &idx) { return m_v.cell(idx); }
    Float &w(const GridIndex3 &idx) { return m_w.cell(idx); }
    Float u(const GridIndex3 &idx) const { return m_u.cell(idx); }
    Float v(const GridIndex3 &idx) const { return m_v.cell(idx); }
    Float w(const GridIndex3 &idx) const { return m_w.cell(idx); }

    Float interpolateU(Float x, Float y, Float z) const
    {
        return m_u.interpolate(x + Float(0.5), y, z);
    }
    Float interpolateV(Float x, Float y, Float z) const
    {
        return m_v.interpolate(x, y + Float(0.5), z);
    }
    Float interpolateW(Float x, Float y, Float z) const
    {
        return m_w.interpolate(x, y, z + Float(0.5));
    }
    Float interpolateU(const Vector3 &v) const { return interpolateU(v.x, v.y, v.z); }
    Float interpolateV(const Vector3 &v) const { return interpolateV(v.x, v.y, v.z); }
    Float interpolateW(const Vector3 &v) const { return interpolateW(v.x, v.y, v.z); }

    // Velocity at cell center.
    glm::tvec3<Float> velocity(int i, int j, int k) const
    {
        return { Float(0.5) * (m_u.cellSafe(i, j, k) + m_u.cellSafe(i + 1, j, k)),
                 Float(0.5) * (m_v.cellSafe(i, j, k) + m_v.cellSafe(i, j + 1, k)),
                 Float(0.5) * (m_w.cellSafe(i, j, k) + m_w.cellSafe(i, j, k + 1)) };
    }
    glm::tvec3<Float> velocity(const GridIndex3 &idx) const
    {
        return velocity(idx.x, idx.y, idx.z);
    }

    glm::tvec3<Float> interpolateVelocity(Float x, Float y, Float z) const
    {
        return { interpolateU(x, y, z), interpolateV(x, y, z), interpolateW(x, y, z) };
    }
    glm::tvec3<Float> interpolateVelocity(const Vector3 &v) const
    {
        return interpolateVelocity(v.x, v.y, v.z);
    }

    // Divergence at cell center.
    Float divergence(int i, int j, int k) const
    {
        return m_scale * (u(i + 1, j, k) - u(i, j, k) + v(i, j + 1, k) - v(i, j, k) +
                          w(i, j, k + 1) - w(i, j, k));
    }
    Float divergence(const GridIndex3 &idx) const { return divergence(idx.x, idx.y, idx.z); }

    const Float dx() const { return m_dx; }
    const Float oneOverDx() const { return m_scale; }

    void clearVelocities()
    {
        m_u.clear();
        m_v.clear();
        m_w.clear();
    }

    const Grid<Float> &uGrid() const { return m_u; }
    const Grid<Float> &vGrid() const { return m_v; }
    const Grid<Float> &wGrid() const { return m_w; }

private:
    const Float m_dx, m_scale;
    Grid<Float> m_u, m_v, m_w;
};

struct SmokeData {
    Float temperature;
    Float concentration;
};

template <typename CellType>
FLS_API void advect(const MACGrid<CellType> &source_grid, Float dt, MACGrid<CellType> &dest_grid);

struct SolidCell {
    int i, j, k;
    Float u, v, w;
    SolidCell() = default;
    SolidCell(const GridIndex3 &idx, const Vector3 &velocity)
        : i(idx.x), j(idx.y), k(idx.z), u(velocity.x), v(velocity.y), w(velocity.z)
    {
    }
    SolidCell(int i_, int j_, int k_, Float u_, Float v_, Float w_)
        : i(i_), j(j_), k(k_), u(u_), v(v_), w(w_)
    {
    }
};

class FluidDensityModel {
public:
    FluidDensityModel(Float rho) { setConstant(rho); }
    FluidDensityModel(Float rho_soot_, Float rho_air_, Float temp_air_)
        : rho_air(rho_air_), rho_soot(rho_soot_), temp_air(temp_air_)
    {
        calcAlpha();
        calcBeta();
    }

    void setConstant(Float rho)
    {
        rho_air = rho;
        rho_soot = 0;
        alpha = 0;
        beta = 0;
    }

    Float rhoSoot() const { return rho_soot; }
    void setRhoSoot(Float rho_soot_)
    {
        rho_soot = rho_soot_;
        calcAlpha();
    }

    Float rhoAir() const { return rho_air; }
    void setRhoAir(Float rho_air_)
    {
        rho_air = rho_air_;
        calcAlpha();
    }

    Float tempAir() const { return temp_air; }
    void setTempAir(Float temp_air_)
    {
        temp_air = temp_air_;
        calcBeta();
    }

    Float density(const SmokeData &cell) const
    {
        const auto temp_delta = cell.temperature - temp_air;
        const auto density = rho_air * (1 + alpha * cell.concentration - beta * temp_delta);
        return clampDensity(density);
    }

    Float clampDensity(Float density) const { return std::max(Float(0.05) * rho_air, density); }

private:
    Float rho_air, rho_soot, temp_air;
    Float alpha, beta;

    Float calcAlpha() { alpha = (rho_soot - rho_air) / rho_air; }
    Float calcBeta() { beta = 1 / temp_air; }
};

class FluidSim {
public:
    typedef MACGrid<SmokeData> FluidGrid;

    static constexpr int MAX_FLUID_CELL_COUNT = 8192;

    FluidSim(const GridSize3 &size, Float dx, Float dt, const FluidDensityModel &density_model)
        : m_size(size)
        , m_dx(dx)
        , m_dt(dt)
        , m_density_model(density_model)
        , m_grid{ { m_size, dx }, { m_size, dx } }
        , m_current_grid(0)
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

    const FluidGrid &grid() const { return m_grid[m_current_grid]; }
    FluidGrid &grid() { return m_grid[m_current_grid]; }
    Float dx() const { return m_dx; }
    Float dt() const { return m_dt; }
    const FluidDensityModel &densityModel() const { return m_density_model; }
    FluidDensityModel &densityModel() { return m_density_model; }

    std::vector<SolidCell> &solidCells() { return m_solid_cells; }

    void advect();
    void pressureSolve();
    void pressureUpdate();

    bool isFluidCell(int i, int j, int k) const
    {
        constexpr Float CONCENTRATION_EMPTY_THRESHOLD = Float(1e-6);
        const auto &cell = grid().cell(i, j, k);
        return cell.concentration >= CONCENTRATION_EMPTY_THRESHOLD;
    }
    bool isFluidCell(const GridIndex3 &idx) const { return isFluidCell(idx.x, idx.y, idx.z); }

    bool isFluidCellBoundsChecked(int i, int j, int k) const
    {
        if (!grid().isValid(i, j, k)) {
            return false;
        }

        return isFluidCell(i, j, k);
    };
    bool isFluidCellBoundsChecked(const GridIndex3 &idx) const
    {
        return isFluidCellBoundsChecked(idx.x, idx.y, idx.z);
    }

    Float pressure(int i, int j, int k) const
    {
        return Float(m_pressure[m_fluid_cell_linear_index.cell(i, j, k)]);
    }
    Float pressure(const GridIndex3 &idx) const { return pressure(idx.x, idx.y, idx.z); }

    void setSolverMaxIterations(int max_iterations) { m_solver.setMaxIterations(max_iterations); }

private:
    const GridSize3 m_size;
    const Float m_dx, m_dt;
    FluidDensityModel m_density_model;
    FluidGrid m_grid[2];
    int m_current_grid;
    std::vector<SolidCell> m_solid_cells;

    // New typedef to have the option of running the solver at a higher precision.
    typedef Float SolverFloat;
    typedef Eigen::SparseMatrix<SolverFloat> SolverSparseMatrix;
    typedef Eigen::Matrix<SolverFloat, Eigen::Dynamic, 1> SolverVector;
    typedef Eigen::IncompleteCholesky<SolverFloat> Preconditioner;
    typedef Eigen::ConjugateGradient<SolverSparseMatrix, Eigen::Lower | Eigen::Upper, Preconditioner> Solver;

    Solver m_solver;

    // Temporary storage for pressure solve.
    typedef uint16_t FluidCellIndex;
    std::vector<GridIndex3> m_fluid_cell_grid_index;
    Grid<FluidCellIndex> m_fluid_cell_linear_index;

    enum class CellType : uint8_t { Empty = 0, Fluid = 1, Solid = 2 };
    struct CellNeighbors {
        CellType iplus : 2;
        CellType iminus : 2;
        CellType jplus : 2;
        CellType jminus : 2;
        CellType kplus : 2;
        CellType kminus : 2;
    };
    std::vector<CellNeighbors> m_cell_neighbors;
    SolverVector m_pressure_rhs, m_pressure;
    SolverSparseMatrix m_pressure_mtx;
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
                const auto new_p = p - dt * source_grid.interpolateVelocity(mid_p);
                dest_grid.cell(i, j, k) = source_grid.interpolate(new_p);
            }
    // Advect the U velocity component.
    for (int i = 0; i <= size.x; ++i)
        for (int j = 0; j < size.y; ++j)
            for (int k = 0; k < size.z; ++k) {
                const auto p = glm::tvec3<Float>(i - 0.5, j, k);
                const auto mid_p = p - Float(0.5) * dt * source_grid.interpolateVelocity(p);
                const auto new_p = p - dt * source_grid.interpolateVelocity(mid_p);
                dest_grid.u(i, j, k) = source_grid.interpolateU(new_p);
            }
    // Advect the V velocity component.
    for (int i = 0; i < size.x; ++i)
        for (int j = 0; j <= size.y; ++j)
            for (int k = 0; k < size.z; ++k) {
                const auto p = glm::tvec3<Float>(i, j - 0.5, k);
                const auto mid_p = p - Float(0.5) * dt * source_grid.interpolateVelocity(p);
                const auto new_p = p - dt * source_grid.interpolateVelocity(mid_p);
                dest_grid.v(i, j, k) = source_grid.interpolateV(new_p);
            }
    // Advect the W velocity component.
    for (int i = 0; i < size.x; ++i)
        for (int j = 0; j < size.y; ++j)
            for (int k = 0; k <= size.z; ++k) {
                const auto p = glm::tvec3<Float>(i, j, k - 0.5);
                const auto mid_p = p - Float(0.5) * dt * source_grid.interpolateVelocity(p);
                const auto new_p = p - dt * source_grid.interpolateVelocity(mid_p);
                dest_grid.w(i, j, k) = source_grid.interpolateW(new_p);
            }
}

#define INSTANTIATE_ADVECT(T) \
    template void advect<T>(const MACGrid<T> &source_grid, Float dt, MACGrid<T> &dest_grid);

INSTANTIATE_ADVECT(float)
INSTANTIATE_ADVECT(double)
INSTANTIATE_ADVECT(SmokeData)

void FluidSim::advect()
{
    sim::advect(m_grid[m_current_grid], m_dt, m_grid[1 - m_current_grid]);
    m_current_grid = 1 - m_current_grid;
}

void FluidSim::pressureSolve()
{
    rmt_ScopedCPUSample(FluidSim_PressureSolve, 0);

    // Set up right hand side.
    const auto setupRHS = [this]() {
        rmt_ScopedCPUSample(FluidSim_PressureSolve_SetupRHS, 0);

        const auto &grid = this->grid();
        FluidCellIndex idx = 0;
        m_cell_neighbors.clear();
        m_fluid_cell_grid_index.clear();
        m_pressure_rhs.resize(MAX_FLUID_CELL_COUNT);
        for (int i = 0; i < m_size.x; ++i)
            for (int j = 0; j < m_size.y; ++j)
                for (int k = 0; k < m_size.z; ++k) {
                    if (idx == MAX_FLUID_CELL_COUNT)
                        return;

                    if (!isFluidCell(i, j, k)) {
                        continue;
                    }
                    m_fluid_cell_linear_index.cell(i, j, k) = idx;
                    m_fluid_cell_grid_index.push_back({ i, j, k });

                    m_pressure_rhs[idx] = -grid.divergence(i, j, k);
                    CellNeighbors cell_neighbors;
                    cell_neighbors.iplus = CellType(isFluidCellBoundsChecked(i + 1, j, k));
                    cell_neighbors.iminus = CellType(isFluidCellBoundsChecked(i - 1, j, k));
                    cell_neighbors.jplus = CellType(isFluidCellBoundsChecked(i, j + 1, k));
                    cell_neighbors.jminus = CellType(isFluidCellBoundsChecked(i, j - 1, k));
                    cell_neighbors.kplus = CellType(isFluidCellBoundsChecked(i, j, k + 1));
                    cell_neighbors.kminus = CellType(isFluidCellBoundsChecked(i, j, k - 1));
                    m_cell_neighbors.push_back(cell_neighbors);

                    ++idx;
                }

        // Add terms due to solid boundaries.
        const auto scale = grid.oneOverDx();
        for (const auto solid : m_solid_cells) {
            const int i = solid.i;
            const int j = solid.j;
            const int k = solid.k;
            if (isFluidCellBoundsChecked(i - 1, j, k)) {
                const int idx = m_fluid_cell_linear_index.cell(i - 1, j, k);
                m_cell_neighbors[idx].iplus = CellType::Solid;
                m_pressure_rhs[idx] += scale * (grid.u(i, j, k) - solid.u);
            }
            if (isFluidCellBoundsChecked(i + 1, j, k)) {
                const int idx = m_fluid_cell_linear_index.cell(i + 1, j, k);
                m_cell_neighbors[idx].iminus = CellType::Solid;
                m_pressure_rhs[idx] -= scale * (grid.u(i + 1, j, k) - solid.u);
            }
            if (isFluidCellBoundsChecked(i, j - 1, k)) {
                const int idx = m_fluid_cell_linear_index.cell(i, j - 1, k);
                m_cell_neighbors[idx].jplus = CellType::Solid;
                m_pressure_rhs[idx] += scale * (grid.v(i, j, k) - solid.v);
            }
            if (isFluidCellBoundsChecked(i, j + 1, k)) {
                const int idx = m_fluid_cell_linear_index.cell(i, j + 1, k);
                m_cell_neighbors[idx].jminus = CellType::Solid;
                m_pressure_rhs[idx] -= scale * (grid.v(i, j + 1, k) - solid.v);
            }
            if (isFluidCellBoundsChecked(i, j, k - 1)) {
                const int idx = m_fluid_cell_linear_index.cell(i, j, k - 1);
                m_cell_neighbors[idx].kplus = CellType::Solid;
                m_pressure_rhs[idx] += scale * (grid.w(i, j, k) - solid.w);
            }
            if (isFluidCellBoundsChecked(i, j, k + 1)) {
                const int idx = m_fluid_cell_linear_index.cell(i, j, k + 1);
                m_cell_neighbors[idx].kminus = CellType::Solid;
                m_pressure_rhs[idx] -= scale * (grid.w(i, j, k + 1) - solid.w);
            }
        }
    };
    setupRHS();

    // Set up the matrix.
    {
        rmt_ScopedCPUSample(FluidSim_PressureSolve_MatrixSetup, 0);

        const Float scale = m_dt / (m_dx * m_dx);

        const auto fluid_cell_count = m_cell_neighbors.size();
        m_pressure_rhs.conservativeResize(fluid_cell_count);
        m_pressure_mtx.resize(fluid_cell_count, fluid_cell_count);
        m_pressure_mtx.setZero();

        if (fluid_cell_count == 0) {
            // Nothing to do if there are no fluids.
            return;
        }

        // TODO: pre-allocate and fill sparse matrix buffers directly, so this can be parallelized.
        FluidCellIndex row = 0;
        for (const auto grid_index : m_fluid_cell_grid_index) {
            const int i = grid_index.x;
            const int j = grid_index.y;
            const int k = grid_index.z;
            if (!isFluidCell(i, j, k)) {
                continue;
            }
            const auto density = m_density_model.density(grid().cell(grid_index));

            const auto handleNeighbor = [this, row, scale, density](const GridIndex3 &neighbor_index) {
                const FluidCellIndex col = m_fluid_cell_linear_index.cell(neighbor_index);
                const auto neighbor_density = m_density_model.density(grid().cell(neighbor_index));
                const auto avg_density = Float(0.5) * (density + neighbor_density);
                m_pressure_mtx.insert(col, row) = -scale / avg_density;
            };

            // Fill this nonzero coeffs of this column in order:
            // k minus, j minus, i minus, current cell, i plus, j plus, k plus
            const auto cell_neighbors = m_cell_neighbors[row];
            if (cell_neighbors.kminus == CellType::Fluid) {
                handleNeighbor({ i, j, k - 1 });
            }
            if (cell_neighbors.jminus == CellType::Fluid) {
                handleNeighbor({ i, j - 1, k });
            }
            if (cell_neighbors.iminus == CellType::Fluid) {
                handleNeighbor({ i - 1, j, k });
            }
            {
                const auto isSolid = [](CellType ct) -> int { return ct == CellType::Solid; };
                const auto solid_count =
                    isSolid(cell_neighbors.iminus) + isSolid(cell_neighbors.iplus) +
                    isSolid(cell_neighbors.jminus) + isSolid(cell_neighbors.jplus) +
                    isSolid(cell_neighbors.kminus) + isSolid(cell_neighbors.kplus);
                const auto nonsolid_count = 6 - solid_count;
                m_pressure_mtx.insert(row, row) = nonsolid_count * scale / density;
            }
            if (cell_neighbors.iplus == CellType::Fluid) {
                handleNeighbor({ i + 1, j, k });
            }
            if (cell_neighbors.jplus == CellType::Fluid) {
                handleNeighbor({ i, j + 1, k });
            }
            if (cell_neighbors.kplus == CellType::Fluid) {
                handleNeighbor({ i, j, k + 1 });
            }

            row += 1;
        }
    }

    // Solve for pressure.
    {
        rmt_ScopedCPUSample(FluidSim_PressureSolve_Solve, 0);

        m_solver.compute(m_pressure_mtx);
        m_pressure = m_solver.solve(m_pressure_rhs);
    }
}

void FluidSim::pressureUpdate()
{
    auto &grid = this->grid();

    // Pressure update.
    const Float scale = m_dt / m_dx;
    for (int idx = 0; idx < m_fluid_cell_grid_index.size(); ++idx) {
        const auto &grid_index = m_fluid_cell_grid_index[idx];
        const int i = grid_index.x;
        const int j = grid_index.y;
        const int k = grid_index.z;
        const Float p = scale * Float(m_pressure[idx]);
        const auto density = m_density_model.density(grid.cell(grid_index));
        const auto avgDensity = [this, density](const GridIndex3 &neighbor) {
            return Float(0.5) * (density + m_density_model.density(this->grid().cellSafe(neighbor)));
        };
        grid.u(i, j, k) -= p / density;
        grid.u(i + 1, j, k) += p / avgDensity({ i + 1, j, k });
        grid.v(i, j, k) -= p / density;
        grid.v(i, j + 1, k) += p / avgDensity({ i, j + 1, k });
        grid.w(i, j, k) -= p / density;
        grid.w(i, j, k + 1) += p / avgDensity({ i, j, k + 1 });
    }
    for (const auto solid : m_solid_cells) {
        const int i = solid.i;
        const int j = solid.j;
        const int k = solid.k;
        // TODO: slippery boundary
        grid.u(i, j, k) = solid.u;
        grid.u(i + 1, j, k) = solid.u;
        grid.v(i, j, k) = solid.v;
        grid.v(i, j + 1, k) = solid.v;
        grid.w(i, j, k) = solid.w;
        grid.w(i, j, k + 1) = solid.w;
    }
}

} // namespace sim

#endif // FLS_IMPLEMENTATION
