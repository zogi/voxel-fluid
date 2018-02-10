#include "common.h"

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
