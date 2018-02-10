#include "gtest/gtest.h"
#include "../common.h"
#include <random>

TEST(PressureSolve, DivergenceFreeFluidAfterSolve)
{
    constexpr double FLOAT_TOLERANCE = 1e-14;
    constexpr int n = 10;

    std::default_random_engine gen(0);

    std::uniform_int_distribution<int> di(0, 1);
    sim::FluidSim fluid_sim(n, 0.5, 0.6, 0.7);
    auto &grid = fluid_sim.grid();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                grid.cell(i, j, k).concentration = di(gen) ? 1.0f : 0.0f;

    std::uniform_real_distribution<double> df(-1, 1);
    for (int i = 0; i <= n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                grid.u(i, j, k) = df(gen);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= n; ++j)
            for (int k = 0; k < n; ++k)
                grid.v(i, j, k) = df(gen);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k <= n; ++k)
                grid.w(i, j, k) = df(gen);

    fluid_sim.pressureSolve();
    fluid_sim.pressureUpdate();

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                if (fluid_sim.cellType(i, j, k) == sim::FluidSim::CellType::Fluid)
                    ASSERT_NEAR(grid.divergence(i, j, k), 0.0, FLOAT_TOLERANCE);
}
