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

TEST(GridInterpolate, ExactOnGridPoint)
{
    constexpr double FLOAT_TOLERANCE = 1e-15;

    sim::Grid<double> g({ 2, 2, 2 });
    for (int i = 0; i < 8; ++i)
        g.cell(i & 1, (i & 2) >> 1, (i & 4) >> 2) = i;

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                ASSERT_EQ(g.interpolate(float(i), float(j), float(k)), double(g.cell(i, j, k)));
}

TEST(GridInterpolate, AverageInMiddle)
{
    constexpr double FLOAT_TOLERANCE = 1e-16;

    sim::Grid<double> g({ 2, 2, 2 });
    for (int i = 0; i < 8; ++i)
        g.cell(i & 1, (i & 2) >> 1, (i & 4) >> 2) = i;

    ASSERT_NEAR(g.interpolate(0.5, 0.5, 0.5), 3.5, FLOAT_TOLERANCE);
}

TEST(GridInterpolate, ZeroOutOfBounds)
{
    sim::Grid<double> g({ 1, 1, 1 });
    g.cell(0, 0, 0) = 42;
    ASSERT_EQ(g.interpolate(-1.1f, -2.5f, 0.1f), 0.0);
}

TEST(GridInterpolate, AlongAxes)
{
    {
        sim::Grid<double> g({ 2, 1, 1 });
        g.cell(0, 0, 0) = 0;
        g.cell(1, 0, 0) = 1;
        ASSERT_EQ(g.interpolate(0.25, 0, 0), 0.25);
        ASSERT_EQ(g.interpolate(0.5, 0, 0), 0.5);
        ASSERT_EQ(g.interpolate(0.75, 0, 0), 0.75);
    }
    {
        sim::Grid<double> g({ 1, 2, 1 });
        g.cell(0, 0, 0) = 0;
        g.cell(0, 1, 0) = 1;
        ASSERT_EQ(g.interpolate(0, 0.25, 0), 0.25);
        ASSERT_EQ(g.interpolate(0, 0.5, 0), 0.5);
        ASSERT_EQ(g.interpolate(0, 0.75, 0), 0.75);
    }
    {
        sim::Grid<double> g({ 1, 1, 2 });
        g.cell(0, 0, 0) = 0;
        g.cell(0, 0, 1) = 1;
        ASSERT_EQ(g.interpolate(0, 0, 0.25), 0.25);
        ASSERT_EQ(g.interpolate(0, 0, 0.5), 0.5);
        ASSERT_EQ(g.interpolate(0, 0, 0.75), 0.75);
    }
}
