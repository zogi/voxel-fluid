#include "gtest/gtest.h"
#define FLS_USE_DOUBLE
#define FLS_IMPLEMENTATION
#include "../fluid_sim.h"
#include <random>

TEST(PressureSolve, DivergenceFreeFluidAfterSolve)
{
    constexpr double FLOAT_TOLERANCE = 1e-14;
    constexpr int n = 10;

    std::default_random_engine gen(0);

    sim::FluidSim fluid_sim({ n, n, n }, 0.5, 0.6, 0.7);
    auto &grid = fluid_sim.grid();

    std::uniform_int_distribution<int> di(0, 1);
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
                if (fluid_sim.isFluidCell(i, j, k))
                    ASSERT_NEAR(grid.divergence(i, j, k), 0.0, FLOAT_TOLERANCE);
}

TEST(PressureSolve, WithNonMovingSolidCells)
{
    const auto testCase = [](int axis, sim::Float speed, sim::Float dx, sim::Float dt, sim::Float rho) {
        constexpr double FLOAT_TOLERANCE = 1e-14;
        sim::GridIndex3 delta = { 0, 0, 0 };
        delta[axis] = 1;
        sim::GridSize3 size = sim::GridSize3(1, 1, 1) + 2 * delta;
        sim::FluidSim fluid_sim(size, dx, dt, rho);
        auto &grid = fluid_sim.grid();
        fluid_sim.solidCells().resize(1);

        // Configuration: solid wall, fluid, fluid
        // At the two boundaries of these 3 cells the u velocity component is -1.
        fluid_sim.solidCells().at(0) = sim::SolidCell(0 * delta, { 0, 0, 0 });
        grid.cell(0 * delta).concentration = 0; // wall
        grid.cell(1 * delta).concentration = 1; // fluid
        grid.cell(2 * delta).concentration = 1; // fluid
        for (int i = 1; i < 3; ++i) {
            if (axis == 0)
                grid.u(i * delta) = -speed;
            else if (axis == 1)
                grid.v(i * delta) = -speed;
            else if (axis == 2)
                grid.w(i * delta) = -speed;
        }
        fluid_sim.pressureSolve();
        // dt == dx == rho == 1 for simplicity
        // grid.u(1, 0, 0) := -u
        // Pressure in the fluid cells: p1 and p2
        // Pressure in the solid: p1 + (u - 0)
        // Laplacian of pressure in the fluid cell (1, 0, 0): 6*p1 - (p_solid) - p2 = 5*p1 - u - p2
        // Laplacian of pressure in the fluid cell (2, 0, 0): 6*p2 - p1
        // Divergence of u in fluid cell (1, 0, 0): 0
        // Divergence of u in fluid cell (2, 0, 0): u
        // The equation:
        // 5*p1 -   p2 =  u
        //  -p1 + 6*p2 = -u
        // Solution: p1 = 5/29 * u; p2 = -4/29 * u
        // If dt == dx == rho == 1 is not the case, the laplacian scales with dt/(rho*dx*dx),
        // the divergence on the rhs and the terms added due to solid boundaries scale with 1/dx,
        // so just scale the solution above computed above with (rho*dx)/dt.
        const auto scale = rho * dx / dt;
        ASSERT_NEAR(fluid_sim.pressure(1 * delta), 5.0 / 29.0 * speed * scale, FLOAT_TOLERANCE);
        ASSERT_NEAR(fluid_sim.pressure(2 * delta), -4.0 / 29.0 * speed * scale, FLOAT_TOLERANCE);
        // Should be divergence-free after pressure update.
        fluid_sim.pressureUpdate();
        ASSERT_NEAR(grid.divergence(1 * delta), 0.0, FLOAT_TOLERANCE);
        ASSERT_NEAR(grid.divergence(2 * delta), 0.0, FLOAT_TOLERANCE);

        // Mirror the configuration along the plane perpendicular to axis.
        std::swap(grid.cell(0 * delta), grid.cell(2 * delta));
        fluid_sim.solidCells().at(0) = sim::SolidCell(2 * delta, { 0, 0, 0 });
        grid.clearVelocities();
        for (int i = 1; i < 3; ++i) {
            if (axis == 0)
                grid.u(i * delta) = speed;
            else if (axis == 1)
                grid.v(i * delta) = speed;
            else if (axis == 2)
                grid.w(i * delta) = speed;
        }
        fluid_sim.pressureSolve();
        ASSERT_NEAR(fluid_sim.pressure(0 * delta), -4.0 / 29.0 * speed * scale, FLOAT_TOLERANCE);
        ASSERT_NEAR(fluid_sim.pressure(1 * delta), 5.0 / 29.0 * speed * scale, FLOAT_TOLERANCE);
        // Should be divergence-free after pressure update.
        fluid_sim.pressureUpdate();
        ASSERT_NEAR(grid.divergence(0 * delta), 0.0, FLOAT_TOLERANCE);
        ASSERT_NEAR(grid.divergence(1 * delta), 0.0, FLOAT_TOLERANCE);
    };

    for (int axis = 0; axis < 3; ++axis) {
        testCase(axis, 1, 1, 1, 1);
        testCase(axis, 0.1, 0.2, 0.3, 0.4);
        testCase(axis, 9, 8, 7, 6);
    }
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

TEST(Advect, ExactCellAmount)
{
    typedef sim::MACGrid<double> MACGrid;
    const double dx = 1.0;
    const double dt = 1.0;
    {
        MACGrid src({ 2, 1, 1 }, dx), dst({ 2, 1, 1 }, dx);
        src.cell(0, 0, 0) = 1;
        src.cell(1, 0, 0) = 2;
        for (int i = 0; i < 3; ++i)
            src.u(i, 0, 0) = -1.0;
        // Other velocity components are zero.
        advect(src, dt, dst);
        ASSERT_EQ(dst.cell(0, 0, 0), 2.0);
        ASSERT_EQ(dst.cell(1, 0, 0), 0.0);
    }
    {
        MACGrid src({ 1, 2, 1 }, dx), dst({ 1, 2, 1 }, dx);
        src.cell(0, 0, 0) = 1;
        src.cell(0, 1, 0) = 2;
        for (int i = 0; i < 3; ++i)
            src.v(0, i, 0) = -1.0;
        // Other velocity components are zero.
        advect(src, dt, dst);
        ASSERT_EQ(dst.cell(0, 0, 0), 2.0);
        ASSERT_EQ(dst.cell(0, 1, 0), 0.0);
    }
    {
        MACGrid src({ 1, 1, 2 }, dx), dst({ 1, 1, 2 }, dx);
        src.cell(0, 0, 0) = 1;
        src.cell(0, 0, 1) = 2;
        for (int i = 0; i < 3; ++i)
            src.w(0, 0, i) = -1.0;
        // Other velocity components are zero.
        advect(src, dt, dst);
        ASSERT_EQ(dst.cell(0, 0, 0), 2.0);
        ASSERT_EQ(dst.cell(0, 0, 1), 0.0);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                ASSERT_EQ(dst.u(i, 0, j), 0.0);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                ASSERT_EQ(dst.v(0, i, j), 0.0);
        ASSERT_EQ(dst.w(0, 0, 0), -1.0);
        ASSERT_EQ(dst.w(0, 0, 1), -1.0);
        ASSERT_EQ(dst.w(0, 0, 2), -0.5);
    }
}

TEST(Advect, ZeroFromEmptyCells)
{
    typedef sim::MACGrid<double> MACGrid;
    const double dx = 1.0;
    const double dt = 1.0;
    {
        MACGrid src({ 1, 1, 1 }, dx), dst({ 1, 1, 1 }, dx);
        src.cell(0, 0, 0) = 42;
        src.v(0, 0, 0) = -1;
        src.v(0, 1, 0) = -1;
        // Other velocity components are zero.
        advect(src, dt, dst);
        ASSERT_EQ(dst.cell(0, 0, 0), 0.0);
        ASSERT_EQ(dst.u(0, 0, 0), 0.0);
        ASSERT_EQ(dst.u(1, 0, 0), 0.0);
        ASSERT_EQ(dst.v(0, 0, 0), -1.0);
        ASSERT_EQ(dst.v(0, 1, 0), -0.5);
        ASSERT_EQ(dst.w(0, 0, 0), 0.0);
        ASSERT_EQ(dst.w(0, 0, 1), 0.0);
    }
}

namespace {
double &velocityComponent(sim::MACGrid<double> &mac_grid, int axis, const sim::GridIndex3 &idx)
{
    if (axis == 0)
        return mac_grid.u(idx);
    if (axis == 1)
        return mac_grid.v(idx);
    if (axis == 2)
        return mac_grid.w(idx);
    abort();
}
} // unnamed namespace

TEST(Advect, VelocitiesAreAdvected)
{
    typedef sim::MACGrid<double> MACGrid;
    const double dx = 1.0;
    const double dt = 1.0;
    for (int axis = 0; axis < 3; ++axis) {
        sim::GridIndex3 delta = { 0, 0, 0 };
        delta[axis] = 1;
        sim::GridSize3 size = sim::GridSize3(1, 1, 1) + 3 * delta;
        MACGrid src(size, dx), dst(size, dx);

        velocityComponent(src, axis, 0 * delta) = 0;
        velocityComponent(src, axis, 1 * delta) = -1;
        velocityComponent(src, axis, 2 * delta) = -2;
        velocityComponent(src, axis, 3 * delta) = -1;
        velocityComponent(src, axis, 4 * delta) = 0;

        sim::advect(src, dt, dst);

        ASSERT_EQ(velocityComponent(dst, axis, 0 * delta), 0);
        ASSERT_EQ(velocityComponent(dst, axis, 1 * delta), -1.5);
        ASSERT_EQ(velocityComponent(dst, axis, 2 * delta), -1);
        ASSERT_EQ(velocityComponent(dst, axis, 3 * delta), -0.5);
        ASSERT_EQ(velocityComponent(dst, axis, 4 * delta), 0);
    }
}
