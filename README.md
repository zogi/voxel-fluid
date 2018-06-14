Voxel-style fluid simulation
============================

This is a low-res Eulerian fluid simulation experiment I made in my free time mainly out of curiosity and because I love Minecraft.

The fluid simulation uses [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)'s sparse
conjugate gradient solver for pressure projection and a basic semi-Lagrangian method for advection.
The voxels are rendered by tracing through a 3D density texture using DDA-style traversal computing the
effect of volume absorption and reflection at air-fluid interfaces.
Emtpy-space skipping and hierarchical data storage schemes are not implemented at this point.
A triangular PDF dither (such as the one [used in INSIDE](https://www.youtube.com/watch?v=RdN06E6Xn9E&t=1259))
is applied to the gamma corrected output to prevent banding.

## Running the demo

The demo needs OpenGL 4.3+ and expects the *shader* folder to be present in the working directory.

F3 toggles overlay display.

## How to build

First of all clone with --recursive, because this project uses submodules.

After that just build with cmake. If you want to build the tests, either have google test available to cmake,
or use conan for building.

I have developed this on Windows using an AMD GPU, but I have tested it a couple of times on Nvidia hardware too.
However I haven't tested the project on Linux.
