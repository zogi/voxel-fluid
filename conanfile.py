from conans import ConanFile, CMake


class FluidConan(ConanFile):
    name = "Fluid"
    version = "0.1"
    settings = "os", "compiler", "build_type", "arch"
    options = {}
    requires = (
        "glew/2.0.0@coding3d/stable",
        "spdlog/0.13.0@memsharded/stable",
        "gtest/1.8.0@lasote/stable",
    )
    default_options = (
        "glew:shared=False",
        "gtest:shared=False",
    )
    generators = "cmake"

    def build(self):
        cmake = CMake(self)
        cmake.configure(source_dir=self.source_folder)
        cmake.build()
