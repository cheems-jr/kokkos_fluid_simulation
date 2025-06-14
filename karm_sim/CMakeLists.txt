cmake_minimum_required(VERSION 3.12)

project(karman_vortex)

find_package(Kokkos REQUIRED)

add_executable(vortex_sim
    src/main.cpp
    src/sim.cpp
)

target_link_libraries(vortex_sim Kokkos::kokkos)
target_compile_features(vortex_sim PRIVATE cxx_std_17)

target_include_directories(vortex_sim PRIVATE include)