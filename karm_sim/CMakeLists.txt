cmake_minimum_required(VERSION 3.12)

set(CMAKE_C_COMPILER "/usr/local/bin/gcc-13")
set(CMAKE_CXX_COMPILER "/usr/local/bin/gcc-13")

project(karman_vortex LANGUAGES CXX)

include_directories(
    karm_sim/include
    /usr/local/include
    /usr/local/Cellar/gcc@13/13.4.0/include/c++/13
    /usr/local/Cellar/gcc@13/13.4.0/include/c++/13/x86_64-apple-darwin23
)

find_package(Kokkos REQUIRED)

add_executable(karm_sim
    src/main.cpp
    src/sim.cpp
)

target_link_libraries(karm_sim 
    PRIVATE 
    Kokkos::kokkos
    stdc++
)
target_compile_features(karm_sim PRIVATE cxx_std_17)

