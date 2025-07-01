#include "sim.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int xn = 401, yn = 201;

        sim sim(xn, yn);
        sim.save_vtk("initial");
        sim.run(10000);
        sim.save_vtk("final");
    }
    Kokkos::finalize();
    return 0;   
}