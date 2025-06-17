#include "sim.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int xn =  256, yn = 128;
        const double xL = 2.0, yL = 1.0;
        const int T = 10;

        sim sim(xn, yn, xL, yL, T);
        sim.run(1000);
        sim.save_vtk("final");
    }
    Kokkos::finalize();
    return 0;
}