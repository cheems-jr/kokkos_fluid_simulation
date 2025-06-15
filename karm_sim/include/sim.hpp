#pragma once
#include <Kokkos_Core.hpp>

class sim
{
private:
    int xn_, yn_;
    double dx_, dy_;
    Kokkos::View<double**> u_, v_, p_;

    void advect(double dt);
    void solve_pressure(double dt);
    void set_boundary_conditions();
public:
    sim(int xn, int yn, double xL, double yL);
    void run(int steps);
    void save_vtk(const std::string& filename);
};
