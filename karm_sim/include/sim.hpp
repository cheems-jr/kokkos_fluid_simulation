#pragma once
#include <Kokkos_Core.hpp>

class sim
{
private:
    int xn_, yn_, T_;
    double dx_, dy_;
    Kokkos::View<double**> u_, v_, p_;
    Kokkos::View<double**> u_star, v_star, p_star;

    void advect(double dt);
    void solve_pressure(double dt);
    void set_boundary_conditions();
    void set_pressure_bc();
    double residual();
    void set_initial_conditions();
    void init_grids();
    KOKKOS_INLINE_FUNCTION
    double p_at_u(int i, int j) {
        return (p_(i,j) + p_(i+1, j)) /2;
    }
    KOKKOS_INLINE_FUNCTION
    double u_at_p(int i, int j) {
        return (u_(i-1,j) + u_(i, j)) /2;
    }
    KOKKOS_INLINE_FUNCTION
    double p_at_v(int i, int j){
        return (p_(i,j) + p_(i, j+1)) /2;
    }
    KOKKOS_INLINE_FUNCTION
    double v_at_p(int i, int j){
        return (v_(i,j-1) + v_(i, j)) /2;
    }

public:
    sim(int xn, int yn, double xL, double yL, int T);
    void run(int steps);
    void save_vtk(const std::string& filename);
};
