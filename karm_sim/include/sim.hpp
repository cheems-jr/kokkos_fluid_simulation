#pragma once

#include <Kokkos_Core.hpp>

class sim
{
private:
    int xn_, yn_, T_;
    double dx_, dy_;
    Kokkos::View<double**> u_, v_, p_;

    void set_boundary_conditions_uv();
    void set_boundary_conditions_p();
    void set_initial_conditions();
    void velocity();
    void poisson_solver();
    void save_vtk(const std::string& filename);

    KOKKOS_INLINE_FUNCTION
    bool in_square(int i, int j){
        if (i_1 < i && i < i_2 && j_1 < j && j < j_2){
            return true;
        }
        return false;
    }
    KOKKOS_INLINE_FUNCTION
    double KKSchemeX(const double u, const Kokkos::View<double**> &f, const int i,
                        const int j) {
    return (u * (-f(i + 2,j) + 8.0 * (f(i + 1,j) - f(i - 1,j)) + f(i - 2,j)) / (12.0 * dx) +
            abs(u) * (f(i + 2,j) - 4.0 * (f(i + 1,j) + f(i - 1,j)) +
            6.0 * f(i,j) + f(i - 2,j)) / (4.0 * dx));
    }
    KOKKOS_INLINE_FUNCTION
    double KKSchemeY(const double v, const Kokkos::View<double**> &f, const int i,
                            const int j) {
    return (v * (-f(i,j + 2) + 8.0 * (f(i,j + 1) - f(i,j - 1)) + f(i,j - 2)) / (12.0 * dy) +
            abs(v) * (f(i,j + 2) - 4.0 * (f(i,j + 1) + f(i,j - 1)) +
            6.0 * f(i,j) + f(i,j - 2)) / (4.0 * dy));
    }

public:
    sim(int xn, int yn);
    void run(double steps);
    void save_vtk(const std::string& filename);
};

