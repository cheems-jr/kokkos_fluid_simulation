#pragma once

#include <Kokkos_Core.hpp>
#include <cmath>

class sim
{
private:
    int xn_, yn_, T_;
    double dx_, dy_;
    Kokkos::View<double**> u_, v_, p_;
    
    static constexpr int i_1 = 96;
    static constexpr int i_2 = 106;
    static constexpr int j_1 = 96;
    static constexpr int j_2 = 106;
    static constexpr double dx = 1.0 / (i_2 - i_1);
    static constexpr double dy = 1.0 / (j_2 - j_1);
    static constexpr double Re = 70.0;
    static constexpr double cfl = 0.2; 
    static constexpr double omegap = 1.0;
    static constexpr int maxitp = 100;
    static constexpr double errorp = 0.0001;
    // Use ternary operator for constexpr min
    static constexpr double dt = cfl * (dx < dy ? dx : dy);

    void set_boundary_conditions_uv();
    void set_boundary_conditions_p();
    void set_initial_conditions();
    void velocity();
    void poisson_solver();
    

    KOKKOS_INLINE_FUNCTION
    bool in_square(int i, int j) const {
        return (i_1 < i && i < i_2 && j_1 < j && j < j_2);
    }
    KOKKOS_INLINE_FUNCTION
    double KKSchemeX(const double u, const Kokkos::View<double**> &f, const int i,
                        const int j) const {
        return (u * (-f(i + 2,j) + 8.0 * (f(i + 1,j) - f(i - 1,j)) + f(i - 2,j)) / (12.0 * dx) +
                std::abs(u) * (f(i + 2,j) - 4.0 * (f(i + 1,j) + f(i - 1,j)) +
                6.0 * f(i,j) + f(i - 2,j)) / (4.0 * dx));
    }
    KOKKOS_INLINE_FUNCTION
    double KKSchemeY(const double v, const Kokkos::View<double**> &f, const int i,
                            const int j) const {
        return (v * (-f(i,j + 2) + 8.0 * (f(i,j + 1) - f(i,j - 1)) + f(i,j - 2)) / (12.0 * dy) +
                std::abs(v) * (f(i,j + 2) - 4.0 * (f(i,j + 1) + f(i,j - 1)) +
                6.0 * f(i,j) + f(i,j - 2)) / (4.0 * dy));
    }

public:
    sim(int xn, int yn);
    void run(double steps);
    void save_vtk(const std::string& filename);
    void save_pressure_matrix_csv(const std::string& filename);
};

