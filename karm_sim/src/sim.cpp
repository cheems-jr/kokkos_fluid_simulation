#include "sim.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <algorithm>

sim::sim(int xn, int yn)
  : xn_(xn), yn_(yn), dx_(1.0 / (i_2 - i_1)), dy_(1.0 / (j_2 - j_1)),
    u_("u", xn+2, yn+2), v_("v", xn+2, yn+2), p_("p", xn+2, yn+2)
{
    Kokkos::deep_copy(u_, 1.0);
    Kokkos::deep_copy(v_, 0.0);
    Kokkos::deep_copy(p_, 0.0);
    Kokkos::fence();
    set_boundary_conditions_uv();
    Kokkos::fence();
    set_boundary_conditions_p();
    Kokkos::fence();
}

void sim::run(double steps) {    
    
    for (int step = 0; step < steps; step++) {
        poisson_solver();
        Kokkos::fence();
        velocity();
        Kokkos::fence();
        set_boundary_conditions_uv();
        Kokkos::fence();
        std::cout << "step: " << step << std::endl;
        std::ostringstream ss;
        ss << "../data/" << std::setw(6) << std::setfill('0') << step << ".csv";
        save_pressure_matrix_csv(ss.str());

    }
}

void sim::set_boundary_conditions_uv() {

    Kokkos::parallel_for("set_bc_u_y", Kokkos::RangePolicy<>(1, yn_+1), KOKKOS_LAMBDA(int j) {
        u_(0, j) = 1.0;
        u_(1, j) = 1.0;
        v_(0, j) = 1.0;
        v_(1, j) = 1.0;
        u_(xn_, j) = 2.0 * u_(xn_-1, j) - u_(xn_-2, j);
        u_(xn_ + 1, j) = 2.0 * u_(xn_, j) - u_(xn_-1, j);
        v_(xn_, j) = 2.0 * v_(xn_-1, j) - v_(xn_-2, j);
        v_(xn_ + 1, j) = 2.0 * v_(xn_, j) - v_(xn_-1, j);
    });
    Kokkos::fence();
    Kokkos::parallel_for("set_bc_v_x", Kokkos::RangePolicy<>(1, xn_+1), KOKKOS_LAMBDA(int i) {
        v_(i, 1) = 2.0 * v_(i, 2) - v_(i, 3);
        v_(i, 0) = 2.0 * v_(i, 1) - v_(i, 2);
        u_(i, 1) = 2.0 * u_(i, 2) - u_(i, 3);
        u_(i, 0) = 2.0 * u_(i, 1) - u_(i, 2);
        v_(i, yn_) = 2.0 * v_(i, yn_-1) - v_(i, yn_-2);
        v_(i, yn_ + 1) = 2.0 * v_(i, yn_) - v_(i, yn_-1);
        u_(i, yn_) = 2.0 * u_(i, yn_-1) - u_(i, yn_-2);
        u_(i, yn_ + 1) = 2.0 * u_(i, yn_) - u_(i, yn_-1);
    });
    Kokkos::fence();
    Kokkos::parallel_for("set_ic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i_1, j_1}, {i_2+1, j_2+1}), 
    KOKKOS_LAMBDA(int i, int j) {
        u_(i, j) = 0.0;
        v_(i, j) = 0.0;
    });
    
    Kokkos::fence();
}

void sim::set_boundary_conditions_p() {

    Kokkos::parallel_for("p_x_bc", Kokkos::RangePolicy<>(1, xn_+1), KOKKOS_LAMBDA(int i) {
        p_(i, 0) = 0.0;
        p_(i, yn_-1) = 0.0;
    });
    Kokkos::fence();
    Kokkos::parallel_for("p_y_bc", Kokkos::RangePolicy<>(1, yn_+1), KOKKOS_LAMBDA(int j) {
        p_(0, j) = 0.0;
        p_(xn_-1, j) = 0.0;
    });
    Kokkos::fence();
    p_(i_1, j_1) = p_(i_1-1, j_1-1);
    p_(i_2, j_2) = p_(i_2+1, j_2+1);
    p_(i_1, j_2) = p_(i_1-1, j_2+1);
    p_(i_2, j_1) = p_(i_2+1, j_1-1);
    Kokkos::parallel_for("p_bloc_bc", Kokkos::RangePolicy<>(i_1+1, i_2), KOKKOS_LAMBDA(int i) {
        p_(i, j_1) = p_(i, j_1-1);
        p_(i, j_2) = p_(i, j_2+1);
    });
    Kokkos::fence();
    Kokkos::parallel_for("p_bloc_bc", Kokkos::RangePolicy<>(j_1+1, j_2), KOKKOS_LAMBDA(int j) {
        p_(i_1, j) = p_(i_1-1, j);
        p_(i_2, j) = p_(i_2+1, j);
    });

    Kokkos::fence();
}

void sim::poisson_solver() {
    Kokkos::View<double**> rhs("rhs", xn_ + 2, yn_ + 2);
    Kokkos::deep_copy(rhs, 0.0);
    Kokkos::fence();
    Kokkos::parallel_for("ps_rhs", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2,2}, {xn_, yn_}), 
    KOKKOS_LAMBDA(int i, int j) {
        if (!in_square(i, j)) {

            double ux = (u_(i+1, j) - u_(i-1, j)) / (2.0 * dx_);
            double uy = (u_(i, j+1) - u_(i, j-1)) / (2.0 * dy_);
            double vx = (v_(i+1, j) - v_(i-1, j)) / (2.0 * dx_);
            double vy = (v_(i, j+1) - v_(i, j-1)) / (2.0 * dy_);
            rhs(i,j) = (ux + vy) / dt - (ux * ux + 2.0 * uy * vx + vy * vy);

        }
    });
    Kokkos::fence();

    for (int iter = 0; iter < maxitp; iter++) {
        double res = 0.0;
        Kokkos::parallel_for("sor", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2,2}, {xn_, yn_}), 
        KOKKOS_LAMBDA(int i, int j) {
            if (!in_square(i, j)) {

                double dp = ((p_(i+1, j) + p_(i-1, j)) / (dx_ * dx_) +
                            (p_(i, j+1) + p_(i, j-1)) / (dy_ * dy_) - rhs(i,j)) /
                            (2.0 / (dx_ * dx_) + 2.0 / (dy_ * dy_)) - p_(i,j);
                p_(i,j) += omegap * dp;
            }
        });
        Kokkos::fence();
        Kokkos::parallel_reduce("sor", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2,2}, {xn_, yn_}), 
        KOKKOS_LAMBDA(int i, int j, double& local_res) {
            if (!in_square(i, j)) {

                double dp = ((p_(i+1, j) + p_(i-1, j)) / (dx_ * dx_) +
                            (p_(i, j+1) + p_(i, j-1)) / (dy_ * dy_) - rhs(i,j)) /
                            (2.0 / (dx_ * dx_) + 2.0 / (dy_ * dy_)) - p_(i,j);

                local_res += dp * dp;
            }
        }, res);
        Kokkos::fence();
        sim::set_boundary_conditions_p();
        res = sqrt(res / double(xn_ * yn_));
        if (res < errorp) break;
    }

    Kokkos::fence();
}

void sim::velocity() {
    Kokkos::View<double**> urhs("rhs", xn_ + 2, yn_ + 2);
    Kokkos::deep_copy(urhs, 0.0);
    Kokkos::View<double**> vrhs("rhs", xn_ + 2, yn_ + 2);
    Kokkos::deep_copy(vrhs, 0.0);

    Kokkos::fence();

    Kokkos::parallel_for("set_vel_rhs", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2,2}, {xn_, yn_}), 
    KOKKOS_LAMBDA(int i, int j) {
        if (!in_square(i, j)) {

            urhs(i,j) = -(p_(i+1,j) - p_(i-1,j)) / (2.0 * dx_) + 
                        (u_(i+1,j) - 2.0 *u_(i,j) + u_(i-1,j)) / (Re * dx_ * dx_) +
                        (u_(i,j+1) - 2.0 *u_(i,j) + u_(i,j-1)) / (Re * dy_ * dy_) -
                        KKSchemeX(u_(i,j), u_, i, j) - KKSchemeY(u_(i,j), u_, i, j);

            vrhs(i,j) = -(p_(i+1,j) - p_(i-1,j)) / (2.0 * dx_) + 
                        (v_(i+1,j) - 2.0 *v_(i,j) + v_(i-1,j)) / (Re * dx_ * dx_) +
                        (v_(i,j+1) - 2.0 *v_(i,j) + v_(i,j-1)) / (Re * dy_ * dy_) -
                        KKSchemeX(v_(i,j), v_, i, j) - KKSchemeY(v_(i,j), v_, i, j);

        }
    });

    Kokkos::fence();

    Kokkos::parallel_for("set_ic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2,2}, {xn_, yn_}), 
    KOKKOS_LAMBDA(int i, int j) {
        if (!in_square(i, j)) {

            u_(i,j) = u_(i,j) + dt * urhs(i,j);
            v_(i,j) = v_(i,j) + dt * vrhs(i,j);

        }
    });

    Kokkos::fence();

}

void sim::save_vtk(const std::string& filename) {
    std::ofstream vtk_file(filename + ".vtk");
    
    // VTK header
    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << "Fluid Simulation Data\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET STRUCTURED_GRID\n";
    
    // Grid dimensions (2D)
    vtk_file << "DIMENSIONS " << xn_+2 << " " << yn_+2 << " 1\n";
    
    // Coordinates (2D plane)
    vtk_file << "POINTS " << (xn_+2)*(yn_+2) << " float\n";
    for (int j = 0; j < yn_+2; ++j) {
        for (int i = 0; i < xn_+2; ++i) {
            vtk_file << i*dx_ << " " << j*dy_ << " 0.0\n";
        }
    }
    
    // Velocity data (staggered to grid points)
    vtk_file << "POINT_DATA " << (xn_+2)*(yn_+2) << "\n";
    vtk_file << "VECTORS velocity float\n";
    for (int j = 0; j < yn_+2; ++j) {
        for (int i = 0; i < xn_+2; ++i) {
            // Interpolate u,v to grid points (simple average)
            float u = (i > 0) ? 0.5f*(u_(i-1,j) + u_(i,j)) : u_(i,j);
            float v = (j > 0) ? 0.5f*(v_(i,j-1) + v_(i,j)) : v_(i,j);
            vtk_file << u << " " << v << " 0.0\n";
        }
    }
    
    // Pressure data (cell-centered → interpolated to nodes)
    vtk_file << "SCALARS pressure float 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (int j = 0; j < yn_+2; ++j) {
        for (int i = 0; i < xn_+2; ++i) {
            // Average pressure from adjacent cells
            float p_val = 0.0f;
            int count = 0;
            if (i > 0 && j > 0) { p_val += p_(i-1,j-1); count++; }
            if (i < xn_+1 && j > 0) { p_val += p_(i,j-1); count++; }
            if (i > 0 && j < yn_+1) { p_val += p_(i-1,j); count++; }
            if (i < xn_+1 && j < yn_+1) { p_val += p_(i,j); count++; }
            vtk_file << (count > 0 ? p_val/count : 0.0f) << "\n";
        }
    }
    
    vtk_file.close();
}

void sim::save_pressure_matrix_csv(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        exit(-1);
    }
    for (int i = yn_; i >= 1; i--) {
        for (int j = 1; j <= xn_; j++) {
            ofs << p_(i, j);
            ofs << ((j == xn_) ? "\n" : ",");
        }
    }
}