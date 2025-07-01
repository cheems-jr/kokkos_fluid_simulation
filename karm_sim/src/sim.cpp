#include "sim.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>
#include <algorithm>


namespace karman_vortex {

constexpr double Re = 70.0;  // Reynolds Number
constexpr double cfl = 0.2;  // CFL Number

/*SOR Pamameters*/
constexpr double omegap = 1.0;
constexpr int maxitp = 100;
constexpr double errorp = 0.0001;

/* set x-grid parameters*/
constexpr int i_1 = 96;
constexpr int i_2 = 106;

/* set y-grid parameters*/
constexpr int j_1 = 96;
constexpr int j_2 = 106;

/* set delta x,y,t*/
constexpr double dx = 1.0 / (i_2 - i_1);
constexpr double dy = 1.0 / (j_2 - j_1);
constexpr double dt = cfl * fmin(dx, dy);

sim::sim(int xn, int yn)
  : xn_(xn), yn_(yn),
    u_("u", xn+2, yn+2), v_("v", xn+2, yn+2), p_("p", xn+2, yn+2)
{
    Kokkos::deep_copy(u_, 1.0);
    Kokkos::deep_copy(v_, 0.0);
    Kokkos::deep_copy(p_, 0.0);
    sim::set_boundary_conditions_uv();
    sim::set_boundary_conditions_p();
}

void sim::run(double steps) {    
    
    for (int step = 0; step < steps; step++) {
        
        sim::poisson_solver();
        sim::velocity();
        sim::set_boundary_conditions_uv();

        if (step % 5 == 0) {
            save_vtk("output_step_" + std::to_string(step));
        }
    }
}

void sim::set_boundary_conditions_uv() {

    Kokkos::parallel_for("set_bc_u_y", yn_+1, KOKKOS_LAMBDA(int j) {
        u_(0, j) = 1.0;
        u_(1, j) = 1.0;
        v_(0, j) = 1.0;
        v_(1, j) = 1.0;
        u_(xn_, j) = 2.0 * u_(xn_-1, j) - u_(xn_-2, j);
        u_(xn_ + 1, j) = 2.0 * u_(xn_, j) - u_(xn_-1, j);
        v_(xn_, j) = 2.0 * v_(xn_-1, j) - v_(xn_-2, j);
        v_(xn_ + 1, j) = 2.0 * v_(xn_, j) - v_(xn_-1, j);
    });
    
    Kokkos::parallel_for("set_bc_v_x", xn_+1, KOKKOS_LAMBDA(int i) {
        v_(i, 1) = 2.0 * v_(i, 2) - v_(i, 3);
        v_(i, 0) = 2.0 * v_(i, 1) - v_(i, 2);
        u_(i, 1) = 2.0 * u_(i, 2) - u_(i, 3);
        u_(i, 0) = 2.0 * u_(i, 1) - u_(i, 2);
        v_(i, yn_) = 2.0 * v_(i, yn_-1) - v_(i, yn_-2);
        v_(i, yn_ + 1) = 2.0 * v_(i, yn_) - v_(i, yn_-1);
        u_(i, yn_) = 2.0 * u_(i, yn_-1) - u_(i, yn_-2);
        u_(i, yn_ + 1) = 2.0 * u_(i, yn_) - u_(i, yn_-1);
    });

    Kokkos::parallel_for("set_ic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({i_1, j_1}, {i_2, j_2}), 
    KOKKOS_LAMBDA(int i, int j) {
        u_(i, j) = 0.0;
        v_(i, j) = 0.0;
    });

    Kokkos::fence();
}

void sim::set_boundary_conditions_p() {

    Kokkos::parallel_for("p_x_bc", {1, xn_}, KOKKOS_LAMBDA(int i) {
        p_(i, 0) = 0.0;
        p_(i, yn_-1) = 0.0;
    });

    Kokkos::parallel_for("p_y_bc", {1, yn_}, KOKKOS_LAMBDA(int j) {
        p_(0, j) = 0.0;
        p_(xn_-1, j) = 0.0;
    });

    p_(i_1, j_1) = p_(i_1-1, j_1-1);
    p_(i_2, j_2) = p_(i_2+1, j_2+1);
    p_(i_1, j_2) = p_(i_1-1, j_2+1);
    p_(i_2, j_1) = p_(i_2+1, j_1-1);
    Kokkos::parallel_for("p_bloc_bc", {i_1+1, i_2}, KOKKOS_LAMBDA(int i) {
        p_(i, j_1) = p_(i, j_1-1);
        p_(i, j_2) = p_(i, j_2+1);
    });
    Kokkos::parallel_for("p_bloc_bc", {j_1+1, j_2}, KOKKOS_LAMBDA(int j) {
        p_(i_1, j) = p_(i_1-1, j);
        p_(i_2, j) = p_(i_2+1, j);
    });

    Kokkos::fence();
}

void sim::poisson_solver() {
    Kokkos::View<double**> rhs("rhs", xn_ + 2, yn_ + 2);
    Kokkos::deep_copy(rhs, 0.0);
    Kokkos::parallel_for("ps_rhs", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2,2}, {xn_-1, yn_-1}), 
    KOKKOS_LAMBDA(int i, int j) {
        if (in_square(i, j)) continue;

        double ux = u_(i+1, j) - u_(i-1, j) / (2.0 *dx_);
        double uy = u_(i, j+1) - u_(i, j-1) / (2.0 * dy_);
        double vx = v_(i+1, j) - v_(i-1, j) / (2.0 * dx_);
        double vy = v_(i, j+1) - v_(i, j-1) / (2.0 * dy_);
        rhs(i,j) = (ux + vy) / dt - (ux * ux + 2.0 * uy * vx + vy * vy);
    });

    for (int iter = 0; iter < maxitp; i++) {
        double res = 0.0;
        Kokkos::parallel_for("sor", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({2,2}, {xn_-1, yn_-1}), 
        KOKKOS_LAMBDA(int i, int j) {
            if (in_square(i, j)) continue;

            double dp = ((p_(i+1, j) + p_(i-1, j)) / (dx * dx) +
                        (p_(i, j+1) + p_(i, j-1)) / (dy * dy) - rhs(i,j)) /
                        (2.0 / (dx * dx) + 2.0 / (dy * dy)) - p(i,j);

            res += dp * dp;
            p_(i,j) += omegap * dp;
        }
        sim::set_boundary_conditions_p();
        res = sqrt(res / double(xn_, xy_)));
        if (res < errorp) break;
    }

    Kokkos::fence();
}

void sim::velocity() {
    Kokkos::View<double**> urhs("rhs", xn_ + 2, yn_ + 2);
    Kokkos::deep_copy(rhs, 0.0);
    Kokkos::View<double**> vrhs("rhs", xn_ + 2, yn_ + 2);
    Kokkos::deep_copy(rhs, 0.0);

    Kokkos::parallel_for("set_vel_rhs", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,0}, {xn_, yn_}), 
    KOKKOS_LAMBDA(int i, int j) {
        if (in_square(i, j)) continue;

        urhs(i,j) = -(p_(i+1,j) - p_(i-1,j)) / (2.0 * dx_) + 
                    (u_(i+1,j) - 2.0 *u_(i,j) + u_(i-1,j)) / (Re * dx * dx) +
                    (u_(i,j+1) - 2.0 *u_(i,j) + u_(i,j-1)) / (Re * dy * dy) -
                    KKSchemeX(u_(i,j), u_, i, j) - KKSchemeY(u_(i,j), u_, i, j);

        vrhs(i,j) = -(p_(i+1,j) - p_(i-1,j)) / (2.0 * dx_) + 
                    (v_(i+1,j) - 2.0 *v_(i,j) + v_(i-1,j)) / (Re * dx * dx) +
                    (v_(i,j+1) - 2.0 *v_(i,j) + v_(i,j-1)) / (Re * dy * dy) -
                    KKSchemeX(v_(i,j), v_, i, j) - KKSchemeY(v_(i,j), v_, i, j);


    });

    Kokkos::parallel_for("set_ic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,0}, {xn_, yn_}), 
    KOKKOS_LAMBDA(int i, int j) {
        if (in_square(i, j)) continue;

        u_(i,j) = u_(i,j) + dt * urhs(i,j);
        v_(i,j) = v_(i,j) + dt * vrhs(i,j);
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
    vtk_file << "DIMENSIONS " << xn_+1 << " " << yn_+1 << " 1\n";
    
    // Coordinates (2D plane)
    vtk_file << "POINTS " << (xn_+1)*(yn_+1) << " float\n";
    for (int j = 0; j <= yn_; ++j) {
        for (int i = 0; i <= xn_; ++i) {
            vtk_file << i*dx_ << " " << j*dy_ << " 0.0\n";
        }
    }
    
    // Velocity data (staggered to grid points)
    vtk_file << "POINT_DATA " << (xn_+1)*(yn_+1) << "\n";
    vtk_file << "VECTORS velocity float\n";
    for (int j = 0; j <= yn_; ++j) {
        for (int i = 0; i <= xn_; ++i) {
            // Interpolate u,v to grid points (simple average)
            float u = (i > 0) ? 0.5*(u_(i-1,j) + u_(i,j)) : u_(i,j);
            float v = (j > 0) ? 0.5*(v_(i,j-1) + v_(i,j)) : v_(i,j);
            vtk_file << u << " " << v << " 0.0\n";
        }
    }
    
    // Pressure data (cell-centered → interpolated to nodes)
    vtk_file << "SCALARS pressure float 1\n";
    vtk_file << "LOOKUP_TABLE default\n";
    for (int j = 0; j <= yn_; ++j) {
        for (int i = 0; i <= xn_; ++i) {
            // Average pressure from adjacent cells
            float p_val = 0.0;
            int count = 0;
            if (i > 0 && j > 0) { p_val += p_(i-1,j-1); count++; }
            if (i < xn_ && j > 0) { p_val += p_(i,j-1); count++; }
            if (i > 0 && j < yn_) { p_val += p_(i-1,j); count++; }
            if (i < xn_ && j < yn_) { p_val += p_(i,j); count++; }
            vtk_file << (count > 0 ? p_val/count : 0.0) << "\n";
        }
    }
    
    vtk_file.close();
}
}