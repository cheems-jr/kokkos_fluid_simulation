#include "sim.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>
#include <algorithm>

sim::sim(int xn, int yn, double xL, double yL, int T) 
  : xn_(xn), yn_(yn), dx_(xL/xn), dy_(yL/yn), T_(T),
    u_("u", xn+1, yn+1), v_("v", xn+1, yn+1), p_("p", xn, yn),
    u_star("u_star",xn+1, yn+1), v_star("v_star", xn+1, yn+1), p_star("p_star", xn, yn),
    u_adv("u_adv",xn+1, yn+1), v_adv("v_adv", xn+1, yn+1), div_u_star("div_u_star", xn+1, yn+1)
{
    double dt = 0.01;
    sim::set_initial_conditions();
    sim::set_boundary_conditions(dt);
}

double sim::calculate_cfl(double dt) {
    double max_vel = 0.0;
    Kokkos::parallel_reduce("cfl_calc", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_, yn_}),
        KOKKOS_LAMBDA(int i, int j, double& local_max) {
            double vel_mag = sqrt(u_(i,j)*u_(i,j) + v_(i,j)*v_(i,j));
            local_max = (vel_mag > local_max) ? vel_mag : local_max;
        }, Kokkos::Max<double>(max_vel));
    
    double cfl = max_vel * dt * (1.0/dx_ + 1.0/dy_);
    return cfl;
}

void sim::run(double steps) 
{    
    double dt = 0.05;
    double target_cfl = 0.1; // Conservative CFL target
    
    for (int step = 0; step < steps; step++) {
        // Calculate adaptive time step based on CFL condition
        double current_cfl = sim::calculate_cfl(dt);
        if (current_cfl > target_cfl) {
            //dt = dt * target_cfl / current_cfl;
            std::cout << "Reducing time step to " << dt << " (CFL = " << current_cfl << ")" << std::endl;
        }
        
        sim::advect(dt);
        double res = sim::solve_helmholtz(u_, u_adv, u_star, dt);
        res = sim::solve_helmholtz(v_, v_adv, v_star, dt);
        sim::divergence();
        res = sim::solve_poisson(dt);
        sim::update(dt);
        sim::set_boundary_conditions(dt);

        // --- DIAGNOSTICS ---
        // Sum of u_adv and v_adv
        double sum_u_adv = 0.0, sum_v_adv = 0.0;
        Kokkos::parallel_reduce("sum_u_adv", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_, yn_}),
            KOKKOS_LAMBDA(int i, int j, double& local_sum) {
                local_sum += u_adv(i,j);
            }, sum_u_adv);
        Kokkos::parallel_reduce("sum_v_adv", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_, yn_}),
            KOKKOS_LAMBDA(int i, int j, double& local_sum) {
                local_sum += v_adv(i,j);
            }, sum_v_adv);
        // Sum of divergence
        double sum_div = 0.0;
        Kokkos::parallel_reduce("sum_div", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_, yn_}),
            KOKKOS_LAMBDA(int i, int j, double& local_sum) {
                local_sum += div_u_star(i,j);
            }, sum_div);
        std::cout << "Step " << step << ": sum_u_adv = " << sum_u_adv << ", sum_v_adv = " << sum_v_adv << ", sum_div = " << sum_div << std::endl;
        // --- END DIAGNOSTICS ---

        if (step % 5 == 0) {
            std::cout << "Step " << step << ", CFL = " << current_cfl << ", dt = " << dt << std::endl;
            save_vtk("output_step_" + std::to_string(step));
        }
    }
}

void sim::set_initial_conditions() {
    Kokkos::parallel_for("set_ic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,0}, {xn_, yn_}), 
    KOKKOS_LAMBDA(int i, int j) {
        double yL = yn_ * dy_;
        double y = (j+0.5)*dy_;
        u_(i,j) = 0.1;  // Reduced initial velocity to match boundary conditions
        v_(i,j) = 0.0;  // Initialize v-velocity to zero
        p_(i,j) = 0.0;
    });
    Kokkos::fence(); 
}

void sim::set_boundary_conditions(double dt) {

    Kokkos::parallel_for("set_bc_u_y", yn_+1, KOKKOS_LAMBDA(int j) {
        u_(0, j) = 0.0;
        u_(xn_, j) = 0.0;
    });


    Kokkos::parallel_for("set_bc_v_y", yn_+1, KOKKOS_LAMBDA(int j) {
        v_(0, j) = 0.0;
        v_(xn_, j) = 0.0;
    });
    
    // v-velocity boundaries (x-faces)
    Kokkos::parallel_for("set_bc_v_x", xn_+1, KOKKOS_LAMBDA(int i) {
        v_(i, 0) = 0.0;
        v_(i, yn_) = 0.0;
    });

    // u-velocity boundaries (y-faces)
    Kokkos::parallel_for("set_bc_u_x", xn_+1, KOKKOS_LAMBDA(int i) {
        u_(i, 0) = 0.0;
        u_(i, yn_) = 0.0;
    });

    // Pressure boundaries (cell-centered)
    Kokkos::parallel_for("p_x_bc", xn_, KOKKOS_LAMBDA(int i) {
        p_(i, 0) = p_(i, 1);
        p_(i, yn_-1) = p_(i, yn_-2);
    });

    Kokkos::parallel_for("p_y_bc", yn_, KOKKOS_LAMBDA(int j) {
        p_(0, j) = p_(1, j);  // Inlet: high pressure (drives flow)
        p_(xn_-1, j) = 0.0;  // Outlet: low pressure (allows outflow)
    });
    Kokkos::fence();
}

void sim::set_star_boundary_conditions(double dt) {
    // u_star boundaries (x-faces)
    Kokkos::parallel_for("set_bc_u_star_y", yn_+1, KOKKOS_LAMBDA(int j) {
        u_star(0, j) = 0.0;
        u_star(xn_, j) = 0.0;
    });

    // v_star boundaries (y-faces) 
    Kokkos::parallel_for("set_bc_v_star_y", yn_+1, KOKKOS_LAMBDA(int j) {
        v_star(0, j) = 0.0;
        v_star(xn_, j) = 0.0;
    });
    
    // v_star boundaries (x-faces)
    Kokkos::parallel_for("set_bc_v_star_x", xn_+1, KOKKOS_LAMBDA(int i) {
        v_star(i, 0) = 0.0;
        v_star(i, yn_) = 0.0;
    });

    // u_star boundaries (y-faces)
    Kokkos::parallel_for("set_bc_u_star_x", xn_+1, KOKKOS_LAMBDA(int i) {
        u_star(i, 0) = 0.0;
        u_star(i, yn_) = 0.0;
    });

    // p_star boundaries (cell-centered)
    Kokkos::parallel_for("p_star_x_bc", xn_, KOKKOS_LAMBDA(int i) {
        p_star(i, 0) = p_star(i, 1);
        p_star(i, yn_-1) = p_star(i, yn_-2);
    });

    Kokkos::parallel_for("p_star_y_bc", yn_, KOKKOS_LAMBDA(int j) {
        p_star(0, j) = p_star(1, j);  // Inlet: high pressure (drives flow)
        p_star(xn_-1, j) = 0.0;  // Outlet: low pressure (allows outflow)
    });
    Kokkos::fence();
}

void sim::advect(double dt) {
    double max_adv = 0.0;
    Kokkos::parallel_for("advection_diffusion", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>( {1,1}, {xn_, yn_}),
        KOKKOS_LAMBDA(int i, int j) {
            if (u_(i,j) > 0) {
                u_adv(i,j) = - u_(i, j) / dx_ * (u_(i, j) - u_(i-1, j));
            } else {
                u_adv(i,j) = - u_(i, j) / dx_ * (u_(i+1, j) - u_(i, j));
            }
            // Upwind in y for u (cross-term)
            if (v_at_p(i,j) > 0) {
                u_adv(i,j) += v_at_p(i,j) / dy_ * (u_(i, j) - u_(i, j-1));
            } else {
                u_adv(i,j) += v_at_p(i,j) / dy_ * (u_(i, j+1) - u_(i, j));
            }

            // Upwind in x for v
            if (u_at_p(i, j) > 0) {
                v_adv(i,j) = -u_at_p(i, j) / dx_ * (v_(i, j) - v_(i-1, j));
            } else {
                v_adv(i,j) = -u_at_p(i, j) / dx_ * (v_(i, j+1) - v_(i, j));
            }
            // Upwind in y for v (cross-term)
            if (v_(i,j) > 0) {
                v_adv(i,j) += v_(i,j) / dy_ * (v_(i, j) - v_(i, j-1));
            } else {
                v_adv(i,j) += v_(i,j) / dy_ * (v_(i, j+1) - v_(i, j));
            }
        });

        
    max_adv = sim::max_val(u_adv);
    std::cout << "Max advection: " << max_adv << std::endl;
    
    // Add debug output for velocity ranges
    double max_u = sim::max_val(u_);
    double max_v = sim::max_val(v_);
    std::cout << "Max u: " << max_u << ", Max v: " << max_v << std::endl;
    
    Kokkos::fence();
}

double sim::solve_helmholtz(Kokkos::View<double**> &u, Kokkos::View<double**> &u_adv, Kokkos::View<double**> &u_star, double dt) {
    double residual = 0.0;
    double nu = 0.01;  // Increased viscosity from 0.01 to 0.1 for better energy dissipation
    for (int i = 0; i < 200; i++) {

        Kokkos::parallel_for("diffusion", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_, yn_}), KOKKOS_LAMBDA(int i, int j) {
            double laplace = (u_star(i-1,j) + u_star(i+1, j))/ (dx_*dx_) + (u_star(i,j-1) + u_star(i,j+1))/ (dy_*dy_);
            double rhs = u_(i,j) - dt * u_adv(i,j);

            u_star(i,j) = (rhs + dt * nu * laplace) / (1 - dt * nu * (2/(dx_*dx_) + 2/(dy_*dy_)));
            u_adv(i,j) = 0.0;
        });
        if (i % 5 == 0) {
            residual = sim::residual(u_star);
            if (residual < 1e-6) {
                return residual;
            }

        }
        
        sim::set_star_boundary_conditions(dt);
        
    }

    return residual;  
}

void sim::divergence() {
    Kokkos::parallel_for("divergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_, yn_}), KOKKOS_LAMBDA(int i, int j) {
        div_u_star(i,j) = (u_star(i+1,j) - u_star(i-1,j))/(2*dx_) + (v_star(i,j+1) - v_star(i,j-1))/(2*dy_);
    });
    Kokkos::fence();
}

double sim::solve_poisson(double dt) {
    double rho = 1.0;
    double residual = 0.0;
    for (int i = 0; i < 200; i++) {

        Kokkos::parallel_for("poisson", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_, yn_}), KOKKOS_LAMBDA(int i, int j) {
            double laplace = (p_(i-1,j) + p_(i+1, j))/ (dx_*dx_) + (p_(i,j-1) + p_(i,j+1))/ (dy_*dy_);
            double rhs = rho / dt * div_u_star(i,j);

            p_star(i,j) = (laplace - rhs) / (2/(dx_*dx_) + 2/(dy_*dy_));
        });
        if (i % 5 == 0) {
            residual = sim::residual(p_star);
            if (residual < 1e-6) {
                return residual;
            }
        }

        sim::set_star_boundary_conditions(dt);
        
    }
    Kokkos::deep_copy(p_, p_star);
    return residual;
}

void sim::update(double dt) {
    double rho = 1.0;
    Kokkos::parallel_for("velocity_projection", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_-1, yn_-1}), KOKKOS_LAMBDA(int i, int j) {
        u_(i,j) = u_star(i,j) - dt/rho * (p_star(i+1,j) - p_star(i-1,j)) / (2*dx_);
        v_(i,j) = v_star(i,j) - dt/rho * (p_star(i,j+1) - p_star(i,j-1)) / (2*dy_);
    });
    Kokkos::fence();
}

double sim::max_val(Kokkos::View<double**> &u) {
    double max = 0.0;
    Kokkos::parallel_reduce("max_val", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_, yn_}), KOKKOS_LAMBDA(int i, int j, double& local_max) {
        double val = u(i,j);
        local_max = (val > local_max) ? val : local_max;
    }, max);
    return max;
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

double sim::residual(Kokkos::View<double**> &p) 
{
    double residual;
    Kokkos::Max<double> reducer(residual);

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {xn_, yn_}),
        KOKKOS_LAMBDA(int i, int j, double& local_max) {
            // For now, just use the absolute value as a simple residual measure
            double val = fabs(p(i,j));
            local_max = (val > local_max) ? val : local_max;
        },
        reducer
    );
    Kokkos::fence();
    return residual;
}