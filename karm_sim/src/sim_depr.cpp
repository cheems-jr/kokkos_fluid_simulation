#include "sim.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>

sim::sim(int xn, int yn, double xL, double yL, int T) 
  : xn_(xn), yn_(yn), dx_(xL/xn), dy_(yL/yn), T_(T),
    u_("u", xn+1, yn), v_("v", xn, yn+1), p_("p", xn, yn),
    u_star("u_star",xn+1, yn), v_star("v_star", xn, yn+1), p_star("p_star", xn, yn)
{
    sim::init_grids();
    sim::set_initial_conditions();
    sim::set_boundary_conditions();
    sim::set_pressure_bc();
}


void sim::init_grids() 
{
    Kokkos::parallel_for("init_p",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {xn_, yn_}),
        KOKKOS_LAMBDA(int i, int j) {
            p_(i,j) = 0.0; 
            p_star(i,j) = 0.0;
    });
    Kokkos::parallel_for("init_u",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {xn_+1, yn_}),
        KOKKOS_LAMBDA(int i, int j) {
            u_(i,j) = 0.0;
            u_star(i,j) = 0.0;
    });
    Kokkos::parallel_for("init_v",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {xn_, yn_+1}),
        KOKKOS_LAMBDA(int i, int j) {
            v_(i,j) = 0.0;
            v_star(i,j) = 0.0;
    });
    Kokkos::fence(); 
}

void sim::set_initial_conditions() 
{
    Kokkos::parallel_for("set_ic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,0}, {xn_, yn_}), 
    KOKKOS_LAMBDA(int i, int j) {
        double yL = yn_ * dy_;
        double y = (j+0.5)*dy_;
        u_(i,j) = 1.0 * y*(yL-y)/(yL*yL);
    });
    Kokkos::fence(); 
}





void sim::set_boundary_conditions() 
{
    Kokkos::parallel_for("set_bc_u_y", yn_, KOKKOS_LAMBDA(int j) {
        double yL = yn_ * dy_;
        double y = (j+0.5)*dy_;
        u_(0,j) = 6.0 * y*(yL-y)/(yL*yL);
        u_(xn_,j) = u_(xn_-1, j);
    });

    Kokkos::parallel_for("set_bc_v_y", yn_+1, KOKKOS_LAMBDA(int j) {
        v_(0, j) = 0.0;
        v_(xn_-1, j) = v_(xn_-2, j);
    });
    
    Kokkos::parallel_for("set_bc_v_x", xn_, KOKKOS_LAMBDA(int i) {
        v_(i, 0) = 0.0;
        v_(i, yn_) = 0.0;
    });
    Kokkos::parallel_for("set_bc_u_x", xn_+1, KOKKOS_LAMBDA(int i) {
        u_(i, 0) = 0.0;
        u_(i, yn_-1) = 0.0;
    });
    Kokkos::fence(); 
}

void sim::set_pressure_bc() 
{
    Kokkos::parallel_for("p_x_bc", xn_, KOKKOS_LAMBDA(int i) {
        p_(i, 0) = p_(i, 1);
        p_(i, yn_-1) = p_(i, yn_-2);
    });

    Kokkos::parallel_for("p_y_bc", yn_, KOKKOS_LAMBDA(int j) {
        p_(0, j) = p_(1, j);
        p_(xn_-1, j) = p_(xn_-2, j);
    });
    Kokkos::fence(); 
}

double sim::residual() 
{
    double residual;
    Kokkos::Max<double> reducer(residual);

    Kokkos::parallel_reduce(
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1, 1}, {xn_-1, yn_-1}),
        KOKKOS_LAMBDA(int i, int j, double& local_max) {
            double diff = fabs(p_star(i,j) - p_(i,j));
            local_max = (diff > local_max) ? diff : local_max;
        },
        reducer
    );
    Kokkos::fence();
    return residual;
}





void sim::advect(double dt) 
{
    double nu = 10.0;

    Kokkos::parallel_for("advection_diffusion", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>( {1,1}, {xn_-1, yn_-1}),
        KOKKOS_LAMBDA(int i, int j) {
            // Central differencing for advection (more stable)
            double u_adv = 0.5 * u_(i, j) * dt/dx_ * (u_(i+1, j) - u_(i-1, j)) 
                    + 0.5 * v_at_p(i,j) * dt/dy_ * (u_(i, j+1) - u_(i, j-1));
            double v_adv = 0.5 * u_at_p(i, j) * dt/dx_ * (v_(i+1, j) - v_(i-1, j)) 
                    + 0.5 * v_(i,j) * dt/dy_ * (v_(i, j+1) - v_(i, j-1));

            u_star(i,j) = u_(i,j) + dt * (u_adv);
            v_star(i,j) = v_(i,j) + dt * (v_adv);
    });

    Kokkos::deep_copy(u_, u_star);
    Kokkos::deep_copy(v_, v_star);
    sim::set_boundary_conditions();
    Kokkos::fence(); 
}


void sim::diffuse() 
{
    double nu = 10.0;
    for (int i = 0; i < 200; i++) {
        Kokkos::parallel_for("diffusion", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_-1, yn_-1}), KOKKOS_LAMBDA(int i, int j) {
            (I - dt*nu*Laplacian) u* = u^n + dt*(-Adv_u)
            (I - dt*nu*Laplacian) v* = v^n + dt*(-Adv_v)
            u_star(i,j) - dt * nu * (u_star(i-1,j) + 2 * u_star(i,j) + u_star(i+1))/ (dx_*dx_) + (u_star(i,j-1) + 2 * u_star(i,j) + u_star(i,j+1))/ (dy_*dy_)
             = u_(i,j) + dt * u_adv(i,j)
            rhs = un + dt * Adv_u
            laplace = (u_star(i-1,j) + 2 * u_star(i,j) + u_star(i+1))/ (dx_*dx_) + (u_star(i,j-1) + 2 * u_star(i,j) + u_star(i,j+1))/ (dy_*dy_)


            u_star(i,j) = u_(i,j) + dt * (nu * (1/(dx_ * dx_) * (u_(i+1, j) - 2*u_(i,j) + u_(i-1, j)) 
            + 1/(dy_ * dy_) * (u_(i, j+1) - 2*u_(i,j) + u_(i, j-1))) - 
            (0.5 * u_(i, j) * dt/dx_ * (u_(i+1, j) - u_(i-1, j)) 
            + 0.5 * v_at_p(i,j) * dt/dy_ * (u_(i, j+1) - u_(i, j-1)));
            v_star(i,j) = v_(i,j) + dt * (nu * (1/(dx_ * dx_) * (v_(i+1, j) - 2*v_(i,j) + v_(i-1, j)) 
            + 1/(dy_ * dy_) * (v_(i, j+1) - 2*v_(i,j) + v_(i, j-1))) - 
            (0.5 * u_at_p(i, j) * dt/dx_ * (v_(i+1, j) - v_(i-1, j)) 
            + 0.5 * v_(i,j) * dt/dy_ * (v_(i, j+1) - v_(i, j-1)));
    });}
}



void sim::solve_pressure(double dt) 
{
    int max_iteration = 50; // Reduced iterations
    double rho = 48.0;
    double epsilon = 1e-6;   // Reasonable tolerance
    double omega = 1.0;      // No over-relaxation for stability
    
    for (int iteration = 0; iteration < max_iteration; ++iteration) {
        sim::set_pressure_bc();


        Kokkos::parallel_for("pressure_solve",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_-1, yn_-1}),
            KOKKOS_LAMBDA(int i, int j) {
                // Divergence of velocity
                double div_u = (u_(i, j) - u_(i-1, j))/dx_ + (v_(i, j) - v_(i, j-1))/dy_;
                
                // Laplacian of pressure
                double laplacian_p = (p_(i+1, j) + 2 * p_(i, j) + p_(i-1, j))/(dx_*dx_) + 
                                   (p_(i, j+1) + 2 * p_(i, j) + p_(i, j-1))/(dy_*dy_);
                

                double rhs = rho/dt * div_u;
                

                double p_(i,j) = (laplacian_p - rhs);
            });

        if (iteration % 10 == 0) {
            if (sim::residual() < epsilon) {
                Kokkos::deep_copy(p_, p_star); 
                break;
            }
        }
        Kokkos::deep_copy(p_, p_star);
    }



    Kokkos::parallel_for("velocity_update",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_-1, yn_-1}),
        KOKKOS_LAMBDA(int i, int j) {
            // Use the correct physical scaling
            u_(i,j) = u_(i,j) - dt/rho * ((p_(i+1,j) - p_(i, j))/dx_);
            v_(i,j) = v_(i,j) - dt/rho * ((p_(i,j+1) - p_(i, j))/dy_);
        });
        
    Kokkos::fence(); 
}




void sim::run() 
{    
    
    double cfl = 0.1; // Less conservative CFL number
    double max_vel = 1.5; // Maximum velocity from initial condition
    double dt_cfl = cfl * std::min(dx_, dy_) / max_vel;
    
    // Additional stability check for pressure solver
    double dt_pressure = 0.5 * std::min(dx_*dx_, dy_*dy_) / 10;
    double dt = std::min(dt_cfl, 1.0);
    std::cout << "dt_cfl = " << dt_cfl << std::endl;
    std::cout << "dt_pressure = " << dt_pressure << std::endl;

    int double_steps = T_/dt;
    int steps = round(double_steps);
    
    std::cout << "Final dt = " << dt << std::endl;
    
    for (int step = 0; step < steps; step++) {
        sim::set_boundary_conditions();
        sim::set_pressure_bc();
        sim::advect(dt);
        sim::diffuse();
        sim::solve_pressure(dt);
        if (step % 5 == 0) {
            std::cout << "Step " << step << std::endl;
            save_vtk("output_step_" + std::to_string(step));
        }
    }
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