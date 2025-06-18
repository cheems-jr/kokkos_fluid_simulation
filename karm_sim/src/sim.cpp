#include "sim.hpp"
#include <string>
#include <vector>
#include <fstream>

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
    Kokkos::parallel_for("init_p",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {xn_+1, yn_}),
        KOKKOS_LAMBDA(int i, int j) {
            u_(i,j) = 0.0;
            u_star(i,j) = 0.0;
    });
    Kokkos::parallel_for("init_p",
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
        u_(i,j) = 6.0 * y*(yL-y)/(yL*yL);
    });
    Kokkos::fence(); 
}





void sim::set_boundary_conditions() 
{
    Kokkos::parallel_for("set_bc_y", yn_, KOKKOS_LAMBDA(int j) {
        double yL = yn_ * dy_;
        double y = (j+0.5)*dy_;
        u_(0,j) = 6.0 * y*(yL-y)/(yL*yL);
        u_(xn_, j) = u_(xn_-1, j);
    });

    Kokkos::parallel_for("set_bc_y", yn_+1, KOKKOS_LAMBDA(int j) {
        v_(0, j) = 0.0;
        v_(xn_-1, j) = v_(xn_-2, j);
    });
    
    Kokkos::parallel_for("set_bc_x", xn_, KOKKOS_LAMBDA(int i) {
        v_(i, 0) = 0.0;
        v_(i, yn_) = 0.0;
    });
    Kokkos::parallel_for("set_bc_x", xn_+1, KOKKOS_LAMBDA(int i) {
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
    double nu = 1.0;

    Kokkos::parallel_for("advection_diffusion", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>( {1,1}, {xn_-1, yn_-1}),
        KOKKOS_LAMBDA(int i, int j) {
            double u_adv = u_(i, j) * dt/dx_ * (u_(i, j) - u_(i-1, j)) 
                    + v_at_p(i,j) * dt/dy_ * (u_(i, j) - u_(i, j-1));
            double v_adv = u_at_p(i, j) * 1/dx_ * (v_(i, j) - v_(i-1, j)) 
                    + v_(i,j) * 1/dy_ * (v_(i, j) - v_(i, j-1));

            double u_diff = nu * (1/(dx_ * dx_) * (u_(i+1, j) - 2*u_(i,j) + u_(i-1, j)) 
            + 1/(dy_ * dy_) * (u_(i, j+1) - 2*u_(i,j) + u_(i, j-1)));
            double v_diff = nu * (1/(dx_ * dx_) * (v_(i+1, j) - 2*v_(i,j) + v_(i-1, j)) 
            + 1/(dy_ * dy_) * (v_(i, j+1) - 2*v_(i,j) + v_(i, j-1)));

            u_star(i,j) = u_(i,j) + dt * (u_diff - u_adv);
            v_star(i,j) = v_(i,j) + dt * (v_diff - v_adv);
    });

    Kokkos::deep_copy(u_, u_star);
    Kokkos::deep_copy(v_, v_star);
    sim::set_boundary_conditions();
    Kokkos::fence(); 
}



void sim::solve_pressure(double dt) 
{
    int max_iteration = 10;
    double rho = 1.0;
    double epsilon = 1e-3;
    for (int iteration = 0; iteration < max_iteration; ++iteration) {
        sim::set_pressure_bc();


        Kokkos::parallel_for("pressure_solve",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_-1, yn_-1}),
            KOKKOS_LAMBDA(int i, int j) {
                p_star(i,j) = 
                (((p_(i+1, j) + p_(i-1, j))/(dx_*dx_)) +
                ((p_(i, j+1) + p_(i, j-1))/(dy_*dy_)) - 
                (rho/dt * (((u_(i, j) - u_(i-1, j))/dx_) 
                + ((v_(i, j) - v_(i, j-1))/dy_)))) 
                / (2/(dx_*dx_) + 2/(dy_*dy_));
            });


        if (iteration % 5 == 0) {
            if (sim::residual() < epsilon) {Kokkos::deep_copy(p_, p_star); break;}
        }
        Kokkos::deep_copy(p_, p_star);
    }



    Kokkos::parallel_for("velocity_update",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_-1, yn_-1}),
        KOKKOS_LAMBDA(int i, int j) {
            u_(i,j) = u_(i,j) - dt/rho * ((p_(i+1,j) - p_(i, j))/dx_);
            v_(i,j) = v_(i,j) - dt/rho * ((p_(i,j+1) - p_(i, j))/dy_);
        });
        
    Kokkos::fence(); 
}




void sim::run(int steps) 
{
    double dt = T_/steps;
    for (int step = 0; step < steps; step++) {
        sim::set_boundary_conditions();
        sim::set_pressure_bc();
        sim::advect(dt);
        sim::solve_pressure(dt);
        if (step % 100 == 0) {
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