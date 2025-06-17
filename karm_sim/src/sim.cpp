#include "sim.hpp"
#include <fstream>

sim::sim(int xn, int yn, double xL, double yL, int T) 
  : xn_(xn), yn_(yn), dx_(xL/xn), dy_(yL/yn), T_(T),
    u_("u", xn, yn), v_("v", xn, yn), p_("p", xn, yn) 
{
    sim::init_grids();
    sim::set_initial_conditions();
    sim::set_boundary_conditions();
    sim::set_pressure_bc();
}


void sim::init_grids() 
{
    Kokkos::parallel_for("init_p", xn_, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j<yn_; j++) {
            p_(i,j) = 0.0;
        };
    });
    Kokkos::parallel_for("init_u", xn_, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j<yn_; j++) {
            u_(i,j) = 0.0;
        };
    });
    Kokkos::parallel_for("init_v", xn_, KOKKOS_LAMBDA(const int i) {
        for (int j = 0; j<yn_; j++) {
            v_(i,j) = 0.0;
        };
    });
}

void sim::set_initial_conditions() 
{
    Kokkos::parallel_for("set_ic", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0}, {xn_, yn_}), 
    KOKKOS_LAMBDA(int i, int j) {
        u_(i,j) = 2.0;
    });
}





void sim::set_boundary_conditions() 
{
    Kokkos::parallel_for("set_ic", yn_, KOKKOS_LAMBDA(int j) {
        u_(0, j) = 2.0;
        u_(xn_, j) = u_(xn_-1, j);
        v_(0, j) = 0.0;
        v_(xn_, j) = v_(xn_-1, j);
    });
    
    Kokkos::parallel_for("set_ic", xn_, KOKKOS_LAMBDA(int i) {
        u_(i, 0) = 0.0;
        u_(i, yn_) = 0.0;
        v_(i, 0) = 0.0;
        v_(i, yn_) = 0.0;
    });
}

void sim::set_pressure_bc() 
{
    Kokkos::parallel_for("pressure_bc", xn_, KOKKOS_LAMBDA(int i) {
        p_(i, 0) = p_(i, 1);
        p_(i, yn_) = p_(i, yn_-1);
    });

    Kokkos::parallel_for("pressure_bc", yn_, KOKKOS_LAMBDA(int j) {
        p_(0, j) = p_(1, j);
        p_(xn_, j) = 0.0;
    });
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
    double dx = 2.0 / xn_;
    double dy = 2.0 / yn_;
    Kokkos::View<double**> u_star("u_star", xn_, yn_);
    Kokkos::View<double**> v_star("v_star", xn_, yn_);

    Kokkos::parallel_for("advection_diffusion", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>( {1,1}, {xn_-1, yn_-1}),
        KOKKOS_LAMBDA(int i, int j) {
        for (int j=0; j<yn_; j++) {
            u_star(i, j) = u_(i,j) 
                - u_(i, j) * dt/dx * (u_(i, j) - u_(i-1, j)) 
                - v_(i,j) * dt/dy * (u_(i, j) - u_(i, j-1)) 
                + nu * (dt/(dx * dx) * (u_(i+1, j) - 2*u_(i,j) + u_(i-1, j)) 
            + dt/(dy * dy) * (u_(i, j+1) - 2*u_(i,j) + u_(i, j-1)));

            v_star(i, j) = v_(i,j) 
                - u_(i, j) * dt/dx * (v_(i, j) - v_(i-1, j)) 
                - v_(i,j) * dt/dy * (v_(i, j) - v_(i, j-1)) 
                + nu * (dt/(dx * dx) * (v_(i+1, j) - 2*v_(i,j) + v_(i-1, j)) 
            + dt/(dy * dy) * (v_(i, j+1) - 2*v_(i,j) + v_(i, j-1)));

        }
    });
    sim::set_boundary_conditions();

    Kokkos::deep_copy(u_, u_star);
    Kokkos::deep_copy(v_, v_star);
}



void sim::solve_pressure(double dt) 
{
    int max_iteration = 20;
    double rho = 1.0;
    double dx = 2.0 / xn_;
    double dy = 2.0 / yn_;
    double epsilon = 1e-3;
    Kokkos::View<double**> p_star("p_star", xn_, yn_);

    for (int iteration = 0; iteration < max_iteration; ++iteration) {
        sim::set_pressure_bc();


        Kokkos::parallel_for("pressure_solve",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_-1, yn_-1}),
            KOKKOS_LAMBDA(int i, int j) {
                p_star(i,j) = 
                (((p_(i+1, j) + p_(i-1, j))/(dx*dx)) +
                ((p_(i, j+1) + p_(i, j-1))/(dy*dy)) - 
                (rho/dt * (((u_(i, j) - u_(i-1, j))/dx) 
                + ((v_(i, j) - v_(i, j-1))/dy)))) 
                / (2/(dx*dx) + 2/(dy*dy));
            });


        if (iteration % 5 == 0) {
            if (sim::residual() < epsilon) {Kokkos::deep_copy(p_, p_star); break;}
        }
        Kokkos::deep_copy(p_, p_star);
    }



    Kokkos::parallel_for("velocity_update",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1}, {xn_-1, yn_-1}),
        KOKKOS_LAMBDA(int i, int j) {
            u_(i,j) = u_(i,j) - dt/rho * ((p_(i+1,j) - p_(i, j))/dx);
            v_(i,j) = v_(i,j) - dt/rho * ((p_(i,j+1) - p_(i, j))/dy);
        });
}




void sim::run(int steps) 
{
    double dt = T_/steps;
    Kokkos::parallel_for("run", steps, KOKKOS_LAMBDA(const int i) {
        sim::advect(dt);
        sim::solve_pressure(dt);
    });
}
