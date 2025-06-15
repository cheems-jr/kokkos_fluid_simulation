#include "sim.hpp"
#include <fstream>

sim::Simulator(int NX, int NY, double Lx, double Ly) 
  : NX_(NX), NY_(NY), dx_(Lx/NX), dy_(Ly/NY),
    u_("u", NX, NY), v_("v", NX, NY), p_("p", NX, NY) 
{
  // Initialize flow fields
  Kokkos::parallel_for("init", NX_, KOKKOS_LAMBDA(const int i) {
    for (int j = 0; j < NY_; j++) {
      u_(i,j) = 1.0; // Uniform flow in x-direction
    }
  });
}

void Simulator::advect(double dt) {
  // Implement advection here using Kokkos::parallel_for
}

void Simulator::save_vtk(const std::string& prefix) {
  std::ofstream file("output/" + prefix + ".vtk");
  file << "DATASET STRUCTURED_GRID\n";
  file << "DIMENSIONS " << NX_ << " " << NY_ << " 1\n";
  // Write grid points and field data...
}