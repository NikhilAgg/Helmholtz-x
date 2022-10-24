from dolfinx.fem import FunctionSpace,Constant
from dolfinx.mesh import meshtags, locate_entities,create_unit_interval
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import matplotlib.pyplot as plt
from helmholtz_x.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.passive_flame_x import PassiveFlame
from helmholtz_x.active_flame_x import ActiveFlameNT
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer, normalize, OneDimensionalSetup
from petsc4py import PETSc
import params_dim


# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 30

mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)

# Define the boundary conditions

boundary_conditions = {1: {'Dirichlet'},  # inlet
                       2: {'Dirichlet'}}  # outlet

# Define Speed of sound

# c = params_dim.c(mesh)

c = Constant(mesh, PETSc.ScalarType(400))
# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tag, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C

eta = params_dim.eta
tau = params_dim.tau_linear(mesh,params_dim.x_f, params_dim.a_f, params_dim.tau_u, params_dim.tau_d, degree=degree)

rho = params_dim.rho(mesh, params_dim.x_f, params_dim.a_f, params_dim.rho_d, params_dim.rho_u)
w = params_dim.gaussianFunction(mesh, params_dim.x_r, params_dim.a_r)
h = params_dim.h(mesh, params_dim.x_f, params_dim.a_f)

target = 200 * 2 * np.pi # 150 * 2 * np.pi

D = ActiveFlameNT(mesh, subdomains, w, h, rho, params_dim.Q_tot, params_dim.U_bulk, eta, tau,  
                    degree=degree)
D.assemble_submatrices()
# E = fixed_point_iteration_pep(matrices, D, target, nev=2, i=0, print_results= False)

# omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

# xdmf_writer("Results/p", mesh, uh)

# if MPI.COMM_WORLD.rank == 0:
#     print(f"Eigenfrequency ->  {omega/(2*np.pi):.3f}")

# fig, ax = plt.subplots(2, figsize=(12, 6))
# ax[0].plot(uh.x.array.real)
# ax[1].plot(uh.x.array.imag)
# plt.savefig("Results/"+"1DActive"+".png")

# # uh_normalized = normalize(uh)
# # xdmf_writer("Results/p_normalized", mesh, uh_normalized)
