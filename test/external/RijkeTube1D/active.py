import datetime
start_time = datetime.datetime.now()
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
import params_dim

# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 3000

mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)

# Define the boundary conditions

# boundary_conditions = {1: {'Robin': params_dim.R_in},  # inlet
#                        2: {'Robin': params_dim.R_out}}  # outlet
# boundary_conditions = {1: {'Dirichlet'},  # inlet
#                        2: {'Dirichlet'}}  # outlet}

boundary_conditions = {}

# Define Speed of sound

c = params_dim.c_DG(mesh, params_dim.x_f, params_dim.c_u, params_dim.c_d)

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C


rho = params_dim.rho(mesh, params_dim.x_f, params_dim.a_f, params_dim.rho_d, params_dim.rho_u)
w = params_dim.gaussianFunction(mesh, params_dim.x_r, params_dim.a_r)
h = params_dim.gaussianFunction(mesh, params_dim.x_f, params_dim.a_f)

target = 200 * 2 * np.pi # 150 * 2 * np.pi

D = ActiveFlameNT(mesh, subdomains, w, h, rho, 1, 1, params_dim.eta, params_dim.tau, degree=degree)
D.assemble_submatrices()

E = fixed_point_iteration_pep(matrices, D, target, nev=2, i=0, print_results= False)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

xdmf_writer("Results/p", mesh, uh)

fig, ax = plt.subplots(2, figsize=(12, 6))
ax[0].plot(uh.x.array.real)
ax[1].plot(uh.x.array.imag)
plt.savefig("Results/"+"1DActive"+".png")

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)
