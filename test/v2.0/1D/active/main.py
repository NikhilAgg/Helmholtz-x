from dolfinx.fem import FunctionSpace
from dolfinx.mesh import meshtags, locate_entities,create_unit_interval
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from helmholtz_x.active_flame_x import ActiveFlame
from helmholtz_x.flame_transfer_function_x import n_tau
from helmholtz_x.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.passive_flame_x import PassiveFlame
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.dolfinx_utils import OneDimensionalSetup

import params

# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 300
mesh = create_unit_interval(MPI.COMM_WORLD, n_elem)
V = FunctionSpace(mesh, ("Lagrange", degree))

mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)
# Define the boundary conditions

boundary_conditions = {1: {'Robin': params.R_in},  # inlet
                       2: {'Robin': params.R_out}}  # outlet

# Define Speed of sound

c = params.c(mesh)

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C

ftf = n_tau(params.n, params.tau)


D = ActiveFlame(mesh, subdomains,
                    params.x_r, params.rho_in, 1., 1., ftf,
                    degree=degree)

D.assemble_submatrices()

E = fixed_point_iteration_pep(matrices, D, np.pi, nev=2, i=0, print_results= False)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=1, which='right')

plt.plot(uh.x.array.real)
plt.savefig("Results/1Dactive_real.png")
plt.clf()

plt.plot(uh.x.array.imag)
plt.savefig("Results/1Dactive_imag.png")
