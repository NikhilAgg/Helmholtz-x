from helmholtz_x.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.passive_flame_x import PassiveFlame
from helmholtz_x.active_flame_x import ActiveFlameNT
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer, OneDimensionalSetup
from helmholtz_x.parameters_utils import temperature_step, gaussianFunction, rho
from helmholtz_x.solver_utils import start_time, execution_time
start = start_time()
import numpy as np
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

# Introduce Passive Flame Matrices

T = temperature_step(mesh, params_dim.x_f, params_dim.T_u, params_dim.T_d)

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, T, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

# Introduce Active Flame Matrix parameters

rho = rho(mesh, params_dim.x_f, params_dim.a_f, params_dim.rho_d, params_dim.rho_u)
w = gaussianFunction(mesh, params_dim.x_r, params_dim.a_r)
h = gaussianFunction(mesh, params_dim.x_f, params_dim.a_f)

D = ActiveFlameNT(mesh, subdomains, w, h, rho, T, params_dim.eta, params_dim.tau, degree=degree)
D.assemble_submatrices(problem_type='direct')

# Introduce solver object and start

target = 200 * 2 * np.pi # 150 * 2 * np.pi
E = fixed_point_iteration_pep(matrices, D, target, nev=2, i=0, print_results= False)

# Extract eigenvalue and normalized eigenvector 

omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

# Save Eigenvector

xdmf_writer("Results/p", mesh, uh)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, figsize=(12, 6))
ax[0].plot(uh.x.array.real)
ax[1].plot(uh.x.array.imag)
plt.savefig("Results/"+"1DActive"+".png")

execution_time(start)