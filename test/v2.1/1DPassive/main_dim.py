from dolfinx.fem import FunctionSpace,Constant
from dolfinx.mesh import meshtags, locate_entities,create_unit_interval
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import matplotlib.pyplot as plt
from helmholtz_x.eigensolvers_x import eps_solver
from helmholtz_x.passive_flame_x import PassiveFlame
from helmholtz_x.active_flame_x import ActiveFlameNT
from helmholtz_x.eigenvectors_x import normalize_eigenvector, normalize_unit
from helmholtz_x.dolfinx_utils import xdmf_writer, OneDimensionalSetup
from petsc4py import PETSc
import params_dim


# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 40
mesh, subdomains, facet_tags = OneDimensionalSetup(n_elem)

# Define the boundary conditions

boundary_conditions = {1: {'Neumann'},  # inlet
                       2: {'Neumann'}}  # outlet

# Define Speed of sound

# c = params_dim.c(mesh)

c = Constant(mesh, PETSc.ScalarType(340))
# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tag, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_C()

A = matrices.A
C = matrices.C


target = 170 * 2 * np.pi # 150 * 2 * np.pi

E = eps_solver(A,C, target**2, nev=2, print_results= False)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

xdmf_writer("Results/p", mesh, uh)

uh_normalized = normalize_unit(uh)
xdmf_writer("Results/p_normalized", mesh, uh_normalized)

if MPI.COMM_WORLD.rank == 0:
    print(f"Eigenfrequency ->  {omega/(2*np.pi):.3f}")

fig, ax = plt.subplots(2, figsize=(12, 6))
ax[0].plot(uh.x.array.real)
ax[1].plot(uh.x.array.imag)
plt.savefig("Results/"+"1DPassive"+".png")