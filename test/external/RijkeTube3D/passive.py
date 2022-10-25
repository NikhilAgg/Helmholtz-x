from dolfinx.fem import FunctionSpace,Constant
from dolfinx.mesh import meshtags, locate_entities,create_unit_interval
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import matplotlib.pyplot as plt
from helmholtz_x.eigensolvers_x import fixed_point_iteration_pep, pep_solver
from helmholtz_x.passive_flame_x import PassiveFlame
from helmholtz_x.active_flame_x import ActiveFlameNT
from helmholtz_x.eigenvectors_x import normalize_eigenvector
from helmholtz_x.dolfinx_utils import xdmf_writer, normalize, XDMFReader
from petsc4py import PETSc
import params


# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
rijke3d = XDMFReader("MeshDir/rijke")
mesh, subdomains, facet_tags = rijke3d.getAll()
rijke3d.getInfo()
# Define the boundary conditions

boundary_conditions = {4: {'Neumann'},
                       3: {'Neumann'},
                       2: {'Neumann'},
                       1: {'Neumann'}}

boundary_conditions = {}

# Define Speed of sound

c = params.c_DG(mesh, params.x_f, params.c_u, params.c_u)

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tags, boundary_conditions, c, degree=degree)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C


target = 200 * 2 * np.pi # 150 * 2 * np.pi


E = pep_solver(matrices.A, matrices.B, matrices.C, target, nev=2, print_results= True)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=degree, which='right')

xdmf_writer("Results/p", mesh, uh)

if MPI.COMM_WORLD.rank == 0:
    print(f"Eigenfrequency ->  {omega/(2*np.pi):.3f}")

fig, ax = plt.subplots(2, figsize=(12, 6))
ax[0].plot(uh.x.array.real)
ax[1].plot(uh.x.array.imag)
plt.savefig("Results/"+"1DPassive"+".png")

# uh_normalized = normalize(uh)
# xdmf_writer("Results/p_normalized", mesh, uh_normalized)
