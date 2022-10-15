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
from helmholtz_x.dolfinx_utils import xdmf_writer
from petsc4py import PETSc
import params_dim


# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 40
mesh = create_unit_interval(MPI.COMM_WORLD, n_elem)
V = FunctionSpace(mesh, ("Lagrange", degree))

def fl_subdomain_func(x, eps=1e-16):
    x = x[0]
    x_f = 0.25
    a_f = 0.025
    return np.logical_and(x_f - a_f - eps <= x, x <= x_f + a_f + eps)
tdim = mesh.topology.dim
marked_cells = locate_entities(mesh, tdim, fl_subdomain_func)
fl = 0
subdomains = meshtags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))

boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1))]

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker))
facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

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