from dolfinx.fem import FunctionSpace,Constant
from dolfinx.mesh import MeshTags, locate_entities,create_unit_interval
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from helmholtz_x.helmholtz_pkgx.active_flame_x import ActiveFlame
from helmholtz_x.helmholtz_pkgx.flame_transfer_function_x import n_tau
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
import params


# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 30
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
subdomains = MeshTags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))

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
facet_tag = MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# Define the boundary conditions

boundary_conditions = {1: {'Robin': params.Y_in},  # inlet
                       2: {'Robin': params.Y_out}}  # outlet

# Define Speed of sound

c = params.c(mesh)

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tag, boundary_conditions, c)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C

ftf = n_tau(params.n, params.tau)


D = ActiveFlame(mesh, subdomains,
                    params.x_r, params.rho_in, 1., 1., ftf,
                    degree=degree, comm = MPI.COMM_WORLD)

D.assemble_submatrices()

E = fixed_point_iteration_pep(matrices, D, np.pi, nev=2, i=0, print_results= False)

omega, uh = normalize_eigenvector(mesh, E, 0, degree=1, which='right')

plt.plot(uh.x.array.real)
plt.savefig("1Dactive_real.png")
plt.clf()

plt.plot(uh.x.array.imag)
plt.savefig("1Dactive_imag.png")
