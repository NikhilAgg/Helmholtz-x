import dolfinx
import dolfinx.geometry
from dolfinx import FunctionSpace, UnitIntervalMesh
from dolfinx.fem.assemble import assemble_matrix
import basix
import numpy as np
from mpi4py import MPI
from ufl import TestFunction, TrialFunction, dx, inner, Measure
from petsc4py import PETSc

from helmholtz_x.helmholtz_pkgx.active_flame_x import ActiveFlame
from helmholtz_x.helmholtz_pkgx.flame_transfer_function_x import n_tau
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import fixed_point_iteration_pep
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame

import params

# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 30
mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, n_elem, n_elem)
V = FunctionSpace(mesh, ("Lagrange", degree))

def fl_subdomain_func(x, eps=1e-16):
    x = x[0]
    x_f = 0.25
    a_f = 0.025
    return np.logical_and(x_f - a_f - eps <= x, x <= x_f + a_f + eps)
tdim = mesh.topology.dim
marked_cells = dolfinx.mesh.locate_entities(mesh, tdim, fl_subdomain_func)
fl = 0
subdomains = dolfinx.MeshTags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))

boundaries = [(1, lambda x: np.isclose(x[0], 0)),
              (2, lambda x: np.isclose(x[0], 1))]

facet_indices, facet_markers = [], []
fdim = mesh.topology.dim - 1
for (marker, locator) in boundaries:
    facets = dolfinx.mesh.locate_entities(mesh, fdim, locator)
    facet_indices.append(facets)
    facet_markers.append(np.full(len(facets), marker))
facet_indices = np.array(np.hstack(facet_indices), dtype=np.int32)
facet_markers = np.array(np.hstack(facet_markers), dtype=np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = dolfinx.MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# Define the boundary conditions

boundary_conditions = {1: {'Robin': params.Y_in},  # inlet
                       2: {'Robin': params.Y_out}}  # outlet

# Define Speed of sound

c = dolfinx.Constant(mesh, PETSc.ScalarType(1))

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
                    degree=degree)

D.assemble_submatrices()

E = fixed_point_iteration_pep(matrices, D, np.pi, nev=2, i=0, print_results= False)
