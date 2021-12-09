import numpy as np
import dolfinx
from dolfinx.generation import  UnitIntervalMesh
from dolfinx.mesh import MeshTags
from mpi4py import MPI
from dolfinx.fem import Constant
import matplotlib.pyplot as plt

from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import eps_solver
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
from petsc4py import PETSc
# 1D example of helmholtz equation using FE in dolfinx

# approximation space polynomial degree
deg = 1

# number of elements in each direction of mesh
n_elem = 100

mesh = UnitIntervalMesh(MPI.COMM_WORLD, n_elem)

# DEFINITION OF BOUNDARIES

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
facet_tag = MeshTags(mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])

# Define the boundary conditions

boundary_conditions = {1: {'Dirichlet'},  # inlet
                       2: {'Dirichlet'}}  # outlet

# Define Speed of sound

c = Constant(mesh, PETSc.ScalarType(1))

matrices = PassiveFlame(mesh, facet_tag, boundary_conditions, c)

matrices.assemble_A()
matrices.assemble_C()

A = matrices.A
C = matrices.C

print(A.getSizes())


eigensolver = eps_solver(A,C,np.pi,2,print_results=True)
vr, vi = A.createVecs()
omega = eigensolver.getEigenvalue(2)
eigensolver.getEigenvector(2,vr,None)
print(omega)
# omega, uh = normalize_eigenvector(mesh, eigensolver, 2)
# print(omega)
# print(uh.x.array)

plt.plot(vr[:].real)
plt.plot(vr[:].imag)
plt.savefig("1Dpassive.png")

# import dolfinx.plot
# topology, cell_types = dolfinx.plot.create_vtk_topology(mesh, mesh.topology.dim)

# import pyvista
# grid = pyvista.UnstructuredGrid(topology, cell_types, mesh.geometry.x)
# grid.point_data["u"] = uh.compute_point_values().real
# grid.set_active_scalars("u")

# plotter = pyvista.Plotter()
# plotter.add_mesh(grid, show_edges=True, show_scalar_bar=True)
# plotter.view_xy()
# if not pyvista.OFF_SCREEN:
#     plotter.show()
# else:
#     figure = plotter.screenshot("fundamentals.png")









