import numpy as np
import dolfinx
from dolfinx import  UnitIntervalMesh
from dolfinx.io import XDMFFile, VTKFile
from mpi4py import MPI
import matplotlib.pyplot as plt


from helmholtz_x.helmholtz_pkgx.eigenvectors_x import normalize_eigenvector
from helmholtz_x.helmholtz_pkgx.eigensolvers_x import pep_solver
from helmholtz_x.helmholtz_pkgx.passive_flame_x import PassiveFlame
import params
from petsc4py import PETSc

# 1D example of helmholtz equation using FE in dolfinx

# approximation space polynomial degree
deg = 1

# number of elements in each direction of mesh
n_elem = 100

mesh = UnitIntervalMesh(MPI.COMM_WORLD, n_elem)

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

c = params.c(mesh)

# Introduce Passive Flame Matrices

matrices = PassiveFlame(mesh, facet_tag, boundary_conditions, c)

matrices.assemble_A()
matrices.assemble_B()
matrices.assemble_C()

A = matrices.A
B = matrices.B
C = matrices.C


print(A.getSizes())

eigensolver = pep_solver(A,B,C,np.pi, 5,print_results=True)

omega, uh = normalize_eigenvector(mesh, eigensolver, 0)
print(omega)

# plt.plot(uh.vector[:].real)
# plt.plot(uh.vector[:].imag)
# plt.savefig("1Dpassive_R.png")

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









