import dolfinx
import dolfinx.geometry
from dolfinx import FunctionSpace, UnitIntervalMesh
from dolfinx.fem.assemble import assemble_matrix
import basix
import numpy as np
from mpi4py import MPI
from ufl import TestFunction, TrialFunction, dx, inner, Measure
from petsc4py import PETSc
# approximation space polynomial degree
degree = 1
# number of elements in each direction of mesh
n_elem = 30
mesh = dolfinx.UnitIntervalMesh(MPI.COMM_WORLD, n_elem, dolfinx.cpp.mesh.GhostMode.shared_facet)
# mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, n_elem, n_elem)

V = FunctionSpace(mesh, ("Lagrange", degree))
def fl_subdomain_func(x, eps=1e-16):
    x = x[0]
    x_f = 0.25
    a_f = 0.025
    return np.logical_and(x_f - a_f - eps <= x, x <= x_f + a_f + eps)
tdim = mesh.topology.dim
marked_cells = dolfinx.mesh.locate_entities(mesh, tdim, fl_subdomain_func)
fl = 1
subdomain = dolfinx.MeshTags(mesh, tdim, marked_cells, np.full(len(marked_cells), fl, dtype=np.int32))
dx = Measure("dx", subdomain_data=subdomain)
v = TestFunction(V)
V_fl = MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(dolfinx.Constant(mesh, PETSc.ScalarType(1))*dx(fl)), op=MPI.SUM)
b = dolfinx.Function(V)
b.x.array[:] = 0
a = dolfinx.fem.assemble_vector(b.vector, inner(dolfinx.Constant(mesh, (1/V_fl)), v)*dx(fl))
b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
indices1 = np.flatnonzero(a.getArray())
a = b.x.array
a = list(zip(indices1, a[indices1]))
print(a)