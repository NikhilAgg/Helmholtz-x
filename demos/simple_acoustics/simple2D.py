import dolfinx
from dolfinx import Function, FunctionSpace
from dolfinx.fem.assemble import assemble_matrix
from mpi4py import MPI
from ufl import Measure, FacetNormal, TestFunction, TrialFunction, dx, grad, inner
from petsc4py import PETSc
import numpy as np
from slepc4py import SLEPc


mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 8, 8, dolfinx.cpp.mesh.CellType.quadrilateral)


V = FunctionSpace(mesh, ("Lagrange", 1))

u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v))*dx
A = assemble_matrix(a)
A.assemble()

c = -inner(u , v) * dx
C = assemble_matrix(c)
C.assemble()

solver = SLEPc.EPS().create(MPI.COMM_WORLD)
C = - C

solver.setOperators(A, C)
solver.setFromOptions()
solver.solve()

A = solver.getOperators()[0]
vr, vi = A.createVecs()

eig = solver.getEigenvalue(0)
omega = np.sqrt(eig)
solver.getEigenvector(0, vr, vi)
print(omega)

p = Function(V)
p.vector.setArray(vr.array)
import dolfinx.io
p.name = "Acoustic_Wave"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "p.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p)
