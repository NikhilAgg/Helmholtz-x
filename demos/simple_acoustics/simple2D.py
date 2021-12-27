import dolfinx
from dolfinx.fem import Function, FunctionSpace, Constant
from dolfinx.mesh import create_unit_square
from dolfinx.fem.assemble import assemble_matrix,assemble_scalar
from mpi4py import MPI
from ufl import Measure,  TestFunction, TrialFunction, dx, grad, inner,ds
from petsc4py import PETSc
import numpy as np
from slepc4py import SLEPc


mesh = create_unit_square(MPI.COMM_WORLD, 8, 8)
dx = Measure("dx",domain=mesh)

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
p.x.scatter_forward()
import dolfinx.io
p.name = "Acoustic_Wave"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "p.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p)
