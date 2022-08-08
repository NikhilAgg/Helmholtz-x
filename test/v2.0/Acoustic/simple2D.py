import datetime
start_time = datetime.datetime.now()

import dolfinx
from dolfinx.fem import Function, FunctionSpace, Constant, form, assemble_scalar
from dolfinx.mesh import create_unit_square
from dolfinx.fem.petsc import assemble_matrix
from mpi4py import MPI
from ufl import Measure,  TestFunction, TrialFunction, dx, grad, inner,ds
from petsc4py import PETSc
import numpy as np
from slepc4py import SLEPc


mesh = create_unit_square(MPI.COMM_WORLD, 750, 750)
dx = Measure("dx",domain=mesh)

V = FunctionSpace(mesh, ("Lagrange", 1))

u = TrialFunction(V)
v = TestFunction(V)

a = form(inner(grad(u), grad(v))*dx)
A = assemble_matrix(a)
A.assemble()

c = form(-inner(u , v) * dx)
C = assemble_matrix(c)
C.assemble()

solver = SLEPc.EPS().create(MPI.COMM_WORLD)
C = - C

solver.setOperators(A, C)
st = solver.getST()
st.setType('sinvert')
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

if MPI.COMM_WORLD.rank == 0:
    print("Total Execution Time: ", datetime.datetime.now()-start_time)
