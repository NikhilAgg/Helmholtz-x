import dolfinx
from dolfinx.fem import Function, FunctionSpace
from dolfinx.generation import UnitSquareMesh, RectangleMesh
from dolfinx.fem.assemble import assemble_matrix, assemble_scalar
from mpi4py import MPI
from ufl import Measure, FacetNormal, TestFunction, TrialFunction, dx, grad, inner
from petsc4py import PETSc
import numpy as np
from slepc4py import SLEPc


# mesh = UnitSquareMesh(MPI.COMM_WORLD, 8, 8, dolfinx.cpp.mesh.CellType.quadrilateral)

p0 = [0,0,0]
p1 = [1,0.047,0]
mesh = RectangleMesh(MPI.COMM_WORLD,[p0,p1],[200,50])

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
solver.setTarget(np.pi)
st = solver.getST()
st.setType('sinvert')
solver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  # TARGET_REAL or TARGET_IMAGINARY
solver.setDimensions(5, SLEPc.DECIDE)
solver.setFromOptions()
solver.solve()

A = solver.getOperators()[1]
vr, vi = A.createVecs()

eig = solver.getEigenvalue(1)
omega = np.sqrt(eig)
solver.getEigenvector(0, vr, vi)

print(omega)

p = Function(V)
p.vector.setArray(vr.array)

meas = assemble_scalar(p*p*dx)
meas = np.sqrt(meas)

temp = vr.array
temp= temp/meas

p_normalized = Function(V)
p_normalized.vector.setArray(temp)


import dolfinx.io
p_normalized.name = "Acoustic_Wave"
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "p.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(p_normalized)
