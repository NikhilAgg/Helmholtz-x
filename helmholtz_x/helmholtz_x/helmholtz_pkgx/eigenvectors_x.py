from dolfinx.fem.assemble import assemble_scalar
import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc
from dolfinx import ( Function, FunctionSpace)
from ufl import dx, inner

def normalize_1(mesh, vr, degree=1):
    """ This function normalizes the eigensolution vr
     which is obtained from complex slepc build
     (vi is set to zero in complex build) 

    Args:
        mesh ([dolfinx.cpp.mesh.Mesh]): mesh of the domain
        vr ([petsc4py.PETSc.Vec]): eigensolution
        degree (int, optional): degree of finite elements. Defaults to 1.

    Returns:
        [type]: normalized eigensolution with \int (p p dx) = 1
    """
    
    V = FunctionSpace(mesh, ("CG", degree))
    u = Function(V)
    u.vector.setArray(vr.array)

    meas = assemble_scalar(inner(u,u)*dx)
    meas = np.sqrt(meas)
    temp = vr.getArray()
    temp= temp/meas

    u_new = Function(V) # Required for Parallel runs
    u_new.vector.setArray(temp)
    print("New used")

    return u

def normalize_eigenvector(mesh, obj, i, degree=1, which='right'):

    omega = 0.
    A = obj.getOperators()[0]
    vr, vi = A.createVecs()

    if isinstance(obj, SLEPc.EPS):
        eig = obj.getEigenvalue(i)
        omega = np.sqrt(eig)
        if which == 'right':
            obj.getEigenvector(i, vr, vi)
        elif which == 'left':
            obj.getLeftEigenvector(i, vr, vi)

    elif isinstance(obj, SLEPc.PEP):
        eig = obj.getEigenpair(i, vr, vi)
        omega = eig

    
    p = normalize_1(mesh, vr, degree)

    return omega, p