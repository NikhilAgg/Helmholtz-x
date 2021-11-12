from dolfinx.fem.assemble import assemble_scalar
import numpy as np
from slepc4py import SLEPc
from petsc4py import PETSc
from dolfinx import ( Function, FunctionSpace)
from ufl import dx
from .petsc4py_utils import multiply, vector_matrix_vector

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
    p = Function(V)
    
    # index = int(len(vr.array)/2)
    # angle_0 = np.arctan2(vr.array[0].imag,vr.array[0].real)
    # p.vector.setArray(vr.array*np.exp(-angle_0*1j))

    p.vector.setArray(vr.array)
    

    meas = assemble_scalar(p*p*dx)
    meas = np.sqrt(meas)
    
    temp = vr.array
    temp= temp/meas

    u_new = Function(V) # Required for Parallel runs
    u_new.vector.setArray(temp)

    return u_new

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

def normalize_adjoint(omega, p_dir, p_adj, matrices, D=None):
    """
    p_dir and p_adj  are both: <class 'dolfinx.function.function.Function'>
    p_dir_vec and p_adj_vec are both: <class 'petsc4py.PETSc.Vec'>

    """

    B = matrices.B

    p_dir_vec = p_dir.vector
    p_adj_vec = p_adj.vector

    if not B and not D:
        print('not B and not D: return None')
        return None
    elif B and not D:
        # B + 2 \omega C
        dL_domega = (B +
                     matrices.C * (2 * omega))
    elif D and not B:
        # 2 \omega C - D'(\omega)
        dL_domega = (matrices.C * (2 * omega) -
                     D.get_derivative(omega))
    else:
        # B + 2 \omega C - D'(\omega)
        dL_domega = (B +
                     matrices.C * (2 * omega) -
                     D.get_derivative(omega))

    meas = vector_matrix_vector(p_adj_vec, dL_domega, p_dir_vec)
    # print("Normalization: ", meas)

    p_adj_vec = multiply(p_adj_vec, 1 / meas)

    # meas = vector_matrix_vector(p_adj_vec, dL_domega, p_dir_vec)
    # print(meas)

    p_adj1 = p_adj
    p_adj1.vector.setArray(p_adj_vec.getArray())


    return p_adj1