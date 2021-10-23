"""
All the functions in this module work in parallel
"""


def multiply(x0, z):
    """
    multiply a complex vector by a complex scalar
    """

    x1 = x0.copy()

    x1.scale(z)

    return x1

def conjugate(y0):
    """
    Takes complex conjugate of vector y0

    Parameters
    ----------
    y0 : petsc4py.PETSc.Vec
        Complex vector

    Returns
    -------
    y1 : petsc4py.PETSc.Vec
        Complex vector

    """
    y1 = y0.copy()
    y1 = y0.conjugate()

    return y1

def vector_vector(y0, x0):
    """
    it does y0^H * x0 
    y1 = y0^H(conjugate-transpose of y0)

    Parameters
    ----------
    y0 : petsc4py.PETSc.Vec
        Complex vector
    x0 : petsc4py.PETSc.Vec
        Complex vector

    Returns
    -------
    z : Complex scalar product

    """


    y1 = y0.copy()
    y1 = y0.dot(x0)

    return y1


def vector_matrix_vector(y0, A, x0):
    """
    multiplies complex vector, matrix and complex vector 

    Parameters
    ----------
    y0 : petsc4py.PETSc.Vec
        Complex vector
    A : petsc4py.PETSc.Mat
        Matrix.
    x0 : petsc4py.PETSc.Vec
        Complex vector

    Returns
    -------
    z : complex scalar product

    """

    x1 = x0.copy()
    A.mult(x0, x1) # x1 = A'*x0
    z = vector_vector(y0, x1)

    return z

if __name__ == '__main__':

    from petsc4py import PETSc

    x = PETSc.Vec().createSeq(3) # Faster way to create a sequential vector.
    x.setValues(range(3), range(3)) # x = [0 1 ... 3]

    x2 = multiply(x, 1j)
    print(x2.getArray())

    mat = PETSc.Mat().create(PETSc.COMM_WORLD) # MPI.COMM_SELF
    mat.setSizes([(3, 3), (3, 3)])
    mat.setType('aij') 
    mat.setUp()
    mat.setValues([0, 1], [2, 3], [1, 1, 1, 1]) 
    mat.assemblyBegin() 
    mat.assemblyEnd()
    print(mat.getValues(range(3),range(3)))

    mvm = vector_matrix_vector(x,mat,x2)
    print(mvm)

    x3 = x.copy()
    help(x3)
    x3.setArray(x.getArray())
    print(x3[:])
