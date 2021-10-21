from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np
def results(E):

    print()
    print("******************************")
    print("*** SLEPc Solution Results ***")
    print("******************************")
    print()

    its = E.getIterationNumber()
    print("Number of iterations of the method: %d" % its)

    eps_type = E.getType()
    print("Solution method: %s" % eps_type)

    nev, ncv, mpd = E.getDimensions()
    print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = E.getTolerances()
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    nconv = E.getConverged()
    print("Number of converged eigenpairs %d" % nconv)

    A = E.getOperators()[0]
    vr, vi = A.createVecs()

    if nconv > 0:
        print()
    for i in range(nconv):
        k = E.getEigenpair(i, vr, vi)
        print("%15f, %15f" % (k.real, k.imag))
    print()

def eps_solver(A, C, target, nev, two_sided=False, print_results=False):

    E = SLEPc.EPS().create(MPI.COMM_WORLD)

    C = - C
    E.setOperators(A, C)

    # E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

    # spectral transformation
    st = E.getST()
    st.setType('sinvert')

    E.setTarget(target)
    E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)  # TARGET_REAL or TARGET_IMAGINARY
    #print("target is : ", E.getTarget())
    E.setTwoSided(two_sided)

    E.setDimensions(nev, SLEPc.DECIDE)
    E.setTolerances(1e-15)
    E.setFromOptions()

    E.solve()

    if print_results:
        results(E)

    return E

def pep_solver(A, B, C, target, nev, print_results=False):
    """
    This function defines solved instance for
    A + wB + w^2 C = 0

    Parameters
    ----------
    A : petsc4py.PETSc.Mat
        Matrix of Grad term
    B : petsc4py.PETSc.Mat
        Empty Matrix
    C : petsc4py.PETSc.Mat
        Matrix of w^2 term.
    target : float
        targeted nondimensional eigenvalue
    nev : int
        Requested number of eigenvalue
    print_results : boolean, optional
        Prints the results. The default is False.

    Returns
    -------
    Q : slepc4py.SLEPc.PEP
        Solution instance of eigenvalue problem.

    """

    Q = SLEPc.PEP().create()

    operators = [A, B, C]
    Q.setOperators(operators)

    # Q.setProblemType(SLEPc.PEP.ProblemType.GENERAL)

    # spectral transformation
    st = Q.getST()
    st.setType('sinvert')

    Q.setTarget(target)
    Q.setWhichEigenpairs(SLEPc.PEP.Which.TARGET_MAGNITUDE)  # TARGET_REAL or TARGET_IMAGINARY
    #print("target is : ", Q.getTarget())
    Q.setDimensions(nev, SLEPc.DECIDE)
    Q.setTolerances(1e-15)
    Q.setFromOptions()

    Q.solve()

    if print_results:
        results(Q)
    return Q
    

def fixed_point_iteration_pep(operators, D, target, nev=2, i=0,
                              tol=1e-8, maxiter=50,
                              print_results=False,
                              problem_type='direct'):

    A = operators.A
    C = operators.C
    B = operators.B
    if problem_type == 'adjoint':
        B = operators.B_adj

    omega = np.zeros(maxiter, dtype=complex)
    f = np.zeros(maxiter, dtype=complex)
    alpha = np.zeros(maxiter, dtype=complex)

    E = pep_solver(A, B, C, target, nev, print_results=print_results)
    vr, vi = A.getVecs()
    eig = E.getEigenpair(i, vr, vi)

    omega[0] = eig
    alpha[0] = .5

    domega = 2 * tol
    k = - 1

    # formatting
    s = "{:.0e}".format(tol)
    s = int(s[-2:])
    s = "{{:+.{}f}}".format(s)

    while abs(domega) > tol:

        k += 1

        D.assemble_matrix(omega[k], problem_type)
        D_Mat = D.matrix
        if problem_type == 'adjoint':
            D_Mat = D.adjoint_matrix

        nlinA = A - D_Mat

        E = pep_solver(nlinA, B, C, target, nev, print_results=print_results)
        eig = E.getEigenpair(i, vr, vi)

        f[k] = eig

        if k != 0:
            alpha[k] = 1 / (1 - ((f[k] - f[k-1]) / (omega[k] - omega[k-1])))

        omega[k+1] = alpha[k] * f[k] + (1 - alpha[k]) * omega[k]

        domega = omega[k+1] - omega[k]

        print('iter = {:2d},  omega = {}  {}j,  |domega| = {:.2e}'.format(
            k + 1, s.format(omega[k + 1].real), s.format(omega[k + 1].imag), abs(domega)
        ))

    return E

