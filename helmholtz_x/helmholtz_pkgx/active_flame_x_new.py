import dolfinx
import basix
from dolfinx.fem  import Function, FunctionSpace, Constant, form
from dolfinx.fem.petsc import assemble_vector
from dolfinx.geometry import BoundingBoxTree,compute_collisions,compute_colliding_cells
from mpi4py import MPI
from ufl import Measure, FacetNormal, TestFunction, TrialFunction, inner, as_vector, grad
import ufl
from petsc4py import PETSc
import numpy as np
from scipy.sparse import csr_matrix
# from numba import njit, jitclass#, jit
# from numba.experimental import jitclass
from numba import jit

# @jitclass
class ActiveFlame:

    gamma = 1.4

    def __init__(self, mesh, subdomains, w, rho, Q, U, FTF,x_r, degree=1, bloch_object=None):

        self.mesh = mesh
        self.subdomains = subdomains
        self.w = w
        self.rho = rho
        self.x_r = x_r
        self.Q = Q
        self.U = U
        self.FTF = FTF
        self.degree = degree
        self.bloch_object = bloch_object
        

        self.coeff = (self.gamma - 1) * Q / U

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        # __________________________________________________

        self._a = {}
        self._b = {}
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        # __________________________________________________

        self.V = FunctionSpace(mesh, ("Lagrange", degree))


        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        for fl, x in enumerate(self.x_r):
            # print(fl,x)
            self._a[str(fl)] = self._assemble_left_vector(fl)
            self._b[str(fl)] = self._assemble_right_vector(x)

    @property
    def matrix(self):
        return self._D
    @property
    def submatrices(self):
        return self._D_kj
    @property
    def adjoint_submatrices(self):
        return self._D_kj_adj
    @property
    def adjoint_matrix(self):
        return self._D_adj
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b

    def _assemble_left_vector(self, fl):

        dx = Measure("dx", subdomain_data=self.subdomains)

        phi_k = self.v
        volume_form = form(Constant(self.mesh, PETSc.ScalarType(1))*dx(fl))
        V_fl = MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(volume_form), op=MPI.SUM)
        b = Function(self.V)
        b.x.array[:] = 0
        const = Constant(self.mesh, (PETSc.ScalarType(1/V_fl))) 
        gradient_form = form(inner(const, phi_k)*dx(fl))
        a = assemble_vector(b.vector, gradient_form)
        # print(a.array)
        b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        # print(b.x.array)
        indices1 = np.array(np.flatnonzero(a.array),dtype=np.int32)
        a = b.x.array
        dofmaps = self.V.dofmap
        global_indices = dofmaps.index_map.local_to_global(indices1)
        a = list(zip(global_indices, a[indices1]))
        
        # Parallelization        
        a = self.comm.gather(a, root=0)
        if a:
            a = [j for i in a for j in i]
            print("Length of left vector: ", len(a))
        else:
            a=[]
        # print("A", a, "Process", self.rank)

        return a

    def _assemble_right_vector(self, point):

        tdim = self.mesh.topology.dim
        if tdim==1:
            n = as_vector([1])
        elif tdim ==2:
            n = as_vector([1,0])
        else:
            n = as_vector([0,0,1])
        
        b = Function(self.V)
        b.x.array[:] = 0
        form_to_assemble = form(inner(n,grad(self.v)*self.w/self.rho)*ufl.dx)
        right_vector = assemble_vector(b.vector, form_to_assemble)
        b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        right_vector = b.x.array

        tol = 1e-5
        right_vector.real[abs(right_vector.real) < tol] = 0.0
        right_vector.imag[abs(right_vector.imag) < tol] = 0.0

        indices2 = np.array(np.flatnonzero(right_vector),dtype=np.int32)
        dofmaps = self.V.dofmap
        global_indices = dofmaps.index_map.local_to_global(indices2)
        right_vector = list(zip(global_indices, right_vector[indices2]))
        # Parallelization
        right_vector = self.comm.gather(right_vector, root=0)
        if self.rank==0:
            right_vector = [j for i in right_vector for j in i]
            print("Length of right vector: ", len(right_vector))
        else:
            right_vector=[]
        # print("B", right_vector, "Process", self.rank)
        return right_vector

    def assemble_submatrices(self, problem_type='direct'):
        
        global_size = self.V.dofmap.index_map.size_global
        local_size = self.V.dofmap.index_map.size_local

        if problem_type == 'direct':
            A = self._a[str(0)]
            B = self._b[str(0)]

        elif problem_type == 'adjoint':
            A = self._b[str(0)]
            B = self._a[str(0)]
        
        # print(A,B)
        row = [item[0] for item in A]
        col = [item[0] for item in B]

        row_vals = [item[1] for item in A]
        col_vals = [item[1] for item in B]

        product = np.outer(row_vals,col_vals) 
        # print(product)

        val = product.flatten()

        mat = PETSc.Mat().create(PETSc.COMM_WORLD) # MPI.COMM_SELF
        mat.setSizes([(local_size, global_size), (local_size, global_size)])
        mat.setType('aij') 
        mat.setUp()
        mat.setValues(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        if problem_type == 'direct':
            self._D_kj = mat
        elif problem_type == 'adjoint':
            self._D_kj_adj = mat

    def assemble_matrix(self, omega, problem_type='direct'):

        if problem_type == 'direct':

            z = self.FTF(omega)
            self._D = self._D_kj*z*self.coeff

        elif problem_type == 'adjoint':

            z = np.conj(self.FTF(np.conj(omega)))
            self._D_adj = self.coeff * z * self._D_kj_adj

    def get_derivative(self, omega):

        z = self.FTF(omega, k=1)
        dD_domega = z * self._D_kj
        dD_domega = self.coeff * dD_domega

        return dD_domega

    def blochify(self, problem_type='direct'):

        if problem_type == 'direct':

            D_kj_bloch = self.bloch_object.blochify(self.submatrices)
            self._D_kj = D_kj_bloch

        elif problem_type == 'adjoint':

            D_kj_adj_bloch = self.bloch_object.blochify(self.adjoint_submatrices)
            self._D_kj_adj = D_kj_adj_bloch
