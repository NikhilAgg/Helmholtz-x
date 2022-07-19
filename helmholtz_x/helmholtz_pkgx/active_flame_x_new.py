import dolfinx
from dolfinx.fem  import Function, FunctionSpace, Constant, form
from dolfinx.fem.petsc import assemble_vector
from mpi4py import MPI
from ufl import Measure, FacetNormal, TestFunction, TrialFunction, inner, as_vector, grad
import ufl
from petsc4py import PETSc
import numpy as np
from math import ceil

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
        self.size = self.comm.Get_size()

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
        # a = list(zip(global_indices, a[indices1]))
        a = list(map(list, zip(global_indices, a[indices1])))

        # # New method - Parallelization        
        # a = self.comm.gather(a, root=0)
        
        # if self.rank == 0:
        #     a = [j for i in a for j in i]
        #     print("Before chunking: \n",a)
        #     print("Length of left vector: ", len(a))
        #     # dividing data into chunks
        #     chunks = [[] for _ in range(self.size)]
        #     for i, chunk in enumerate(a):
        #         chunks[i % self.size].append(chunk)
        # else:
        #     a = None 
        #     chunks = None          
        # a = self.comm.scatter(chunks, root=0)
        # print ("process", self.rank, "a_local:", a)
        # return a

        # Parallelization        
        a = self.comm.gather(a, root=0)
        if a:
            a = [j for i in a for j in i]
            # print("Before broadcasting of A :", a)
        else:
            a=[]
        a = self.comm.bcast(a,root=0)


        return a

        # Parallelization        
        a = self.comm.gather(a, root=0)
        if a:
            a = [j for i in a for j in i]
            print("Length of left vector: ", len(a))
        else:
            a=[]
        print("A", a, "Process", self.rank)

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

        # New method - Parallelization        
        right_vector = self.comm.gather(right_vector, root=0)
        
        if self.rank == 0:
            right_vector = [j for i in right_vector for j in i]
            # print("Before chunking of B: \n",right_vector)
            # dividing data into chunks
            chunks = [[] for _ in range(self.size)]
            for i, chunk in enumerate(right_vector):
                chunks[i % self.size].append(chunk)
        else:
            right_vector = None 
            chunks = None          
        right_vector = self.comm.scatter(chunks, root=0)
        return right_vector

        # Parallelization
        right_vector = self.comm.gather(right_vector, root=0)
        if self.rank==0:
            right_vector = [j for i in right_vector for j in i]
            print("Length of right vector: ", len(right_vector))
        else:
            right_vector=[]
        print("B", right_vector, "Process", self.rank)
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
        
        # print("A_local", A, "Process: ", self.rank)
        # print("B_local", B, "Process:", self.rank)


        # print(A,B)
        row = [item[0] for item in A]
        col = [item[0] for item in B]
        
        row_vals = [item[1] for item in A]
        col_vals = [item[1] for item in B]

        product = np.outer(row_vals,col_vals) 
        # print(product,"process", self.rank)

        # print("ROWS: ", len(row), self.rank)
        # print("COLS: ", col, self.rank)

        val = product.flatten()
        mat = PETSc.Mat().create(PETSc.COMM_WORLD) 
        mat.setSizes([(local_size, global_size), (local_size, global_size)])
        mat.setType('mpiaij')
        NNZ = len(row)
        NNZ1 = 1*len(row)*np.ones(local_size,dtype=np.int32) 
        mat.setPreallocationNNZ([NNZ,NNZ])
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        mat.setUp()
        mat.setValues(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        # from scipy.sparse import csr_matrix
        # ai, aj, av = mat.getValuesCSR()
        # CSR = csr_matrix((av, aj, ai), shape=(global_size,global_size))
        # import matplotlib.pyplot as plt
        # plt.spy(CSR)
        # plt.savefig("CSR.pdf")

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

class ActiveFlameNT:

    gamma = 1.4

    def __init__(self, mesh, subdomains, w, rho, Q, U, n, tau, omega, x_r, degree=1):

        self.mesh = mesh
        self.subdomains = subdomains
        self.w = w
        self.rho = rho
        self.x_r = x_r
        self.Q = Q
        self.U = U
        self.n = n
        self.tau = tau
        self.omega = omega
        self.degree = degree
        

        self.coeff = (self.gamma - 1) * Q / U

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

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

        n = self.n
        tau = self.tau 
        tau_func = Function(FunctionSpace(self.mesh, ("Lagrange", self.degree)))
        tau_func.x.array[:] = np.exp(self.omega*1j*tau.x.array) 

        gradient_form = form(n * tau_func * inner(const, phi_k) * dx(fl))        
        a = assemble_vector(b.vector, gradient_form)
        b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        indices1 = np.array(np.flatnonzero(a.array),dtype=np.int32)
        a = b.x.array
        dofmaps = self.V.dofmap
        global_indices = dofmaps.index_map.local_to_global(indices1)
        a = list(map(list, zip(global_indices, a[indices1]))) 

        # Parallelization        
        a = self.comm.gather(a, root=0)
        if a:
            a = [j for i in a for j in i]
        else:
            a=[]
        a = self.comm.bcast(a,root=0)


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

        # New method - Parallelization        
        right_vector = self.comm.gather(right_vector, root=0)
        
        if self.rank == 0:
            right_vector = [j for i in right_vector for j in i]
            # print("Before chunking of B: \n",right_vector)
            # dividing data into chunks
            chunks = [[] for _ in range(self.size)]
            for i, chunk in enumerate(right_vector):
                chunks[i % self.size].append(chunk)
        else:
            right_vector = None 
            chunks = None          
        right_vector = self.comm.scatter(chunks, root=0)
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

        row = [item[0] for item in A]
        col = [item[0] for item in B]
        
        row_vals = [item[1] for item in A]
        col_vals = [item[1] for item in B]

        product = np.outer(row_vals,col_vals) 

        val = product.flatten()
        mat = PETSc.Mat().create(PETSc.COMM_WORLD) 
        mat.setSizes([(local_size, global_size), (local_size, global_size)])
        mat.setType('mpiaij')
        NNZ = len(row)
        mat.setPreallocationNNZ([NNZ,NNZ])
        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        mat.setUp()
        mat.setValues(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        if problem_type == 'direct':
            self._D = mat*self.coeff
        elif problem_type == 'adjoint':
            self._D = mat*self.coeff