from dolfinx.fem  import Function, FunctionSpace, Constant, form, assemble_scalar
from dolfinx.fem.petsc import assemble_vector
from dolfinx.geometry import compute_collisions, compute_colliding_cells, BoundingBoxTree
from .solver_utils import info
from .parameters_utils import gamma_function
from mpi4py import MPI
from ufl import Measure, TestFunction, TrialFunction, inner, as_vector, grad
from petsc4py import PETSc
import basix
import numpy as np
import ufl

class ActiveFlame:

    gamma = 1.4

    def __init__(self, mesh, subdomains, x_r, rho_u, Q, U, FTF, degree=1, bloch_object=None):

        self.mesh = mesh
        self.subdomains = subdomains
        self.x_r = x_r
        self.rho_u = rho_u
        self.Q = Q
        self.U = U
        self.FTF = FTF
        self.degree = degree
        self.bloch_object = bloch_object

        self.coeff = (self.gamma - 1) / rho_u * Q / U

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
        """
        Assembles \int v(x) \phi_k dV
        Parameters
        ----------
        fl : int
            flame tag
        Returns
        -------
        v : <class 'tuple'>
            includes assembled nonzero elements of a and row indices
        """

        dx = Measure("dx", subdomain_data=self.subdomains)

        phi_k = self.v
        volume_form = form(Constant(self.mesh, PETSc.ScalarType(1))*dx(fl))
        V_fl = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
        b = Function(self.V)
        b.x.array[:] = 0
        const = Constant(self.mesh, (PETSc.ScalarType(1/V_fl))) 
        gradient_form = form(inner(const, phi_k)*dx(fl))
        a = assemble_vector(b.vector, gradient_form)
        b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        indices1 = np.array(np.flatnonzero(a.array),dtype=np.int32)
        a = b.x.array
        dofmaps = self.V.dofmap
        global_indices = dofmaps.index_map.local_to_global(indices1)
        a = list(zip(global_indices, a[indices1]))

        return a

    def _assemble_right_vector(self, point):
        """
        Calculates degree of freedoms and column indices of 
        right vector of
        
        \nabla(\phi_j(x_r)) . n
        
        which includes gradient value of test function at
        reference point x_r
        
        Parameters
        ----------
        x : np.array
            flame location vector
        Returns
        -------
        np.array
            Array of degree of freedoms and indices as vector b.
        """
        tdim = self.mesh.topology.dim

        v = np.array([[0, 0, 1]]).T
        if tdim == 1:
            v = np.array([[1]])
        elif tdim == 2:
            v = np.array([[1, 0]]).T

        # Finds the basis function's derivative at point x
        # and returns the relevant dof and derivative as a list
        bb_tree = BoundingBoxTree(self.mesh, tdim)
        cell_candidates = compute_collisions(bb_tree, point)

        # Choose one of the cells that contains the point
        if tdim == 1:
            if len(cell_candidates.array)>0: # For 1D Parallel Runs
                cell = [cell_candidates.array[0]]
            else:
                cell = []
        else:
            cell = compute_colliding_cells(self.mesh, cell_candidates, point)

        # Data required for pull back of coordinate
        gdim = self.mesh.geometry.dim
        num_local_cells = self.mesh.topology.index_map(tdim).size_local
        num_dofs_x = self.mesh.geometry.dofmap.links(0).size  # NOTE: Assumes same cell geometry in whole mesh
        t_imap = self.mesh.topology.index_map(tdim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        
        x = self.mesh.geometry.x
        x_dofs = self.mesh.geometry.dofmap.array.reshape(num_cells, num_dofs_x)
        cell_geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
        points_ref = np.zeros((1, tdim)) #INTERESTINGLY REQUIRED FOR PARALLEL RUNS

        # Data required for evaluation of derivative
        ct = self.mesh.topology.cell_type
        element = basix.create_element(basix.finite_element.string_to_family( #INTERESTINGLY REQUIRED FOR PARALLEL RUNS
        "Lagrange", ct.name), basix.cell.string_to_type(ct.name), self.degree, basix.LagrangeVariant.equispaced)
        dofmaps = self.V.dofmap
        coordinate_element = basix.create_element(basix.finite_element.string_to_family(
                "Lagrange", ct.name), basix.cell.string_to_type(ct.name), 1, basix.LagrangeVariant.equispaced)

        point_ref = None
        B = []
        if len(cell) > 0:
            # Only add contribution if cell is owned
            cell = cell[0]
            
            if cell < num_local_cells:
                cell_geometry[:] = x[x_dofs[cell], :gdim]
                point_ref = self.mesh.geometry.cmap.pull_back([point[:gdim]], cell_geometry)
                dphi = coordinate_element.tabulate(1, point_ref)[1:,0,:]
                dphi = dphi.reshape((dphi.shape[0], dphi.shape[1]))
                J = np.dot(cell_geometry.T, dphi.T)
                Jinv = np.linalg.inv(J)  

                cell_dofs = dofmaps.cell_dofs(cell)
                global_dofs = dofmaps.index_map.local_to_global(cell_dofs)
                d_dx = (Jinv.T @ dphi).T
                d_dv = np.dot(d_dx, v)[:, 0]
                for i in range(len(d_dv)):
                    B.append([global_dofs[i], d_dv[i]])
                    
            else:
                print(MPI.COMM_WORLD.rank, "Ghost", cell) 
        root = 0 #it was -1
        if len(B) > 0:
            root = MPI.COMM_WORLD.rank
        b_root = MPI.COMM_WORLD.allreduce(root, op=MPI.MAX)
        B = MPI.COMM_WORLD.bcast(B, root=b_root)
        return B

    @staticmethod
    def _csr_matrix(a, b):
        """ This function gets row, column and values of (row.col) pairs of vector a and b

        Args:
            a ([type]): [description]
            b ([type]): [description]

        Returns:
            [type]: rows columns and values
        """
        nnz = len(a) * len(b)
        
        row = np.zeros(nnz)
        col = np.zeros(nnz)
        val = np.zeros(nnz, dtype=np.complex128)

        for i, c in enumerate(a):
            for j, d in enumerate(b):
                row[i * len(b) + j] = c[0]
                col[i * len(b) + j] = d[0]
                val[i * len(b) + j] = c[1] * d[1]

        row = row.astype(dtype='int32')
        col = col.astype(dtype='int32')

        return row, col, val

    def assemble_submatrices(self, problem_type='direct'):
        """
        This function handles efficient cross product of the 
        vectors a and b calculated above and generates highly sparse 
        matrix D_kj which represents active flame matrix without FTF and
        other constant multiplications.
        Parameters
        ----------
        problem_type : str, optional
            Specified problem type. The default is 'direct'.
            Matrix can be obtained by selecting problem type, other
            option is adjoint.
        
        """

        num_fl = len(self.x_r)  # number of flames
        global_size = self.V.dofmap.index_map.size_global
        local_size = self.V.dofmap.index_map.size_local

        row = dict()
        col = dict()
        val = dict()

        for fl in range(num_fl):

            u = None
            v = None

            if problem_type == 'direct':
                u = self._a[str(fl)]
                v = self._b[str(fl)]

            elif problem_type == 'adjoint':
                u = self._b[str(fl)]
                v = self._a[str(fl)]

            row[str(fl)], col[str(fl)], val[str(fl)] = self._csr_matrix(u, v)

        row = np.concatenate([row[str(fl)] for fl in range(num_fl)])
        col = np.concatenate([col[str(fl)] for fl in range(num_fl)])
        val = np.concatenate([val[str(fl)] for fl in range(num_fl)])

        i = np.argsort(row)

        row = row[i]
        col = col[i]
        val = val[i]

        if len(val)==0:
            
            mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD) #PETSc.COMM_WORLD
            mat.setSizes([(local_size, global_size), (local_size, global_size)])
            mat.setFromOptions()
            mat.setUp()
            mat.assemble()
            
        else:
            mat = PETSc.Mat().create(PETSc.COMM_WORLD) # MPI.COMM_SELF
            mat.setSizes([(local_size, global_size), (local_size, global_size)])
            mat.setType('aij') 
            mat.setUp()
            for i in range(len(row)):
                mat.setValue(row[i],col[i],val[i], addv=PETSc.InsertMode.ADD_VALUES)
            mat.assemblyBegin()
            mat.assemblyEnd()

        if problem_type == 'direct':
            self._D_kj = mat
        elif problem_type == 'adjoint':
            self._D_kj_adj = mat

    def assemble_matrix(self, omega, problem_type='direct'):
        """
        This function handles the multiplication of the obtained matrix D
        with Flame Transfer Function and other constants.
        The operation is
        
        D = (gamma - 1) / rho_u * Q / U * D_kj       
        
        At the end the matrix D is petsc4py.PETSc.Mat
        Parameters
        ----------
        omega : complex
            eigenvalue that found by solver
        problem_type : str, optional
            Specified problem type. The default is 'direct'.
            Matrix can be obtained by selecting problem type, other
            option is adjoint.
        Returns
        -------
        petsc4py.PETSc.Mat
        """

        if problem_type == 'direct':

            z = self.FTF(omega)
            self._D = self._D_kj*z*self.coeff
                        
        elif problem_type == 'adjoint':

            z = np.conj(self.FTF(np.conj(omega)))
            self._D_adj = self.coeff * z * self._D_kj_adj

    def get_derivative(self, omega):

        """ Derivative of the unsteady heat release (flame) operator D
        wrt the eigenvalue (complex angular frequency) omega."""

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

    def __init__(self, mesh, subdomains, w, h, rho, T, eta, tau, degree=1):

        self.mesh = mesh
        self.subdomains = subdomains
        self.w = w
        self.h = h
        self.rho = rho
        self.T = T
        self.eta = eta
        self.tau = tau
        self.degree = degree
        self.gamma = gamma_function(self.mesh, self.T)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.tol = 1e-5
        self.omega = 0

        self._a = None
        self._b = None
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.dofmaps = self.V.dofmap
        self.dimension = self.mesh.topology.dim

        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)

        self.dx = ufl.dx

        self._a = self._assemble_left_vector()
        self._b = self._assemble_right_vector()

    @property
    def submatrices(self):
        return self._D_kj
    @property
    def adjoint_submatrices(self):
        return self._D_kj_adj
    @property
    def matrix(self):
        return self._D
    @property
    def adjoint_matrix(self):
        return self._D_adj
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b

    def _indices_and_values(self, form):

        temp = Function(self.V)

        assemble_vector(temp.vector, form)
        temp.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        packed = temp.x.array
        packed.real[abs(packed.real) < self.tol] = 0.0
        packed.imag[abs(packed.imag) < self.tol] = 0.0

        indices = np.array(np.flatnonzero(packed),dtype=np.int32)
        global_indices = self.dofmaps.index_map.local_to_global(indices)
        packed = list(zip(global_indices, packed[indices]))

        return packed

    def _remove_repeating_dofs(self, vector):
        seen = set()
        keep = []
        for ind, value in vector:
            if ind in seen:
                pass
            else:
                seen.add(ind)
                keep.append((ind, value))
        return keep

    def _assemble_left_vector(self, Derivative=False):

        
        coefficient = (self.gamma - 1) # * q_tot / self.U_bulk

        if Derivative == False:
            form_to_assemble = form(coefficient *  self.phi_i  * self.eta * self.h * ufl.exp(1j*self.omega*self.tau)  *  self.dx)
        else:
            form_to_assemble = form(coefficient *  1j * self.tau * self.phi_i * self.h * self.n * ufl.exp(1j*self.omega*self.tau)  *  self.dx)

        left_vector = self._indices_and_values(form_to_assemble)

        # Parallelization
        left_vector = self.comm.gather(left_vector, root=0)
        if self.rank == 0:
            left_vector = [self._remove_repeating_dofs([item for sublist in left_vector for item in sublist])]
            left_vector = [j for i in left_vector for j in i]
            chunks = [[] for _ in range(self.size)]
            for i, chunk in enumerate(left_vector):
                chunks[i % self.size].append(chunk)
        else:
            left_vector = None
            chunks = None
        left_vector = self.comm.scatter(chunks, root=0)
        # print("Left vector: ", left_vector, "RANK: ", self.rank)
        return left_vector

    def _assemble_right_vector(self):

        if self.dimension == 1:
            n_ref = as_vector([1])
        elif self.dimension == 2:
            n_ref = as_vector([1,0])
        else:
            n_ref = as_vector([0,0,1])

        gradient_form = form(inner(n_ref,grad(self.phi_j)) / self.rho * self.w * self.dx)

        right_vector = self._indices_and_values(gradient_form)
        
        # Parallelization
        right_vector = self.comm.gather(right_vector, root=0)
        if right_vector:
            right_vector = [j for i in right_vector for j in i]
        else:
            right_vector=[]
        right_vector = self.comm.bcast(right_vector,root=0)

        right_vector = self._remove_repeating_dofs(right_vector)
        # print("Right vector: ", right_vector, "RANK: ", self.rank)
        return right_vector

    def assemble_submatrices(self, problem_type='direct'):

        if problem_type == 'direct':
            A = self._a
            B = self._b

        elif problem_type == 'adjoint':
            A = self._b
            B = self._a

        row = [item[0] for item in A]
        col = [item[0] for item in B]
        # print("ROW:", row, "RANK: ", self.rank)
        # print("COL:", col, "RANK: ", self.rank)
        row_vals = [item[1] for item in A]
        col_vals = [item[1] for item in B]
        # print("row_vals:", row_vals, "RANK: ", self.rank)
        # print("col_vals:", col_vals, "RANK: ", self.rank)
        product = np.outer(row_vals,col_vals)
        val = product.flatten()

        global_size = self.V.dofmap.index_map.size_global
        local_size = self.V.dofmap.index_map.size_local

        info("- Generating Matrix D..")

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(local_size, global_size), (local_size, global_size)])
        mat.setType('mpiaij')
        
        ONNZ = len(col)*np.ones(local_size,dtype=np.int32)
        mat.setPreallocationNNZ([ONNZ, ONNZ])

        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        mat.setUp()
        mat.setValues(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        # from scipy.sparse import csr_matrix
        # ai, aj, av = mat.getValuesCSR()
        # CSR = csr_matrix((av, aj, ai),shape=(local_size, global_size))
        # import matplotlib.pyplot as plt
        # plt.spy(CSR)
        # plt.savefig(f"CSR_{self.rank}.png")

        self._D = mat

    def assemble_matrix(self, omega, problem_type='direct'):
        self.omega = omega
        self._a = self._assemble_left_vector()
        self.assemble_submatrices(problem_type)
        info("- Matrix D is assembled.")

    def get_derivative(self, omega, problem_type='direct'):
        self.omega = omega
        self._a = self._assemble_left_vector(Derivative=True)
        self.assemble_submatrices(problem_type)
        info("- Derivative of D is assembled.")
        return self._D

class ActiveFlameNT4:

    def __init__(self, mesh, subdomains, w, h, rho, Q_tot, U_bulk, eta, tau, degree=1):

        self.mesh = mesh
        self.subdomains = subdomains
        self.w = w
        self.h = h
        self.rho = rho
        self.U_bulk = U_bulk
        self.Q_tot = Q_tot
        self.eta = eta
        self.tau = tau
        self.degree = degree
        self.gamma = 1.35

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.tol = 1e-5
        self.omega = 0

        self._a = None
        self._b = None
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.dofmaps = self.V.dofmap
        self.dimension = self.mesh.topology.dim

        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)

        self.dx = ufl.dx

        self._a = self._assemble_left_vector()
        self._b = self._assemble_right_vector()

    @property
    def submatrices(self):
        return self._D_kj
    @property
    def adjoint_submatrices(self):
        return self._D_kj_adj
    @property
    def matrix(self):
        return self._D
    @property
    def adjoint_matrix(self):
        return self._D_adj
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b

    def _indices_and_values(self, form):

        temp = Function(self.V)

        assemble_vector(temp.vector, form)
        temp.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        packed = temp.x.array
        packed.real[abs(packed.real) < self.tol] = 0.0
        packed.imag[abs(packed.imag) < self.tol] = 0.0

        indices = np.array(np.flatnonzero(packed),dtype=np.int32)
        global_indices = self.dofmaps.index_map.local_to_global(indices)
        packed = list(zip(global_indices, packed[indices]))

        return packed

    def _remove_repeating_dofs(self, vector):
        seen = set()
        keep = []
        for ind, value in vector:
            if ind in seen:
                pass
            else:
                seen.add(ind)
                keep.append((ind, value))
        return keep

    def _assemble_left_vector(self, Derivative=False):

        volume_form = form(Constant(self.mesh, PETSc.ScalarType(1))*self.dx)
        V_domain = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
        q_tot = Constant(self.mesh, (PETSc.ScalarType(self.Q_tot/V_domain)))

        coefficient = (self.gamma - 1) # * q_tot / self.U_bulk

        if Derivative == False:
            form_to_assemble = form(coefficient *  self.phi_i  * self.eta * self.h * ufl.exp(1j*self.omega*self.tau)  *  self.dx)
        else:
            form_to_assemble = form(coefficient *  1j * self.tau * self.phi_i * self.h * self.n * ufl.exp(1j*self.omega*self.tau)  *  self.dx)

        left_vector = self._indices_and_values(form_to_assemble)

        # Parallelization
        left_vector = self.comm.gather(left_vector, root=0)
        if self.rank == 0:
            left_vector = [self._remove_repeating_dofs([item for sublist in left_vector for item in sublist])]
            left_vector = [j for i in left_vector for j in i]
            chunks = [[] for _ in range(self.size)]
            for i, chunk in enumerate(left_vector):
                chunks[i % self.size].append(chunk)
        else:
            left_vector = None
            chunks = None
        left_vector = self.comm.scatter(chunks, root=0)
        # print("Left vector: ", left_vector, "RANK: ", self.rank)
        return left_vector

    def _assemble_right_vector(self):

        if self.dimension == 1:
            n_ref = as_vector([1])
        elif self.dimension == 2:
            n_ref = as_vector([1,0])
        else:
            n_ref = as_vector([0,0,1])

        gradient_form = form(inner(n_ref,grad(self.phi_j)) / self.rho * self.w * self.dx)

        right_vector = self._indices_and_values(gradient_form)
        
        # Parallelization
        right_vector = self.comm.gather(right_vector, root=0)
        if right_vector:
            right_vector = [j for i in right_vector for j in i]
        else:
            right_vector=[]
        right_vector = self.comm.bcast(right_vector,root=0)

        right_vector = self._remove_repeating_dofs(right_vector)
        # print("Right vector: ", right_vector, "RANK: ", self.rank)
        return right_vector

    def assemble_submatrices(self, problem_type='direct'):

        if problem_type == 'direct':
            A = self._a
            B = self._b

        elif problem_type == 'adjoint':
            A = self._b
            B = self._a

        row = [item[0] for item in A]
        col = [item[0] for item in B]
        # print("ROW:", row, "RANK: ", self.rank)
        # print("COL:", col, "RANK: ", self.rank)
        row_vals = [item[1] for item in A]
        col_vals = [item[1] for item in B]
        # print("row_vals:", row_vals, "RANK: ", self.rank)
        # print("col_vals:", col_vals, "RANK: ", self.rank)
        product = np.outer(row_vals,col_vals)
        val = product.flatten()

        global_size = self.V.dofmap.index_map.size_global
        local_size = self.V.dofmap.index_map.size_local

        info("- Generating Matrix D..")

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(local_size, global_size), (local_size, global_size)])
        mat.setType('mpiaij')
        
        ONNZ = len(col)*np.ones(local_size,dtype=np.int32)
        mat.setPreallocationNNZ([ONNZ, ONNZ])

        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        mat.setUp()
        mat.setValues(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        # from scipy.sparse import csr_matrix
        # ai, aj, av = mat.getValuesCSR()
        # CSR = csr_matrix((av, aj, ai),shape=(local_size, global_size))
        # import matplotlib.pyplot as plt
        # plt.spy(CSR)
        # plt.savefig(f"CSR_{self.rank}.png")

        self._D = mat

    def assemble_matrix(self, omega, problem_type='direct'):
        self.omega = omega
        self._a = self._assemble_left_vector()
        self.assemble_submatrices(problem_type)
        info("- Matrix D is assembled.")

    def get_derivative(self, omega, problem_type='direct'):
        self.omega = omega
        self._a = self._assemble_left_vector(Derivative=True)
        self.assemble_submatrices(problem_type)
        info("- Derivative of D is assembled.")
        return self._D
        
class ActiveFlameNT2:

    def __init__(self, mesh, subdomains, w, h, rho, Q, U, n, tau,x_f,a_f , degree=1):

        self.mesh = mesh
        self.subdomains = subdomains
        self.w = w
        self.h = h
        self.rho = rho
        self.U = U
        self.Q = Q
        self.n = n
        self.tau = tau
        self.x_f = x_f
        self.a_f = a_f
        self.gamma = 1.4
        self.degree = degree

        self.dx = Measure("dx", subdomain_data=self.subdomains)

        self.coeff = (self.gamma - 1) / U
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.tol = 1e-5
        self.omega = 0
        # __________________________________________________

        self._a = {}
        self._b = {}
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        # __________________________________________________

        self.V = FunctionSpace(mesh, ("Lagrange", degree))

        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)

        self.dofmaps = self.V.dofmap

        self.flame_tag = 0

        self._a[str(self.flame_tag)] = self._assemble_left_vector()
        self._b[str(self.flame_tag)] = self._assemble_right_vector()

    @property
    def submatrices(self):
        return self._D_kj
    @property
    def adjoint_submatrices(self):
        return self._D_kj_adj
    @property
    def matrix(self):
        return self._D
    @property
    def adjoint_matrix(self):
        return self._D_adj
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b

    def _indices_and_values(self, form):

        temp = Function(self.V)

        assemble_vector(temp.vector, form)
        temp.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        packed = temp.x.array
        packed.real[abs(packed.real) < self.tol] = 0.0
        packed.imag[abs(packed.imag) < self.tol] = 0.0

        indices = np.array(np.flatnonzero(packed),dtype=np.int32)
        global_indices = self.dofmaps.index_map.local_to_global(indices)
        packed = list(zip(global_indices, packed[indices]))

        return packed

    def _assemble_left_vector(self):

        from helmholtz_x.parameters_utils import Q_uniform, n_bump

        q_uniform = Q_uniform(self.mesh, self.subdomains, self.Q)
        q_tot = n_bump(self.mesh, self.x_f, self.a_f, q_uniform.value)

        coefficient = (self.gamma - 1) * q_tot / self.U
        tau = Function(self.V)
        tau.x.array[:] = np.exp( 1j*self.omega*self.tau.x.array)
        tau.x.scatter_forward()

        n = Constant(self.mesh, (PETSc.ScalarType(self.n)))

        form_to_assemble = form(coefficient * n * tau  * self.phi_i * self.dx)

        left_vector = self._indices_and_values(form_to_assemble)

        # Parallelization
        left_vector = self.comm.gather(left_vector, root=0)
        if left_vector:
            left_vector = [j for i in left_vector for j in i]
        else:
            left_vector=[]
        left_vector = self.comm.bcast(left_vector,root=0)

        return left_vector

    def _assemble_right_vector(self):

        tdim = self.mesh.topology.dim
        if tdim==1:
            n_ref = as_vector([1])
        elif tdim ==2:
            n_ref = as_vector([1,0])
        else:
            n_ref = as_vector([0,0,1])

        gradient_form = form(inner(n_ref,grad(self.phi_j)) / self.rho * self.w * self.dx)

        right_vector = self._indices_and_values(gradient_form)

        # Parallelization
        right_vector = self.comm.gather(right_vector, root=0)

        if self.rank == 0:
            right_vector = [j for i in right_vector for j in i]
            chunks = [[] for _ in range(self.size)]
            for i, chunk in enumerate(right_vector):
                chunks[i % self.size].append(chunk)
        else:
            right_vector = None
            chunks = None
        right_vector = self.comm.scatter(chunks, root=0)
        return right_vector

    def assemble_submatrices(self, problem_type='direct'):

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

        global_size = self.V.dofmap.index_map.size_global
        local_size = self.V.dofmap.index_map.size_local

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

        self._D = mat

    def assemble_matrix(self, omega, problem_type='direct'):
        self.omega = omega
        self._a[str(self.flame_tag)] = self._assemble_left_vector()
        self.assemble_submatrices(problem_type)
        if self.rank==0:
            print("- Matrix D is assembled.")


class ActiveFlameNT3:

    def __init__(self, mesh, subdomains, w, h, rho, Q_tot, U_bulk, n, tau, degree=1):

        self.mesh = mesh
        self.subdomains = subdomains
        self.w = w
        self.h = h
        self.rho = rho
        self.U_bulk = U_bulk
        self.Q_tot = Q_tot
        self.n = n
        self.tau = tau
        self.degree = degree
        self.gamma = 1.4

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.tol = 1e-5
        self.omega = Constant(mesh,(PETSc.ScalarType(0)))

        self._a = None
        self._b = None
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.dofmaps = self.V.dofmap
        self.dimension = self.mesh.topology.dim

        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)

        self.dx = ufl.dx

        self.D = self._assemble()
        

    def _assemble(self):

        volume_form = form(Constant(self.mesh, PETSc.ScalarType(1))*self.dx)
        V_domain = MPI.COMM_WORLD.allreduce(assemble_scalar(volume_form), op=MPI.SUM)
        q_tot = Constant(self.mesh, (PETSc.ScalarType(self.Q_tot/V_domain)))

        coefficient = (self.gamma - 1) * q_tot / self.U_bulk

        exp_tau = ufl.exp(1j*self.omega*self.tau)

        if self.dimension == 1:
            n_ref = as_vector([1])
        elif self.dimension == 2:
            n_ref = as_vector([1,0])
        else:
            n_ref = as_vector([0,0,1])

        total_form = form(coefficient *  self.phi_i * self.h * self.n * exp_tau *  inner(n_ref,grad(self.phi_j) / self.rho * self.w) * self.dx )

        from dolfinx.fem.petsc import assemble_matrix
        D = assemble_matrix(total_form)
        D.assemble()
        
        from scipy.sparse import csr_matrix
        ai, aj, av = D.getValuesCSR()
        CSR = csr_matrix((av, aj, ai))
        import matplotlib.pyplot as plt
        plt.spy(CSR)
        plt.savefig(f"CSR_NT3.png")
        plt.clf()

        return D


    def assemble_matrix(self, omega, problem_type='direct'):
        self.omega.value = omega
        print("OMEGA: ", self.omega.value)
        if self.rank==0:
            print("- Matrix D is assembled.")
        self.matrix = self.D




class ActiveFlameRR:

    def __init__(self, mesh, w, h, rho, eta, tau, degree=1):

        self.mesh = mesh
        self.w = w
        self.h = h
        self.rho = rho
        self.eta = eta
        self.tau = tau
        self.degree = degree
        self.gamma = 1.4

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.tol = 1e-5
        self.omega = 0

        self._a = None
        self._b = None
        self._D_kj = None
        self._D_kj_adj = None
        self._D = None
        self._D_adj = None

        self.V = FunctionSpace(mesh, ("Lagrange", degree))
        self.dofmaps = self.V.dofmap
        self.dimension = self.mesh.topology.dim

        self.phi_i = TrialFunction(self.V)
        self.phi_j = TestFunction(self.V)

        self.dx = ufl.dx

        self._a = self._assemble_left_vector()
        self._b = self._assemble_right_vector()

    @property
    def submatrices(self):
        return self._D_kj
    @property
    def adjoint_submatrices(self):
        return self._D_kj_adj
    @property
    def matrix(self):
        return self._D
    @property
    def adjoint_matrix(self):
        return self._D_adj
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b

    def _indices_and_values(self, form):

        temp = Function(self.V)

        assemble_vector(temp.vector, form)
        temp.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        packed = temp.x.array
        packed.real[abs(packed.real) < self.tol] = 0.0
        packed.imag[abs(packed.imag) < self.tol] = 0.0

        indices = np.array(np.flatnonzero(packed),dtype=np.int32)
        global_indices = self.dofmaps.index_map.local_to_global(indices)
        packed = list(zip(global_indices, packed[indices]))

        return packed

    def _remove_repeating_dofs(self, vector):
        seen = set()
        keep = []
        for ind, value in vector:
            if ind in seen:
                pass
            else:
                seen.add(ind)
                keep.append((ind, value))
        return keep

    def _assemble_left_vector(self, Derivative=False):

        coefficient = (self.gamma - 1) # * q_tot / self.U_bulk

        if Derivative == False:
            form_to_assemble = form(coefficient *  self.phi_i  * self.eta * self.h * ufl.exp(1j*self.omega*self.tau)  *  self.dx)
        else:
            form_to_assemble = form(coefficient *  1j * self.tau * self.phi_i * self.h * self.n * ufl.exp(1j*self.omega*self.tau)  *  self.dx)

        left_vector = self._indices_and_values(form_to_assemble)

        # Parallelization
        left_vector = self.comm.gather(left_vector, root=0)
        if self.rank == 0:
            left_vector = [self._remove_repeating_dofs([item for sublist in left_vector for item in sublist])]
            left_vector = [j for i in left_vector for j in i]
            chunks = [[] for _ in range(self.size)]
            for i, chunk in enumerate(left_vector):
                chunks[i % self.size].append(chunk)
        else:
            left_vector = None
            chunks = None
        left_vector = self.comm.scatter(chunks, root=0)
        
        return left_vector

    def _assemble_right_vector(self):

        if self.dimension == 1:
            n_ref = as_vector([1])
        elif self.dimension == 2:
            n_ref = as_vector([1,0])
        else:
            n_ref = as_vector([0,0,1])

        gradient_form = form(inner(n_ref,grad(self.phi_j)) / self.rho * self.w * self.dx)

        right_vector = self._indices_and_values(gradient_form)

        # Parallelization
        right_vector = self.comm.gather(right_vector, root=0)
        if right_vector:
            right_vector = [j for i in right_vector for j in i]
        else:
            right_vector=[]
        right_vector = self.comm.bcast(right_vector,root=0)

        right_vector = self._remove_repeating_dofs(right_vector)
        
        return right_vector

    def assemble_submatrices(self, problem_type='direct'):

        if problem_type == 'direct':
            A = self._a
            B = self._b

        elif problem_type == 'adjoint':
            A = self._b
            B = self._a

        row = [item[0] for item in A]
        col = [item[0] for item in B]
        row_vals = [item[1] for item in A]
        col_vals = [item[1] for item in B]

        product = np.outer(row_vals,col_vals)
        val = product.flatten()
        

        global_size = self.V.dofmap.index_map.size_global
        local_size = self.V.dofmap.index_map.size_local

        info("- Generating Matrix D..")

        mat = PETSc.Mat().create(PETSc.COMM_WORLD)
        mat.setSizes([(local_size, global_size), (local_size, global_size)])
        mat.setType('mpiaij')
        
        ONNZ = len(col)*np.ones(local_size,dtype=np.int32)
        mat.setPreallocationNNZ([ONNZ, ONNZ])

        mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        mat.setUp()
        mat.setValues(row, col, val, addv=PETSc.InsertMode.ADD_VALUES)
        mat.assemblyBegin()
        mat.assemblyEnd()

        # from scipy.sparse import csr_matrix
        # ai, aj, av = mat.getValuesCSR()
        # CSR = csr_matrix((av, aj, ai),shape=(local_size, global_size))
        # import matplotlib.pyplot as plt
        # plt.spy(CSR)
        # plt.savefig(f"CSR_{self.rank}.png")

        self._D = mat

    def assemble_matrix(self, omega, problem_type='direct'):
        self.omega = omega
        self._a = self._assemble_left_vector()
        self.assemble_submatrices(problem_type)
        del self._a
        info("- Matrix D is assembled.")

    def get_derivative(self, omega, problem_type='direct'):
        self.omega = omega
        self._a = self._assemble_left_vector(Derivative=True)
        self.assemble_submatrices(problem_type)
        info("- Derivative of D is assembled.")
        return self._D