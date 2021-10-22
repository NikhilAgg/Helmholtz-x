import dolfinx
import basix
from dolfinx import Function, FunctionSpace
from mpi4py import MPI
from ufl import Measure, FacetNormal, TestFunction, TrialFunction, dx, grad, inner
from petsc4py import PETSc
import numpy as np

class ActiveFlame:

    gamma = 1.4

    def __init__(self, mesh, subdomains, x_r, rho_u, Q, U, FTF, degree=1, comm=None,
                 constrained_domain=None):

        self.comm = comm

        self.mesh = mesh
        self.subdomains = subdomains
        self.x_r = x_r
        self.rho_u = rho_u
        self.Q = Q
        self.U = U
        self.FTF = FTF
        self.degree = degree

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
            print(fl,x)
            self._a[str(fl)] = self._assemble_left_vector(fl)
            self._b[str(fl)] = self._assemble_right_vector(x)

    @property
    def submatrices(self):
        return self._D_kj

    @property
    def matrix(self):
        return self._D
    @property
    def a(self):
        return self._a
    @property
    def b(self):
        return self._b        
    @property
    def adjoint_submatrices(self):
        return self._D_kj_adj

    @property
    def adjoint_matrix(self):
        return self._D_adj

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
            includes assembled elements of a_1 and a_2

        """

        dx = Measure("dx", subdomain_data=self.subdomains)

        v = self.v

        V_fl = MPI.COMM_WORLD.allreduce(dolfinx.fem.assemble_scalar(dolfinx.Constant(self.mesh, PETSc.ScalarType(1))*dx(fl)), op=MPI.SUM)
        b = dolfinx.Function(self.V)
        b.x.array[:] = 0
        const = dolfinx.Constant(self.mesh, (1/V_fl))
        a = dolfinx.fem.assemble_vector(b.vector, inner(const, v)*dx(fl))
        b.vector.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        indices1 = np.array(np.flatnonzero(a.getArray()),dtype=np.int32)
        a = b.x.array
        dofmaps = self.V.dofmap
        indices1 = dofmaps.index_map.local_to_global(indices1)
        a = list(zip(indices1, a[indices1]))
        print("A", a)
        return a

    def _assemble_right_vector(self, point):
        """
        Calculates degree of freedoms and indices of 
        right vector of
        
        \nabla(\phi_j(x_r)) . n
        
        which includes gradient value of test fuunction at
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
        num_local_cells = self.mesh.topology.index_map(tdim).size_local
        bb_tree = dolfinx.geometry.BoundingBoxTree(self.mesh, tdim, np.arange(num_local_cells, dtype=np.int32))
        cell_candidates = dolfinx.geometry.compute_collisions_point(bb_tree, point)
        # Choose one of the cells that contains the point
        cell = dolfinx.geometry.select_colliding_cells(self.mesh, cell_candidates, point, 1)

        # Data required for pull back of coordinate
        gdim = self.mesh.geometry.dim
        num_local_cells = self.mesh.topology.index_map(tdim).size_local
        num_dofs_x = self.mesh.geometry.dofmap.links(0).size  # NOTE: Assumes same cell geometry in whole mesh
        t_imap = self.mesh.topology.index_map(tdim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        x = self.mesh.geometry.x
        x_dofs = self.mesh.geometry.dofmap.array.reshape(num_cells, num_dofs_x)
        cell_geometry = np.zeros((num_dofs_x, gdim), dtype=np.float64)
        points_ref = np.zeros((1, tdim))

        # Data required for evaluation of derivative
        ct = dolfinx.cpp.mesh.to_string(self.mesh.topology.cell_type)
        element = basix.create_element(basix.finite_element.string_to_family(
                "Lagrange", ct), basix.cell.string_to_type(ct), self.degree, basix.LagrangeVariant.equispaced)
        dofmaps = self.V.dofmap
        coordinate_element = basix.create_element(basix.finite_element.string_to_family(
                "Lagrange", ct), basix.cell.string_to_type(ct), 1, basix.LagrangeVariant.equispaced)

        point_ref = None
        B = []
        if len(cell) > 0:
            cell = cell[0]
            # Only add contribution if cell is owned
            if cell < num_local_cells:
                # Map point in cell back to reference element
                cell_geometry[:] = x[x_dofs[cell], :gdim]
                point_ref = self.mesh.geometry.cmap.pull_back([point[:gdim]], cell_geometry)
                dphi = coordinate_element.tabulate(1, point_ref)[1:,0,:]
                J = np.dot(cell_geometry.T, dphi.T)
                Jinv = np.linalg.inv(J)  

                cell_dofs = dofmaps.cell_dofs(cell)
                global_dofs = dofmaps.index_map.local_to_global(cell_dofs)
                # Compute gradient on physical element by multiplying by J^(-T)
                d_dx = (Jinv.T @ element.tabulate(1, point_ref)[1:, 0, :]).T
                d_dv = np.dot(d_dx, v)[:, 0]
                for i in range(len(cell_dofs)):
                    B.append([global_dofs[i], d_dv[i]])
            else:
                print(MPI.COMM_WORLD.rank, "Ghost", cell) 
        root = -1
        if len(B) > 0:
            root = MPI.COMM_WORLD.rank
        b_root = MPI.COMM_WORLD.allreduce(root, op=MPI.MAX)
        B = MPI.COMM_WORLD.bcast(B, root=b_root)
        print("B ",B)
        return B

    @staticmethod
    def _csr_matrix(a, b):

        # len(a) and len(b) are not the same

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
        print("ROW: ",row,
        "COL: ",col,
        "VAL: ",val)
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
 
        print("LOCAL SIZE: ",local_size)

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
        
        # print("ROW: ",repr(row))
        # print("COLUMN: ",repr(col))
        # print("VAL: ",repr(val))
        if len(val)==0:
            
            mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD) #PETSc.COMM_WORLD
            mat.setSizes([(local_size, global_size), (local_size, global_size)])
            mat.setFromOptions()
            mat.setUp()
            mat.assemble()
            
        else:
            # indptr = np.bincount(row, minlength=local_size)
            # indptr = np.insert(indptr, 0, 0).cumsum()
            # indptr = indptr.astype(dtype='int32')
            mat = PETSc.Mat().create(PETSc.COMM_WORLD) # MPI.COMM_SELF
            mat.setSizes([(local_size, global_size), (local_size, global_size)])
            mat.setType('aij') 
            mat.setUp()
            # mat.setValuesCSR(indptr, col, val)
            for i in range(len(row)):
                mat.setValue(row[i],col[i],val[i], addv=PETSc.InsertMode.ADD_VALUES)
            mat.assemblyBegin()
            mat.assemblyEnd()

        # print(mat.getValues(range(global_size),range(global_size)))
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
            self._D_adj = self.coeff * z * self._D_adj

    def get_derivative(self, omega):

        """ Derivative of the unsteady heat release (flame) operator D
        wrt the eigenvalue (complex angular frequency) omega."""

        (D_11, D_12, D_21, D_22) = self._D_kj
        z = self.FTF(omega, k=1)
        dD_domega = mult_complex_scalar_real_matrix(z, D_11, D_12, D_21, D_22)
        dD_domega = self.coeff * dD_domega

        return dD_domega


# if __name__ == '__main__':