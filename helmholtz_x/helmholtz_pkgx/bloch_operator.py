from dolfinx.fem import locate_dofs_topological
from scipy.sparse import csr_matrix
import numpy as np
from petsc4py import PETSc

class Blochifier:
    def __init__(self, geometry, boundary_conditions, BlochNumber, passive_matrices,  active_matrix=None):
        self.passive_matrices = passive_matrices
        self.active_matrix = active_matrix

        self.f = np.exp(1j * 2 * np.pi / BlochNumber) # impose periodicity scalar (B is bloch elements)

        self.mesh = geometry.mesh
        self.facet_tags = geometry.facet_tags
        self.V = passive_matrices.V

        self._A = None
        self._B = None
        self._B_adj = None
        self._C = None
        self._D = None

        master_tag = list(boundary_conditions.keys())[list(boundary_conditions.values()).index("Master")]
        slave_tag  = list(boundary_conditions.keys())[list(boundary_conditions.values()).index("Slave")]
        fdim = self.mesh.topology.dim - 1
        facets_master = np.array(self.facet_tags.indices[self.facet_tags.values == master_tag])
        dofs_master = locate_dofs_topological(self.V, fdim, facets_master)

        facets_slave = np.array(self.facet_tags.indices[self.facet_tags.values == slave_tag])
        dofs_slave = locate_dofs_topological(self.V, fdim, facets_slave)
        #assert len(dofs_master) ==  len(dofs_slave)

        boundary_map_points = np.vstack([dofs_master, dofs_slave])

        x_size, y_size = passive_matrices.A.createVecs()
        self.N = len(x_size.array)

        Nb = len(dofs_slave)

        direct_map_points = [item for item in range(self.N) if item not in dofs_master]
        direct_map_points = np.vstack([direct_map_points, direct_map_points])

        # Make matrix that maps from circulant points to neumann points
        ii = np.concatenate((direct_map_points[0, :], boundary_map_points[0, :]))
        jj = np.concatenate((direct_map_points[1, :], boundary_map_points[1, :]))

        BN_csr = csr_matrix((np.full(self.N, 1), (ii, jj)), shape=(self.N, self.N), dtype=np.complex128)
        NB_csr = csr_matrix((np.full(self.N, 1), (jj, ii)), shape=(self.N, self.N), dtype=np.complex128)

        for nn in range(Nb):
            x_ = boundary_map_points[0][nn]
            y_ = boundary_map_points[1][nn]
            BN_csr[x_, y_] = self.f

            x__ = boundary_map_points[1][nn]
            y__ = boundary_map_points[0][nn]
            NB_csr[x__, y__] = 1 / self.f

        self.BN_csr = self._dropcols_fancy(BN_csr, dofs_master)
        self.NB_csr = self._droprows_fancy(NB_csr, dofs_master)
        self._BN = PETSc.Mat().createAIJ(size=self.BN_csr.shape, csr=(self.BN_csr.indptr, self.BN_csr.indices, self.BN_csr.data))

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def B_adj(self):
        return self._B_adj

    @property
    def C(self):
        return self._C

    @property
    def D(self):
        return self._D

    @property
    def remapper(self):
        return self._BN

    def blochify(self, matrix):

        ai, aj, av = matrix.getValuesCSR()
        matrix_csr = csr_matrix((av, aj, ai), shape=(self.N, self.N))
        bloch_matrix = self.NB_csr * matrix_csr * self.BN_csr
        petsc_matrix = PETSc.Mat().createAIJ(size=bloch_matrix.shape,
                                             csr=(bloch_matrix.indptr, bloch_matrix.indices, bloch_matrix.data))
        return petsc_matrix

    @staticmethod
    def _dropcols_fancy(M, idx_to_drop):
        idx_to_drop = np.unique(idx_to_drop)
        keep = ~np.in1d(np.arange(M.shape[1]), idx_to_drop, assume_unique=True)
        return M[:, np.where(keep)[0]]

    @staticmethod
    def _droprows_fancy(M, idx_to_drop):
        idx_to_drop = np.unique(idx_to_drop)
        keep = ~np.in1d(np.arange(M.shape[0]), idx_to_drop, assume_unique=True)
        return M[np.where(keep)[0], :]

    def blochify_A(self):
        A = self.blochify(self.passive_matrices.A)
        self._A = A

    def blochify_B(self):
        B = self.blochify(self.passive_matrices.B)
        self._B = B

    def blochify_C(self):
        C = self.blochify(self.passive_matrices.C)
        self._C = C

    # def blochify_D(self):
    #     D = self.blohicfy(self.active_matrix.submatrices)
    #     self._D = D
