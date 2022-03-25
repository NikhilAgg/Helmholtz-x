import dolfinx
from dolfinx.fem import Function, FunctionSpace, dirichletbc, form
from mpi4py import MPI
from ufl import Measure, FacetNormal, TestFunction, TrialFunction, dx, grad, inner
from petsc4py import PETSc
import numpy as np
from dolfinx_mpc import MultiPointConstraint, assemble_matrix
from dolfinx.mesh import MeshTags

class PassiveFlame:

    def __init__(self, mesh, facet_tags, boundary_conditions,
                 c, periodic_relation, BlochNumber, degree=1):
        """
        

        This class defines the matrices A,B,C in
        A + wB + w^2 C = 0 for the boundary conditions that consist
        Robin BC.
    
        Parameters
        ----------
        mesh : dolfinx.cpp.mesh.Mesh
            mesh of the domain
        facet_tags : dolfinx.cpp.mesh.MeshFunctionSizet
            boundary data of the mesh
        boundary_conditions : dict
            boundary conditions for corresponding mesh.
        c : Expression
            Speed of sound
        degree : int, optional
            degree of the basis functions. The default is 1.
    
        Returns
        -------
        A : petsc4py.PETSc.Mat
            Matrix of Grad term
        B : petsc4py.PETSc.Mat
            Empty Matrix
        C : petsc4py.PETSc.Mat
            Matrix of w^2 term.

        """

        self.mesh = mesh
        self.facet_tag = facet_tags
        self.fdim = mesh.topology.dim - 1
        self.boundary_conditions = boundary_conditions
        self.c = c
        self.degree = degree
        self.periodic_relation = periodic_relation

        self.dx = Measure('dx', domain=mesh)
        self.ds = Measure('ds', domain=mesh, subdomain_data=facet_tags)


        self.V = FunctionSpace(mesh, ("Lagrange", degree))


        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)


        self.bcs = []
        self.integrals_R = []
        for i in boundary_conditions:
            if 'Dirichlet' in boundary_conditions[i]:
                u_bc = Function(self.V)
                u_bc.interpolate(lambda x:x[0]*0)
                u_bc.x.scatter_forward()
                facets = np.array(self.facet_tag.indices[self.facet_tag.values == i])
                dofs = dolfinx.fem.locate_dofs_topological(self.V, self.fdim, facets)
                bc = dirichletbc(u_bc, dofs)
                self.bcs.append(bc)
            if 'Robin' in boundary_conditions[i]:
                Y = boundary_conditions[i]['Robin']
                integral_R = 1j * Y * self.c * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integral_R) 
            if 'Bloch' in boundary_conditions[i]:
                facets = np.array(self.facet_tag.indices[self.facet_tag.values == i])
                fdim = self.mesh.topology.dim - 1
                self.bloch_tag = i
                self.mt = MeshTags(mesh, fdim, facets, np.full(len(facets), i, dtype=np.int32))

        self.mpc = MultiPointConstraint(self.V)
        # We define a scaler to impose periodicity
        scaler = PETSc.ScalarType(np.exp(1j*2*np.pi/BlochNumber))
        self.mpc.create_periodic_constraint_topological(self.mt, self.bloch_tag, periodic_relation, self.bcs, scale = scaler)
        self.mpc.finalize()

        self._A = None
        self._B = None
        self._B_adj = None
        self._C = None


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


    def assemble_A(self):

        # defining ufl forms
        a = form(-self.c**2 * inner(grad(self.u), grad(self.v))*self.dx)
        A = assemble_matrix(a, self.mpc, bcs=self.bcs)
        self._A = A

    def assemble_B(self):

        N = self.V.dofmap.index_map.size_global
        n = self.V.dofmap.index_map.size_local


        if self.integrals_R:
            B = assemble_matrix(form(sum(self.integrals_R)), self.mpc)
            

        else:

            B = PETSc.Mat().create()
            B.setSizes([(n, N), (n, N)])
            B.setFromOptions()
            B.setUp()
            B.assemble()

        B_adj = B.copy()
        B_adj.transpose()
        B_adj.conjugate()

        self._B = B
        self._B_adj = B_adj    

    def assemble_C(self):

        c = form(inner(self.u , self.v) * self.dx)
        C = assemble_matrix(c, self.mpc, self.bcs)
        self._C = C


# if __name__ == '__main__':
