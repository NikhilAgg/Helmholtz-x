import dolfinx
from dolfinx import Function, FunctionSpace, UnitIntervalMesh, fem
from dolfinx.fem.assemble import assemble_matrix
from mpi4py import MPI
from ufl import Measure, FacetNormal, TestFunction, TrialFunction, dx, grad, inner
from petsc4py import PETSc
import numpy as np

class PassiveFlame:

    def __init__(self, mesh, facet_tags, boundary_conditions,
                 c, degree=1, constrained_domain=None):
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
        constrained_domain : TYPE, optional
            DESCRIPTION. The default is None.
    
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


        self.dx = Measure('dx', domain=mesh)
        self.ds = Measure('ds', domain=mesh, subdomain_data=facet_tags)


        self.V = FunctionSpace(mesh, ("Lagrange", degree))


        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)


        self.bcs = []
        self.integrals_R = []
        for i in boundary_conditions:
            if 'Dirichlet' in boundary_conditions[i]:
                u_bc = dolfinx.Function(self.V)
                u_bc.interpolate(lambda x:x[0]*0)
                u_bc.x.scatter_forward()
                facets = np.array(self.facet_tag.indices[self.facet_tag.values == i])
                dofs = dolfinx.fem.locate_dofs_topological(self.V, self.fdim, facets)
                bc = dolfinx.DirichletBC(u_bc, dofs)
                self.bcs.append(bc)
            if 'Robin' in boundary_conditions[i]:
                Y = boundary_conditions[i]['Robin']
                integral_R = 1j * Y * self.c * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integral_R) 

        self._A = None
        self._B = None
        self._B_astuple = None
        self._B_adj = None
        self._B_adj_astuple = None
        self._C = None
        self._C_astuple = None

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
        a = -self.c**2 * inner(grad(self.u), grad(self.v))*self.dx

        A = assemble_matrix(a, self.bcs)
        A.assemble()

        self._A = A

    def assemble_B(self):

        N = self.V.dofmap.index_map.size_global
        n = self.V.dofmap.index_map.size_local


        if self.integrals_R:

            B = assemble_matrix(sum(self.integrals_R))
            B.assemble()

        else:

            B = PETSc.Mat().create()
            B.setSizes([(n, N), (n, N)])
            B.setFromOptions()
            B.setUp()
            B.assemble()

        B_adj = B.copy()
        B.transpose(B_adj)

        self._B = B
        self._B_adj = B_adj

    def assemble_zB(self, z):

        if self._B is None:
            self.assemble_B()

        Bz = self._B * z

        return Bz            

    def assemble_C(self):

        c = inner(self.u , self.v) * self.dx
        C = assemble_matrix(c, self.bcs)
        C.assemble()
        self._C = C

    def assemble_zC(self, z):

        if self._C is None:
            self.assemble_C()
            
        Cz = self._C * z

        return Cz 

# if __name__ == '__main__':