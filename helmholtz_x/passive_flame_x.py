from dolfinx.fem import Function, FunctionSpace, dirichletbc, form, locate_dofs_topological
from dolfinx.fem.petsc import assemble_matrix
from .dolfinx_utils import info
from ufl import Measure, TestFunction, TrialFunction, grad, inner
from petsc4py import PETSc
import numpy as np

class PassiveFlame:

    def __init__(self, mesh, facet_tags, boundary_conditions,
                 c, degree=1):
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
        c : dolfinx.fem.function.Function
            Speed of sound
        degree : int, optional
            degree of the basis functions. The default is 1.
    
        Returns
        -------
        A : petsc4py.PETSc.Mat
            Matrix of Grad term
        B : petsc4py.PETSc.Mat
            Matrix for Robin boundary condition
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
            gamma = 1.33
            M_i = -0.10 
            M_o = 1.0
            if 'Dirichlet' in boundary_conditions[i]:
                u_bc = Function(self.V)
                facets = np.array(self.facet_tag.indices[self.facet_tag.values == i])
                dofs = locate_dofs_topological(self.V, self.fdim, facets)
                bc = dirichletbc(u_bc, dofs)
                self.bcs.append(bc)

            if 'Robin' in boundary_conditions[i]:
                R = boundary_conditions[i]['Robin']
                Z = (1+R)/(1-R)
                integrals_Impedance = 1j * self.c / Z * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integrals_Impedance)   

            if 'InletChoked' in boundary_conditions[i]:
                print("InletChoked BC is working")
                integral_C_i = -1j * M_i * c * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integral_C_i) 
                # u_bc = Function(self.V)
                # u_bc.x.array[:] = 1.0
                # u_bc.x.scatter_forward()
                # facets = np.array(self.facet_tag.indices[self.facet_tag.values == i])
                # dofs = locate_dofs_topological(self.V, self.fdim, facets)
                # bc = dirichletbc(u_bc, dofs)
                # self.bcs.append(bc)

            if 'OutletChoked' in boundary_conditions[i]:
                print("OutletChoked BC is working")
                # integral_C_o = 1j*(gamma-1)*M_o*c/2 * inner(self.u, self.v) * self.ds(i)
                integral_C_o = 1j*(gamma)*340/2 * inner(self.u, self.v) * self.ds(i)
                self.integrals_R.append(integral_C_o) 

        self._A = None
        self._B = None
        self._B_adj = None
        self._C = None

        info("- Passive matrices are assembling..")

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

        a = form(-self.c**2 * inner(grad(self.u), grad(self.v))*self.dx)
        A = assemble_matrix(a, bcs=self.bcs)
        A.assemble()
        info("- Matrix A is assembled.")
        self._A = A

    def assemble_B(self):

        N = self.V.dofmap.index_map.size_global
        n = self.V.dofmap.index_map.size_local

        if self.integrals_R:
            B = assemble_matrix(form(sum(self.integrals_R)))
            B.assemble()
        else:
            B = PETSc.Mat().create()
            B.setSizes([(n, N), (n, N)])
            B.setFromOptions()
            B.setUp()
            B.assemble()
            info("! Note: It can be faster to use EPS solver.")

        B_adj = B.copy()
        B_adj.transpose()
        B_adj.conjugate()

        info("- Matrix B is assembled.")
        self._B = B
        self._B_adj = B_adj    

    def assemble_C(self):

        c = form(inner(self.u , self.v) * self.dx)
        C = assemble_matrix(c, self.bcs)
        C.assemble()
        info("- Matrix C is assembled.\n")
        self._C = C

