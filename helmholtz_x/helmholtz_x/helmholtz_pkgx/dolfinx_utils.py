import dolfinx
import numpy as np

class BoundaryCondition():
    def __init__(self, type, marker, values):
        self._type = type
        if type == "Dirichlet":
            u_D = dolfinx.Function(V)
            u_D.interpolate(values)
            u_D.x.scatter_forward()
            facets = np.array(facet_tag.indices[facet_tag.values == marker])
            dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facets)
            self._bc = dolfinx.DirichletBC(u_D, dofs)
        elif type == "Neumann":
                pass
        elif type == "Robin":
            print(values)
            self._integral_R = 1j * values * c * inner(u,v) * ds(marker)
        else:
            raise TypeError("Unknown boundary condition: {0:s}".format(type))
    @property
    def bc(self):
        return self._bc
    @property
    def integral_R(self):
        return self._integral_R

    @property
    def type(self):
        return self._type