
from helmholtz_x.geometry_pkgx.xdmf_utils import load_xdmf_mesh, write_xdmf_mesh, XDMFReader
from dolfinx.fem import Function, FunctionSpace, dirichletbc, set_bc, locate_dofs_topological
from mpi4py import MPI
import dolfinx.io
import numpy as np

write_xdmf_mesh("MeshDir/Micca",dimension=3)


shape_derivatives={1: [(-106216.64819010241-28842.47439768992j), (-103946.74776043865-18915.913927299673j)],
                   2: [(-120427.8118610474-22496.42216972298j), (-115882.5094856025-7674.228319903667j)],
                   3: [(-63458.781851744425-33189.66853070488j), (-63112.45562449481-26373.75179107947j)],
                   4: [(-73717.70010223417-131939.37288347658j), (-49697.42069054075-63019.47856905913j)],
                #    5: [(684957.2778507026-579238.1325467043j), (532694.4248702549-1595269.9056810471j)],
                #    6: [(967578.1344932624+909531.8896318685j), (999481.6784963995+1340884.680977521j)],
                #    7: [(10869188.591875758+7106584.5287730675j), (10621772.732638739+8758862.148712022j)],
                   8: [(53168.429515806725-26932.53306653828j), (54254.63724106943-37712.01325530575j)],
                   9: [(1810.7615785676226-13717.787385280362j), (3043.4302117964553-17031.987110311344j)],
                   10: [(3039.992465215346-18417.460987942064j), (4626.13921984906-22611.29868677982j)],
                   11: [(-1736.0096429541263+9576.824700545447j), (-4317.264466591498+12156.929409672326j)]}

print(shape_derivatives[1])

micca = XDMFReader("MeshDir/Micca")

mesh, subdomains, facet_tags = micca.getAll()

V = FunctionSpace(mesh, ("CG",1))

fdim = mesh.topology.dim - 1

bcs = []
U = Function(V)
for i in shape_derivatives:
    print(i,shape_derivatives[i][0])           
    facets = np.array(facet_tags.indices[facet_tags.values == i])
    dofs = locate_dofs_topological(V, fdim, facets)
    U.x.array[dofs] = shape_derivatives[i][0] #first element of boundary

print(U.x.array)
    

# from helmholtz_x.geometry_pkgx.xdmf_utils import write_xdmf_mesh

# set_bc(U_R.vector, [bc])
# with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "Results/derivatives_real.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_function(u_real)


# bcs = []
   
# u_real = Function(V)
# facets = np.array(facet_tags.indices[facet_tags.values == 1])
# dofs = locate_dofs_topological(V, fdim, facets)
# u_real.x.array[dofs] = 2

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "Results/derivatives.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(U)














