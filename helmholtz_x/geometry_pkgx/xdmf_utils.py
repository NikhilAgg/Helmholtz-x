from mpi4py import MPI
import meshio
import dolfinx.io
from dolfinx.fem import Function, FunctionSpace, locate_dofs_topological
from numpy import array

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def create_mesh(mesh, cell_type, prune_z):
    
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

def write_xdmf_mesh(name, dimension):

    if rank == 0:
        # Read in mesh
        msh_name = name + ".msh"
        msh = meshio.read(msh_name)

        if dimension == 2:
            prune_z = True
            volume_mesh = create_mesh(msh, "triangle",prune_z)
            tag_mesh = create_mesh(msh, "line",prune_z)

        elif dimension == 3:
            prune_z = False
            volume_mesh = create_mesh(msh, "tetra",prune_z)
            tag_mesh = create_mesh(msh, "triangle",prune_z)
            
        # Create and save one file for the mesh, and one file for the facets 
        
        xdmf_name = name + ".xdmf"
        xdmf_tags_name = name + "_tags.xdmf"
        meshio.write(xdmf_name, volume_mesh)
        meshio.write(xdmf_tags_name, tag_mesh)
    print(str(dimension)+"D XDMF mesh is generated.")

def load_xdmf_mesh(name):
    mesh_loader_name = name + ".xdmf"
    tag_loader_name = name + "_tags.xdmf"
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_loader_name, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tag_loader_name, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")
    if MPI.COMM_WORLD.rank == 0:
        print("XDMF Mesh is loaded.")
    return mesh, cell_tags, facet_tags

class XDMFReader:
    def __init__(self, name):
        self.name = name
        self._mesh = None
        self._cell_tags = None
        self._facet_tags = None
        self._gdim = None
        mesh_loader_name = name + ".xdmf"
        tag_loader_name = name + "_tags.xdmf"
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_loader_name, "r") as xdmf:
            self._mesh = xdmf.read_mesh(name="Grid")
            self._cell_tags = xdmf.read_meshtags(self.mesh, name="Grid")
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim-1)
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tag_loader_name, "r") as xdmf:
            self._facet_tags = xdmf.read_meshtags(self.mesh, name="Grid")
        if MPI.COMM_WORLD.rank == 0:
            print("XDMF Mesh is loaded.")

    @property
    def mesh(self):
        return self._mesh
    @property
    def subdomains(self):
        return self._cell_tags
    @property
    def facet_tags(self):
        return self._facet_tags   
    @property
    def dimension(self):
        return self._mesh.topology.dim    

    def getAll(self):
        return self.mesh, self.subdomains, self.facet_tags
    
    def getNumberofCells(self):
        t_imap = self.mesh.topology.index_map(self.mesh.topology.dim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        total_num_cells = comm.allreduce(num_cells, op=MPI.SUM) #sum all cells and distribute to each process
        if rank==0:
            print("Number of cells: {:,}".format(total_num_cells))
            print("Number of cores: ", size)
        return total_num_cells
        


def derivatives_visualizer(filename, shape_derivatives, geometry, normalize=True):
    """ Filename should specify path excluding extension (don't write .xdmf)
        Geometry should be object that is built by XDMF Reader class

    Args:
        filename (str): file name (or path )
        shape_derivatives (dict): Should have shape derivatives as a dictionary
        geometry (XDMFReader): geometry object
    """
    shape_derivatives_real = shape_derivatives.copy()
    shape_derivatives_imag = shape_derivatives.copy()


    for key, value in shape_derivatives.items():
        shape_derivatives_real[key] = value[0].real
        shape_derivatives_imag[key] = value[0].imag 
        shape_derivatives[key] = value[0]  # get the first eigenvalue of each list

    if normalize:
        

        max_key_real = max(shape_derivatives_real, key=lambda y: abs(shape_derivatives_real[y]))
        max_value_real = abs(shape_derivatives_real[max_key_real])
        max_key_imag = max(shape_derivatives_imag, key=lambda y: abs(shape_derivatives_imag[y]))
        max_value_imag = abs(shape_derivatives_imag[max_key_imag])

        normalized_derivatives = shape_derivatives.copy()

        for key, value in shape_derivatives.items():
            normalized_derivatives[key] =  value.real/max_value_real + 1j*value.imag/max_value_imag

        shape_derivatives = normalized_derivatives

    V = FunctionSpace(geometry.mesh, ("CG",1))
    fdim = geometry.mesh.topology.dim - 1
    U = Function(V)

    # print(shape_derivatives)
    for tag, derivative in shape_derivatives.items():
        print(tag, derivative)           
        facets = array(geometry.facet_tags.indices[geometry.facet_tags.values == tag])
        dofs = locate_dofs_topological(V, fdim, facets)
        U.x.array[dofs] = derivative #first element of boundary

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename+".xdmf", "w") as xdmf:
        xdmf.write_mesh(geometry.mesh)
        xdmf.write_function(U)
