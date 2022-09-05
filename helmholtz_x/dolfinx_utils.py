from dolfinx.fem import Function, FunctionSpace, form, locate_dofs_topological
from dolfinx.fem.assemble import assemble_scalar
from dolfinx.mesh import meshtags
from dolfinx.io import XDMFFile, VTXWriter
from mpi4py import MPI
from scipy import interpolate
import ufl
import numpy as np
import meshio
import json
import ast

def info(str):
    """Only prints information message for once. Useful for logging,

    Args:
        str ('str'): log entry
    """
    if MPI.COMM_WORLD.Get_rank()==0:
        print(str)

def dict_writer(filename, dictionary, extension = ".txt"):
    """Writes dictionary object into a text file.

    Args:
        filename ('str'): file path
        dictionary ('dict'): dictionary object
        extension (str, optional): file extension. Defaults to ".txt".
    """
    with open(filename+extension, 'w') as file:
        file.write(json.dumps(str(dictionary))) 
    if MPI.COMM_WORLD.rank==0:
        print(filename+extension, " is saved.")

def dict_loader(filename, extension = ".txt"):
    """Loads dictionary into python script

    Args:
        filename ('str'): file path
        extension (str, optional): file extension. Defaults to ".txt".

    Returns:
        dictionary ('dict'): dictionary object
    """
    with open(filename+extension) as f:
        data = json.load(f)
    data = ast.literal_eval(data)
    if MPI.COMM_WORLD.rank==0:
        print(filename+extension, " is loaded.")
    return data

def cyl2cart(rho, phi, zeta):
    # cylindrical to Cartesian
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = zeta
    return x, y, z


def cart2cyl(x, y, z):
    # cylindrical to Cartesian
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    zeta = z
    return rho, phi, zeta

def interpolator(xs, ys, data, mesh):
    """Interpolates matrix data onto dolfinx grid

    Args:
        xs ([numpy array]): [x coordinates of matrix data]
        ys ([numpy array]): [y coordinates of matrix data]
        data ([numpy array]): [square numpy matrix]
        mesh ([dolfinx.mesh.Mesh]): [dolfinx mesh]

    Returns:
        (dolfinx.fem.function.Function): Interpolated dolfinx Function
    """
    f = interpolate.interp2d(xs, ys, data, kind='linear')
    V = FunctionSpace(mesh, ('CG',1))
    mesh_coord_V = V.tabulate_dof_coordinates()
    xs_dolf = mesh_coord_V[:,0]
    ys_dolf = mesh_coord_V[:,1]
    new_data = np.zeros(len(xs_dolf))
    for ii in range(len(xs_dolf)):
        new_data[ii] = f(xs_dolf[ii],ys_dolf[ii])
    dolf_function = Function(V)
    dolf_function.x.array[:] = new_data
    dolf_function.x.scatter_forward()
    info("Interpolation is done.")
    return dolf_function

def normalize(func):
    """Normalizes dolfinx function such that it integrates to 1 over the domain.

    Args:
        func (dolfinx.fem.function.Function): Dolfinx Function

    Returns:
        dolfinx.fem.function.Function: Normalized dolfinx function
    """

    integral_form = form(func*ufl.dx)
    integral= MPI.COMM_WORLD.allreduce(assemble_scalar(integral_form), op=MPI.SUM)

    func.x.array[:] /= integral
    func.x.scatter_forward()

    return func

def xdmf_writer(name, mesh, function):
    """ writes functions into xdmf file

    Args:
        name (string): name of the file
        mesh (dolfinx.mesh.Mesh]): Dolfinx mesh
        function (dolfinx.fem.function.Function): Dolfinx function to be saved.
    """
    with XDMFFile(MPI.COMM_WORLD, name+".xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(function)

def vtx_writer(name, mesh, function):
    """ writes functions into xdmf file

    Args:
        name (string): name of the file
        mesh (dolfinx.mesh.Mesh]): Dolfinx mesh
        function (dolfinx.fem.function.Function): Dolfinx function to be saved.
    """
    with VTXWriter(mesh.comm, name+".bp", function) as vtx:
        vtx.write(0.0)
        

def create_mesh(mesh, cell_type, prune_z):
    """Subroutine for mesh creation by using meshio library

    Args:
        mesh (meshio._mesh.Mesh): mesh to be converted into Dolfinx mesh
        cell_type ('str'): type of cell (it becomes tetrahedral most of the time)
        prune_z ('bool'): whether consider the 3th dimension's coordinate or not, (it should be False for 2D cases)

    Returns:
        meshio._mesh.Mesh: converted dolfinx mesh
    """

    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"name_to_read":[cell_data]})
    return out_mesh

def write_xdmf_mesh(name, dimension):
    """Writes gmsh (.msh) mesh as an .xdmf mesh

    Args:
        name ('str'): filename
        dimension ('int'): Dimension of the mesh (2 or 3)
    """

    if MPI.COMM_WORLD.Get_rank() == 0:
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
    """Loads xdmf mesh into python script

    Args:
        name ('str'): Name of the .xdmf file

    Returns:
        tuple: mesh, boundary tags and volume tags of the geometry
    """
    mesh_loader_name = name + ".xdmf"
    tag_loader_name = name + "_tags.xdmf"
    with XDMFFile(MPI.COMM_WORLD, mesh_loader_name, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
        cell_tags = xdmf.read_meshtags(mesh, name="Grid")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    with XDMFFile(MPI.COMM_WORLD, tag_loader_name, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")
    if MPI.COMM_WORLD.rank == 0:
        print("XDMF Mesh is loaded.")
    return mesh, cell_tags, facet_tags

class XDMFReader:
    """This class generates geometry objec to load its instances and information about it (number of elements etc.)
    """
    def __init__(self, name):
        self.name = name
        self._mesh = None
        self._cell_tags = None
        self._facet_tags = None
        self._gdim = None
        mesh_loader_name = name + ".xdmf"
        tag_loader_name = name + "_tags.xdmf"
        with XDMFFile(MPI.COMM_WORLD, mesh_loader_name, "r") as xdmf:
            self._mesh = xdmf.read_mesh(name="Grid")
            self._cell_tags = xdmf.read_meshtags(self.mesh, name="Grid")
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, self.mesh.topology.dim-1)
        with XDMFFile(MPI.COMM_WORLD, tag_loader_name, "r") as xdmf:
            self._facet_tags = xdmf.read_meshtags(self.mesh, name="Grid")    
        info("XDMF Mesh is loaded.")
    
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
    
    def getInfo(self):
        t_imap = self.mesh.topology.index_map(self.mesh.topology.dim)
        num_cells = t_imap.size_local + t_imap.num_ghosts
        total_num_cells = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM) #sum all cells and distribute to each process
        if MPI.COMM_WORLD.Get_rank()==0:
            print("Number of cells:  {:,}".format(total_num_cells))
            print("Number of cores: ", MPI.COMM_WORLD.Get_size(), "\n")
        return total_num_cells

def derivatives_visualizer(filename, shape_derivatives, geometry, normalize=True):
    """ This function generates a ,xdmf file which can visualize shape derivative values on boundaries of the geometry.
        Normalized shape derivatives can be done if wanted.
        Filename should specify path excluding extension (don't write .xdmf)
        Geometry should be object that is built by XDMF Reader class

    Args:
        filename (str): file name (or path )
        shape_derivatives (dict): Should have shape derivatives as a dictionary
        geometry (XDMFReader): geometry object
        normalize('bool'): Normalize or not (default is True)
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
        facets = np.array(geometry.facet_tags.indices[geometry.facet_tags.values == tag])
        dofs = locate_dofs_topological(V, fdim, facets)
        U.x.array[dofs] = derivative #first element of boundary
        U.x.scatter_forward()

    with XDMFFile(MPI.COMM_WORLD, filename+".xdmf", "w") as xdmf:
        xdmf.write_mesh(geometry.mesh)
        xdmf.write_function(U)

def ParallelMeshVisualizer(filename):
    """This function visualizes the paralel mesh partittion. 
       XDMFReaderT should be used in paraview to read ranktags.
    Args:
        filename (str): Path of the mesh file
    """
    Geometry = XDMFReader(filename)
    mesh, subdomains, facet_tags = Geometry.getAll()

    #mesh = create_unit_square(MPI.COMM_WORLD, 10, 10)

    num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
    mt = meshtags(mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32), np.full(num_cells_local, mesh.comm.rank, dtype=np.int32))

    with XDMFFile(mesh.comm, "Ranks.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(mt)