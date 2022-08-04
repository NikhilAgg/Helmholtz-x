from dolfinx.fem import Function, FunctionSpace, form
from dolfinx.fem.assemble import assemble_scalar
from dolfinx.io import XDMFFile
from mpi4py import MPI
from scipy import interpolate
import ufl
import numpy as np


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
    print("Interpolation is done.")
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
        function (dolfinx.fem.function.Function): _description_
    """
    with XDMFFile(MPI.COMM_WORLD, name+".xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_function(function)