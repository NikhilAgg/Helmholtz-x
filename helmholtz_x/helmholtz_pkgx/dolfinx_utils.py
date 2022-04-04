import numpy as np
from scipy import interpolate
from dolfinx.fem import Function, FunctionSpace

def interpolator(xs,ys,data,mesh):
    """Interpolates matrix data onto dolfinx grid

    Args:
        xs ([numpy array]): [x coordinates of matrix data]
        ys ([numpy array]): [y coordinates of matrix data]
        data ([numpy array]): [square numpy matrix]
        mesh ([dolfinx.mesh.Mesh]): [dolfinx mesh]

    Returns:
        [type]: [description]
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
    print("Interpolation is done.")
    return dolf_function