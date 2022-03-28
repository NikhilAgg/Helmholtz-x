
import json
import ast
from mpi4py import MPI

def dict_writer(filename, dictionary, extension = ".txt"):
    with open(filename+extension, 'w') as file:
        file.write(json.dumps(str(dictionary))) 
    if MPI.COMM_WORLD.rank==0:
        print(filename+extension, " is saved.")

def dict_loader(filename, extension = ".txt"):
    with open(filename+extension) as f:
        data = json.load(f)
    data = ast.literal_eval(data)
    if MPI.COMM_WORLD.rank==0:
        print(filename+extension, " is loaded.")
    return data

""" 

from helmholtz_x.io_utils import dict_writer,dict_loader

filename = "shape_derivatives"
dict_writer(filename,shape_derivatives)
data = dict_loader(filename)

"""

# H5 FILE READER FROM JACK HALE!

import numpy as np
import pytest

from petsc4py import PETSc
import h5py

import dolfinx
import dolfinx.io

from dolfinx.geometry import BoundingBoxTree, compute_collisions_point, select_colliding_cells

from ufl import dx


def interpolate_non_matching_meshes(f1, f2, padding=1E-12):
    V1 = f1.function_space
    V2 = f2.function_space

    mesh1 = V1.mesh
    mesh2 = V2.mesh
    assert(mesh1.topology.dim == mesh2.topology.dim)
    assert(mesh1.mpi_comm().size == mesh2.mpi_comm().size == 1)

    tree = BoundingBoxTree(mesh1, mesh1.topology.dim, padding=padding)

    dof_coordinates = V2.tabulate_dof_coordinates()
    for i, coord in enumerate(dof_coordinates):
        cell_candidates = compute_collisions_point(tree, coord)
        cell = select_colliding_cells(mesh1, cell_candidates, coord, 1)
        assert(cell.shape[0] == 1)
        f1_eval = f1.eval(coord, cell)
        f2.vector[i] = f1_eval

    return f2


def write_function_as_vector(function, filename):
    comm = function.function_space.mesh.mpi_comm()
    vector = function.vector

    with h5py.File(filename, "w", driver="mpio", comm=comm) as f:
        global_size = vector.getSize()
        d = f.create_dataset("vector", (global_size,), dtype=np.float64)

        ownership = vector.getOwnershipRange()
        s = slice(ownership[0], ownership[1])

        d[s] = vector.array

        # Metadata for basic sanity checking on read.
        d.attrs["ufl_element"] = str(function.function_space.ufl_element())
        d.attrs["ownership_ranges"] = vector.getOwnershipRanges()
        d.attrs["mpi_comm_size"] = comm.size


def read_function_from_vector(function, filename):
    comm = function.function_space.mesh.mpi_comm()
    vector = function.vector

    with h5py.File(filename, "r", driver="mpio", comm=comm) as f:
        d = f["vector"]
        # Check metadata matches.
        assert(d.attrs["ufl_element"] == str(
            function.function_space.ufl_element()))
        assert(np.array_equal(
            d.attrs["ownership_ranges"], vector.getOwnershipRanges()))
        assert(d.attrs["mpi_comm_size"] == comm.size)

        ownership = vector.getOwnershipRange()
        s = slice(ownership[0], ownership[1])
        vector.array = d[s]
        vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                           mode=PETSc.ScatterMode.FORWARD)


def test_basic():
    def f(x):
        return x[0]**2 + x[1]**2

    mesh = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, 16, 16)
    V = dolfinx.FunctionSpace(mesh, ("CG", 1))
    f_h = dolfinx.Function(V)
    f_h.interpolate(f)

    write_function_as_vector(f_h, "output.h5")

    f_h_in = dolfinx.Function(V)
    read_function_from_vector(f_h_in, "output.h5")
    assert(f_h_in.vector.equal(f_h.vector))

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "native_output.xdmf", "w") as of:
        of.write_mesh(mesh)
        of.write_function(f_h_in)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("n1,n2", [(16, 32), (4, 12), (24, 64), (3, 9), (12, 4), (32, 128), (7, 65)])
def test_interpolation_non_matching_meshes(n1, n2, order):
    mesh1 = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, n1, n1)
    mesh2 = dolfinx.UnitSquareMesh(MPI.COMM_WORLD, n2, n2)

    V1 = dolfinx.FunctionSpace(mesh1, ("CG", order))
    V2 = dolfinx.FunctionSpace(mesh2, ("CG", order))

    def f(x):
        return x[0]**2 + x[1]**2

    f_V1 = dolfinx.Function(V1)
    f_V1.interpolate(f)

    # Interpolate forward
    f_V2 = dolfinx.Function(V2)
    interpolate_non_matching_meshes(f_V1, f_V2)

    # Interpolate back
    f_V2_V1 = dolfinx.Function(V1)
    interpolate_non_matching_meshes(f_V2, f_V2_V1)

    if (n1 % n2 == 0 and n2 > n1) or order == 2:
        s1 = dolfinx.fem.assemble_scalar(f_V1*dx)
        s2 = dolfinx.fem.assemble_scalar(f_V2*dx)
        assert(np.allclose(s1, s2))
        assert(np.allclose(f_V1.vector.norm(), f_V2_V1.vector.norm()))


if __name__ == "__main__":
    test_basic()