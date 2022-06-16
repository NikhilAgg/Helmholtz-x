from helmholtz_x.geometry_pkgx.geomdl_utils import ParametricCurveCylinderical
from helmholtz_x.geometry_pkgx.xdmf_utils import  XDMFReader
from mpi4py import MPI
import numpy as np

x_ref, radius, Length = 0 , 0.0235 , 1

CPT_number = 7 # Number of control points

ControlPoints = []

for z_coord in np.linspace(0, Length, CPT_number):
    ControlPoints.append([x_ref, radius, z_coord])

# For Curvature Test
ControlPoints[2][1] +=0.005 

Geometry = XDMFReader("rijke_passive")

BoundaryTag = 3

curve = ParametricCurveCylinderical(Geometry, BoundaryTag, ControlPoints)

from dolfinx.io import XDMFFile

with XDMFFile(MPI.COMM_WORLD, "V.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
    
    xdmf.write_mesh(Geometry.mesh)
    xdmf.write_function(curve.V[3])


print(curve.K)

with XDMFFile(MPI.COMM_WORLD, "K.xdmf", "w", encoding=XDMFFile.Encoding.HDF5 ) as xdmf:
    
    xdmf.write_mesh(Geometry.mesh)
    xdmf.write_function(curve.K)