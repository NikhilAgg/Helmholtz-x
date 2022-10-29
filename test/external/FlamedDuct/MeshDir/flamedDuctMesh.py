import gmsh
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

filename = 'FlamedDuct/tube'

def fltk_options():

    # Type of entity label (0: description,
    #                       1: elementary entity tag,
    #                       2: physical group tag)
    gmsh.option.setNumber("Geometry.LabelType", 2)

    gmsh.option.setNumber("Geometry.PointNumbers", 0)
    gmsh.option.setNumber("Geometry.LineNumbers", 0)
    gmsh.option.setNumber("Geometry.SurfaceNumbers", 2)
    gmsh.option.setNumber("Geometry.VolumeNumbers", 2)

    # Mesh coloring(0: by element type, 1: by elementary entity,
    #                                   2: by physical group,
    #                                   3: by mesh partition)
    gmsh.option.setNumber("Mesh.ColorCarousel", 0)

    gmsh.option.setNumber("Mesh.Lines", 0)
    gmsh.option.setNumber("Mesh.SurfaceEdges", 0)
    gmsh.option.setNumber("Mesh.SurfaceFaces", 0) # CHANGE THIS FLAG TO 0 TO SEE LABELS

    gmsh.option.setNumber("Mesh.VolumeEdges", 2)
    gmsh.option.setNumber("Mesh.VolumeFaces", 2)

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 1)

gmsh.model.add(filename)
gmsh.option.setString('Geometry.OCCTargetUnit', 'M')

path = os.path.dirname(os.path.abspath(__file__))

gmsh.model.occ.importShapes(os.path.join(path, filename+'.step'))
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

gmsh.option.setNumber("Mesh.MeshSizeMax", 0.015)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.Algorithm3D", 10)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
gmsh.model.mesh.generate(3)

sur_tags=gmsh.model.getEntities(dim=2)

vol_tags=gmsh.model.getEntities(dim=3)


for i in sur_tags:
    gmsh.model.addPhysicalGroup(2, [i[1]], tag=i[1])

gmsh.model.addPhysicalGroup(3, [1], tag=99)
# gmsh.model.addPhysicalGroup(3, [2, 1002, 3, 1003] , tag=99)

gmsh.model.occ.synchronize()

gmsh.write("{}.msh".format(dir_path +"/"+filename))

if '-nopopup' not in sys.argv:
    fltk_options()
    gmsh.fltk.run()

gmsh.finalize()

from helmholtz_x.dolfinx_utils import  write_xdmf_mesh

write_xdmf_mesh(dir_path +"/"+filename,dimension=3)
