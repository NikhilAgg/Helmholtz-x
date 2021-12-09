from mpi4py import MPI

rank = MPI.COMM_WORLD.rank 

import meshio
import dolfinx.io

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
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, tag_loader_name, "r") as xdmf:
        facet_tags = xdmf.read_meshtags(mesh, name="Grid")
    print("XDMF Mesh is loaded.")
    return mesh, cell_tags, facet_tags