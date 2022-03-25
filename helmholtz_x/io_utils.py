
import json
import ast
from mpi4py import MPI

def dict_writer(filename, dictionary, extension = ".txt"):
    with open(filename+extension, 'w') as file:
        file.write(json.dumps(str(dictionary))) 
    if MPI.COMM_WORLD==0:
        print(filename, "is saved.")

def dict_loader(filename, extension = ".txt"):
    with open(filename+extension) as f:
        data = json.load(f)
    data = ast.literal_eval(data)
    if MPI.COMM_WORLD==0:
        print(filename, "is loaded.")
    return data

""" 

from helmholtz_x.io_utils import dict_writer,dict_loader

filename = "shape_derivatives"
dict_writer(filename,shape_derivatives)
data = dict_loader(filename)

"""