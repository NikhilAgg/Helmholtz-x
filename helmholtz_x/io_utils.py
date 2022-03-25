
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

