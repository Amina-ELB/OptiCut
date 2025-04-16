import ufl
from dolfinx import mesh
from petsc4py import PETSc
import numpy as np
import dolfinx.mesh
from mpi4py import MPI

def create_mesh_2D(lx, ly, Nx, Ny):
	msh = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0,0]), np.array([lx, ly])], [Nx,Ny], cell_type=mesh.CellType.triangle)
	return msh

def create_mesh_3D(lx, ly, lz, Nx, Ny, Nz):
    msh = mesh.create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([lx, ly,lz])], [Nx,Ny,Nz], cell_type=mesh.CellType.tetrahedron)
    return msh
