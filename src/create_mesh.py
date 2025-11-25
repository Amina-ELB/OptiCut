import ufl
from dolfinx import mesh, io, fem
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

def load_mesh(parameters):
    test_case = "L_shape"
    parameters.test_case = test_case

    if test_case == "rectangle":
        msh = create_mesh_2D(parameters.lx, parameters.ly,
                             int(parameters.lx/parameters.h),
                             int(parameters.ly/parameters.h))
        ct = None

    elif test_case == "L_shape":
        with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
            msh, ct, _ = io.gmshio.read_from_msh(
                "mesh/rectangle.msh", MPI.COMM_WORLD, 0, gdim=2
            )
            xdmf.write_mesh(msh)

    elif test_case == "3D":
        msh = create_mesh_3D(...)
        ct = None

    else:
        raise RuntimeError("Unknown test case")

    msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)
    return msh, ct