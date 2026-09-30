import pytest
import numpy as np
from mpi4py import MPI
import ufl
from petsc4py import PETSc
from dolfinx import mesh, fem

# Import from the src directory
from config.parameters import Parameters

@pytest.fixture(scope="module")
def dummy_parameters():
    """Provides a basic configuration for testing."""
    params = Parameters()
    params.cutFEM = 1
    params.p_const = 8
    params.elasticity_limit = 1.0
    params.young_modulus = 21000
    params.poisson = 0.3
    params.alpha_reg_velocity = 0.1
    params.ALM = 0
    params.ALM_lagrangian_multiplicator = 0.1
    params.ALM_penalty_parameter = 1.0
    params.ALM_slack_variable = 0.0
    params.target_constraint = 0.5
    params.cost_func = "Area"
    params.h = 0.2
    return params

@pytest.fixture(scope="module")
def test_mesh():
    """A simple 10x10 rectangular mesh from -1 to 1."""
    msh = mesh.create_rectangle(
        MPI.COMM_WORLD, 
        [np.array([-1.0, -1.0]), np.array([1.0, 1.0])], 
        [20, 20], 
        cell_type=mesh.CellType.triangle
    )
    return msh

@pytest.fixture(scope="module")
def test_spaces(test_mesh):
    """Dictionary of common function spaces."""
    dim = test_mesh.geometry.dim
    V = fem.functionspace(test_mesh, ("Lagrange", 1, (dim,)))
    V_ls = fem.functionspace(test_mesh, ("Lagrange", 1))
    V_DG = fem.functionspace(test_mesh, ("DG", 0, (dim,)))
    Q = fem.functionspace(test_mesh, ("DG", 0))
    return {"V": V, "V_ls": V_ls, "V_DG": V_DG, "Q": Q}

@pytest.fixture(scope="module")
def circle_level_set(test_spaces):
    """A level-set function representing a circle of radius 0.5."""
    V_ls = test_spaces["V_ls"]
    ls_func = fem.Function(V_ls)
    
    def circle_sdf(x):
        return np.sqrt(x[0]**2 + x[1]**2) - 0.5 + 1e-9
        
    ls_func.interpolate(circle_sdf)
    return ls_func

@pytest.fixture(scope="module")
def test_bcs(test_spaces):
    """Simple boundary conditions fixture (clamped on one side)."""
    V = test_spaces["V"]
    msh = V.mesh
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], -1.0))
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    u_bc = np.array([0, 0], dtype=PETSc.ScalarType)
    bc = fem.dirichletbc(u_bc, dofs, V)
    return [bc]

@pytest.fixture(scope="module")
def test_problem_topo():
    """Fixture for AreaProblem topology."""
    from config.problem import AreaProblem
    return AreaProblem()

@pytest.fixture(scope="module")
def test_measures(test_mesh):
    """Simple measure fixture."""
    ds = ufl.Measure("ds", domain=test_mesh)
    return ds

@pytest.fixture(scope="module")
def test_force(test_mesh):
    """A constant force acting downwards."""
    return fem.Constant(test_mesh, np.array([0.0, -1.0], dtype=np.float64))
