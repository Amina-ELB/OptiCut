# Copyright (c) 2025 ONERA and MINES Paris, France 
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari 

# The modules that will be used are imported:
import numpy as np

import ufl
 
# mathematical language for FEM, auto differentiation, python
from dolfinx import fem, io, mesh
from dolfinx.cpp.mesh import h as mesh_size
import matplotlib.pyplot as plt
# meshes, assembly, c++, ython, pybind
from ufl import ds

from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
from typing import TYPE_CHECKING
import pyvista


from dolfinx.fem import Function
import dolfinx.fem.petsc 
from dolfinx.mesh import meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import *

from Parameters import *
from utils.mesh_utils import *
from ersatz_elastic_solver import *
from cutfem_elastic_solver import *
from levelSet_tool import *
from velocity_tools import *
from geometry_initialization import *
import almMethod 

import shutil
import os

import mechanics_tool
import data_manipulation
import opti_tool
import problem
import almMethod 
# Import the boundary conditions module
from boundary_conditions import initialize_boundary_conditions, initialize_shift


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


if rank == 0:
    shutil.rmtree('res')
    os.mkdir('res')
    folder_cost_func = open("res/cost_func.txt", "x")
    folder_cost_compliance = open("res/cost_compliance.txt", "x")
    folder_lagrangian = open("res/lagrangian.txt", "x")
    folder_constraint = open("res/constraint.txt", "x")
    folder_max_vm = open("res/max_vm.txt", "x")
    folder_volume = open("res/volume.txt", "x")
    folder_param_lagrangian = open("res/param_lagrangian.txt", "x")
    folder_param_hist_vm_1 = open("res/vm_1_hist.txt", "x")
    folder_param_hist_vm_final = open("res/vm_final_hist.txt", "x")


compliance = 1
vect_cost = []
vect_volume = []
vect_compliance = []
vect_constraint = []
vect_constraint = []
vect_target_constraint = []
vect_lagrangian = []
vect_max_vm = []

from mpi4py import MPI
from dolfinx import fem, mesh, io
import ufl
from utils.config_utils import load_parameters, init_output_folders
from function_spaces_utils import init_function_spaces
from utils.bc_utils import initialize_boundary_conditions, initialize_shift
from problem import Compliance_Problem, VMLp_Problem, AreaProblem
from ersatz_elastic_solver import *
from cutfem_elastic_solver import *
from levelSet_tool import *
import data_manipulation
import almMethod
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ----------------------------
# Initialize output folders
# ----------------------------
init_output_folders(rank)

# ----------------------------
# Load parameters
# ----------------------------
use_file = 1
param_file = "parameters/param_VonMises.txt"
parameters = load_parameters(use_file=use_file, filename=param_file)


# ----------------------------
# Mesh generation
# ----------------------------
# test_case = "L_shape"  # could be user input
# msh = load_mesh(test_case, parameters, mesh_folder="mesh")
test_case = "L_shape"  # Could be user input
msh = load_mesh(test_case, parameters, mesh_folder="mesh")
msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim-1)

# ----------------------------
# Initialize function spaces
# ----------------------------
spaces = init_function_spaces(msh)
V, V_vm, V_ls, Q, V_DG = spaces["V"], spaces["V_vm"], spaces["V_ls"], spaces["Q"], spaces["V_DG"]

# ----------------------------
# Initialize level set
# ----------------------------
x = ufl.SpatialCoordinate(msh)
if test_case == "L_shape":
    ls_ufl = level_set_L_shape(x)
elif test_case == "rectangle":
    ls_ufl = level_set(x, parameters)
elif test_case == "3D":
    ls_ufl = level_set_3D(x, parameters)
else:
    raise ValueError("Test case not implemented")

ls_expr = fem.Expression(ls_ufl, V_ls.element.interpolation_points())
ls_func = fem.Function(V_ls)
ls_func.interpolate(ls_expr)

# ----------------------------
# Boundary conditions & shift
# ----------------------------
bcs, bc_velocity, ds = initialize_boundary_conditions(test_case, msh, V, V_ls, parameters)
shift = initialize_shift(test_case, msh, parameters)

# ----------------------------
# Select problem type
# ----------------------------
if parameters.cost_func == "compliance":
    problem_topo = Compliance_Problem()
elif parameters.cost_func == "VonMises":
    problem_topo = VMLp_Problem()
elif parameters.cost_func == "Area":
    problem_topo = AreaProblem()
else:
    raise ValueError("Problem type not implemented")

# ----------------------------
# Initialize solvers
# ----------------------------
AdvectionSolver = Advection(ls_func, V_ls=V_ls, dt=parameters.dt)
ReinitSolver = Reinitialization(ls_func, V_ls=V_ls, l=parameters.l_reinit)
ErsatzSolver = ErsatzElasticSolver(ls_func, V_ls, V, ds=ds, bc=bcs, bc_velocity=bc_velocity,
                                   parameters=parameters, shift=shift)
CutFemSolver = CutFEMElasticSolver(ls_func, V_ls, V, ds=ds, bc=bcs, bc_velocity=bc_velocity,
                                   parameters=parameters, problem_topo=problem_topo, shift=shift)

# ----------------------------
# Reinitialization
# ----------------------------
ls_func = ReinitSolver.reinitializationPC(ls_func, parameters.step_reinit)

# ----------------------------
# Solve primal & dual
# ----------------------------
uh, ph = None, None
if parameters.cutFEM == 1:
    uh, ph = CutFemSolver.cutfem_solver(ls_func, parameters, problem_topo)
else:
    uh, ph = ErsatzSolver.ersatz_solver(ls_func, parameters)

# ----------------------------
# Compute initial quantities
# ----------------------------
lame_mu, lame_lambda = mechanics_tool.lame_compute(parameters.young_modulus, parameters.poisson)
measure = CutFemSolver.dxq if parameters.cutFEM == 1 else ufl.dx

cost = problem_topo.cost(uh, ph, lame_mu, lame_lambda, measure, parameters)
shape_derivative = problem_topo.shape_derivative_integrand(uh, ph, lame_mu, lame_lambda, parameters, measure)
vm_list = data_manipulation.create_list_vm(msh, uh, parameters, lame_mu, lame_lambda, 0, ls_func, V_ls, Q, 0)

# ----------------------------
# Save initial results
# ----------------------------
time = 0.0
xdmf_file = io.XDMFFile(msh.comm, "res/level_set.xdmf", "w")
xdmf_file.write_mesh(msh)
for f, name in zip([ls_func, uh, ph, vm_list], ["ls_func", "disp", "dual", "vm_list"]):
    f.name = name
    xdmf_file.write_function(f, time)


# end of the verified part. 


# ---------- Temporary level-set used during line-search / advection ----------
ls_func_temp = fem.Function(V_ls)
ls_func_temp.x.array[:] = CutFemSolver.level_set.x.array
ls_func_temp.x.scatter_forward()

crit_0 = 1e10
crit = [1e+3, 1e+6, 1e+6, 1e+6]  # stagnation criteria history
lagrangian = [1e+3, 1e+6, 1e+6, 1e+6]
cv = 0  # 0 if convergence reached, 1 otherwise


print(style.RED + '##########################################')
print(style.RED + '##### Initialization of the problem  #####')
print(style.RED + '##########################################')
print(style.WHITE + " ")

# temporary placeholder
xsi_temp = fem.Function(V_ls)

# ---------- Initial vm_list and print ----------
k = 1.0
vm_list = data_manipulation.create_list_vm(msh, uh, parameters, lame_mu, lame_lambda, 0,
                                           CutFemSolver.level_set if parameters.cutFEM == 1 else ls_func_temp,
                                           V_ls, Q, 0)
max_vm = k * np.max(vm_list.x.array[:])

# ---------- prepare measures and initial quantities ----------
if parameters.cutFEM == 1:
    measure = CutFemSolver.dxq
    previous_cost = problem_topo.cost(uh, ph, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, measure, parameters)
    # already computed shape_derivative above
    previous_constraint = problem_topo.constraint(uh, lame_mu, lame_lambda, parameters, measure, 0, vm_list)
    almMethod.maj_param_constraint_optim(parameters, previous_constraint)
    dual_operator = problem_topo.dual_operator(uh, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, parameters, msh, measure, vm_list)
    shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(uh,ph,lame_mu,lame_lambda,parameters,ufl.dx)

else:
    measure = ufl.dx
    previous_cost = problem_topo.cost(uh, ph, ErsatzSolver.lame_mu_fic, ErsatzSolver.lame_lambda_fic, measure, parameters)
    previous_constraint = problem_topo.constraint(uh, ErsatzSolver.lame_mu_fic, ErsatzSolver.lame_lambda_fic, parameters, measure, ErsatzSolver.xsi)
    almMethod.maj_param_constraint_optim(parameters, previous_constraint)
    dual_operator = problem_topo.dual_operator(uh, ErsatzSolver.lame_mu_fic, ErsatzSolver.lame_lambda_fic, parameters, msh, measure)
    shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(uh,ph,lame_mu,lame_lambda,parameters,ufl.dx)

# init counters, parameters
lagrangian_cost_previous = 1e3
lagrangian_cost = 1e3
adv_bool = 1
c_param_HJ = 0.5
n_k = 1
c_k = 1
velocity_field = Function(V_ls)
velocity_field.x.array[:] = CutFemSolver.level_set.x.array*0

almMethod.init_param_constraint_optim(previous_constraint, parameters, cost)
resources = prepare_descent(msh, V_ls, parameters)

i = 0
# main optimization loop (stop when max iterations or criteria satisfied)
while (i < parameters.max_incr) and (
    (abs(crit[0]) > parameters.tol_cost_func) or
    (abs(crit[1]) > parameters.tol_cost_func) or
    (abs(crit[2]) > parameters.tol_cost_func) or
    (abs(crit[3]) > parameters.tol_cost_func)
):
    # adapt HJ regularization parameter
    c_param_HJ = opti_tool.adapt_c_HJ(c_param_HJ, crit, parameters.tol_cost_func, lagrangian)
    print("c = ", c_param_HJ)
    cv = 0

    print(style.BLUE + "iteration number : ", i)
    print(style.WHITE + "")

    # update ALM parameters
    almMethod.maj_param_constraint_optim(parameters, previous_constraint)

    # ---------- Descent direction (solve regularized subproblem) ----------
    v_reg = descent_direction(CutFemSolver.level_set, msh, parameters, bc_velocity, V_ls,
                              previous_constraint, shape_derivative_integrand_constraint, shape_derivative,
                              resources)
    # v_reg is expected to be a fem.Function in V_ls
    if isinstance(v_reg, fem.Function):
        velocity_field.x.array[:] = v_reg.x.array
    else:
        # v_reg might be an UFL expression or plain numpy array: try interpolation
        try:
            vel_expr = fem.Expression(v_reg, V_ls.element.interpolation_points())
            velocity_field.interpolate(vel_expr)
        except Exception:
            # as fallback set to zero
            velocity_field.x.array[:] = 0.0
    velocity_field.x.scatter_forward()

    # Normalize velocity (expects ufl expression or fem.Function)
    velocity_field = velocity_normalization(velocity_field, parameters.alpha_reg_velocity)
    # ensure velocity_field is a fem.Function: if velocity_normalization returns a ufl expr, interpolate it:
    if not isinstance(velocity_field, fem.Function):
        try:
            vel_expr = fem.Expression(velocity_field, V_ls.element.interpolation_points())
            vf = fem.Function(V_ls)
            vf.interpolate(vel_expr)
            velocity_field = vf
        except Exception:
            # keep existing velocity_field
            pass

    # compute max velocity across ranks
    max_velocity_local = np.max(np.abs(velocity_field.x.array[:]))
    max_velocity = comm.allreduce(max_velocity_local, op=MPI.MAX)

    # ---------- Advection: Hamilton-Jacobi (level-set) ----------
    solve = fem.Function(V_ls)
    solve.x.array[:] = CutFemSolver.level_set.x.array
    solve.x.scatter_forward()

    solve_temp = fem.Function(V_ls)
    solve_temp.x.array[:] = solve.x.array
    solve_temp.x.scatter_forward()

    adv_inner_loop = True
    while adv_inner_loop:
        j = 0
        cv = 0
        ls_func_temp.x.array[:] = solve.x.array
        CutFemSolver.level_set.x.array[:] = solve.x.array
        CutFemSolver.level_set.x.scatter_forward()
        
        # iterate HJ solver for j_max steps (or early stopping)
        while j < parameters.j_max:
            AdvectionSolver.set_level_set(ls_func_temp)
            # cut_fem_adv should return a fem.Function (new level-set)
            ls_new = AdvectionSolver.cut_fem_adv(velocity_field, (1.0 / adv_bool) * parameters.dt)
            # replace ls_func_temp with the new function values
            ls_func_temp.x.array[:] = ls_new.x.array
            ls_func_temp.x.scatter_forward()
            j += 1

            # periodic reinitialization
            if (j % parameters.freq_reinit) == 0:
                ls_func_temp = ReinitSolver.reinitializationPC(ls_func_temp, parameters.step_reinit)
                ls_func_temp.x.scatter_forward()

        # ---------- Recompute primal/adjoint for the advected domain ----------
        # optionally adapt time stepping logic controlled by parameters.adapt_time_step
        # here we recompute cost/adjoint until cv flag set
        while ((parameters.adapt_time_step + 1) * cv) == 0:
            if parameters.cutFEM == 1:
                # primal solve on updated level-set
                uh = CutFemSolver.primal_problem(ls_func_temp)
                CutFemSolver.update_measures_and_quadratures(ls_func_temp)
                # compute cost and shape derivatives using updated measure
                cost = problem_topo.cost(uh, ph, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, CutFemSolver.dxq, parameters)
                shape_derivative = problem_topo.shape_derivative_integrand(uh, ph, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, parameters, CutFemSolver.dxq)
                vm_list = data_manipulation.create_list_vm(msh, uh, parameters, lame_mu, lame_lambda, 0, CutFemSolver.level_set, V_ls, Q, 0)
                max_vm = np.max(vm_list.x.array[:])

                constraint = problem_topo.constraint(uh, lame_mu, lame_lambda, parameters, CutFemSolver.dxq, 0, vm_list, c_k)
                almMethod.maj_param_constraint_optim_slack(parameters, constraint)

                if parameters.cost_func != "compliance":
                    dual_operator = problem_topo.dual_operator(uh, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, parameters, msh, CutFemSolver.dxq, vm_list, c_k)
                    CutFemSolver.update_measures_and_quadratures(ls_func_temp)
                    ph = CutFemSolver.adjoint_problem(uh, ls_func_temp, dual_operator)

                shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(
                    uh, ph, lame_mu, lame_lambda, parameters, CutFemSolver.dxq, vm_list, c_k
                )
            else:
                xsi_temp = ErsatzSolver.heaviside(ls_func_temp)
                uh, ph = ErsatzSolver.ersatz_solver(ls_func_temp, parameters)
                measure = ufl.dx
                CutFemSolver.update_measures_and_quadratures(ls_func_temp)
                cost = problem_topo.cost(uh, ph, ErsatzSolver.lame_mu_fic, ErsatzSolver.lame_lambda_fic, measure, parameters)
                shape_derivative = problem_topo.shape_derivative_integrand(uh, ph, ErsatzSolver.lame_mu_fic, ErsatzSolver.lame_lambda_fic, parameters, measure)
                constraint = problem_topo.constraint(uh, ErsatzSolver.lame_mu_fic, ErsatzSolver.lame_lambda_fic, parameters, measure, ErsatzSolver.xsi)
                almMethod.maj_param_constraint_optim_slack(parameters, constraint)
                dual_operator = problem_topo.dual_operator(uh, ErsatzSolver.lame_mu_fic, ErsatzSolver.lame_lambda_fic, parameters, msh, measure)

            # compute Lagrangian cost and decide cv
            lagrangian_cost = opti_tool.lagrangian_cost(cost, constraint, parameters)
            print(style.YELLOW + "cost previous = ", previous_cost)
            print(style.YELLOW + "cost = ", cost)
            print(style.WHITE + "C(Ω) = ", float(constraint))

            if cost < (previous_cost * (1.0 + parameters.tol_cost_func)) or (parameters.adapt_time_step == 0):
                cv = 1
            else:
                cv = 0
        if i==1:
            constraint_derivative = abs(constraint - previous_constraint)/(parameters.dt *parameters.j_max)
            cost_derivative = abs(cost - previous_cost) / (parameters.dt *parameters.j_max)
            almMethod.init_param_constraint_optim(constraint_derivative,parameters,cost_derivative)
        # catch NaN / adapt j_max
        parameters.dt, adv_bool = opti_tool.catch_NAN(cost, lagrangian_cost, constraint, parameters.dt, adv_bool)
        if adv_bool < 2:
            parameters.j_max = opti_tool.adapt_HJ(lagrangian_cost, lagrangian_cost_previous, parameters.j_max, parameters.dt, parameters)
        else:
            parameters.j_max = 1

        # stop the advection inner loop
        adv_inner_loop = False

    # end adv loop

    # update iteration history & convergence criteria
    crit[3], crit[2], crit[1], crit[0] = crit[2], crit[1], crit[0], abs(cost - previous_cost) / (previous_cost + 1e-30)
    print("criterion of convergence = ", crit[0])

    # accept new level-set
    CutFemSolver.level_set.x.array[:] = ls_func_temp.x.array
    CutFemSolver.level_set.x.scatter_forward()

    ErsatzSolver.xsi = xsi_temp

    lagrangian_cost_previous = lagrangian_cost

    # gather results from all ranks
    collected_cost = comm.allreduce(cost, op=MPI.SUM)
    vect_cost.append(collected_cost)
    collected_constraint = comm.allreduce(constraint, op=MPI.SUM)
    vect_constraint.append(collected_constraint)
    vect_target_constraint.append(parameters.target_constraint)
    collected_lagrangian_cost_previous = comm.allreduce(lagrangian_cost_previous, op=MPI.SUM)
    previous_cost = cost
    previous_constraint = constraint

    vm_list = data_manipulation.create_list_vm(msh, uh, parameters, lame_mu, lame_lambda, 0, CutFemSolver.level_set, V_ls, Q, 0)


    # write results on rank 0 (prefer using output_utils.write_to_file)
    if rank == 0:
        try:
            folder_cost_func.write("\n" + str(collected_cost))
            folder_constraint.write("\n" + str(collected_constraint))
            folder_param_lagrangian.write("\n" + str(collected_lagrangian_cost_previous))
            folder_max_vm.write("\n" + str(max_vm))
        except Exception:
            # fallback: append to files
            with open("res/cost_func.txt", "a") as f:
                f.write("\n" + str(collected_cost))
            with open("res/constraint.txt", "a") as f:
                f.write("\n" + str(collected_constraint))

    time += 1.0
    for f, name in zip([ls_func, uh, ph, vm_list], ["ls_func", "disp", "dual", "vm_list"]):
        f.name = name
        xdmf_file.write_function(f, time)
    # increment time and iteration counter
    i += 1

# end main optimization loop

