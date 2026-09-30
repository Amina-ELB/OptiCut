# Copyright (c) 2025 ONERA and MINES Paris, France
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari

import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import gc
import time as time_module

start_time_total = time_module.time()


import ufl

from dolfinx import fem, io, mesh
from dolfinx.cpp.mesh import h as mesh_size
import matplotlib.pyplot as plt
from ufl import ds

from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
from typing import TYPE_CHECKING


from dolfinx.fem import Function
import dolfinx.fem.petsc
from dolfinx.mesh import meshtags
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from ufl import *

from config.parameters import *
from utils.ls_utils import *
from solvers.ersatz_elastic_solver import *
from solvers.cutfem_elastic_solver import *
from levelset.levelSet_tool import *
from levelset.velocity_tools import *
from optimization import almMethod
from levelset import geometry_initialization

import shutil
import os

from utils import mechanics_tool
from utils import data_manipulation
from optimization import opti_tool
from config import problem
from optimization import almMethod

# Import the boundary conditions module
from fem.boundary_conditions import initialize_boundary_conditions, initialize_shift

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_memory_usage():
    with open("/proc/self/status") as f:
        for line in f:
            if "VmRSS" in line:
                return int(line.split()[1])
    return 0


class style:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


# Limit history size to avoid memory accumulation
MAX_HISTORY = 1000


compliance = 1
vect_cost = []
vect_volume = []
vect_compliance = []
vect_constraint = []
vect_target_constraint = []
vect_lagrangian = []
vect_max_vm = []

from mpi4py import MPI
from dolfinx import fem, mesh, io
import ufl
from utils.config_utils import load_parameters, init_output_folders
from utils.spaces_utils import init_function_spaces
from config.problem import Compliance_Problem, VMLp_Problem, AreaProblem
from solvers.ersatz_elastic_solver import *
from solvers.cutfem_elastic_solver import *
from levelset.levelSet_tool import *
from utils import data_manipulation
from optimization import almMethod
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ----------------------------
# Initialize output folders
# ----------------------------
init_output_folders(rank)

if rank == 0:
    output_files = {
        "cost_func": open("res/cost_func.txt", "w"),
        "constraint": open("res/constraint.txt", "w"),
        "max_vm": open("res/max_vm.txt", "w"),
        "param_lagrangian": open("res/param_lagrangian.txt", "w"),
        "memory": open("res/memory.txt", "w"),
        "memory_per_process": open("res/memory_per_process.txt", "w"),
    }

# ----------------------------
# Load parameters
# ----------------------------
use_file = 1
param_file = sys.argv[1] if len(sys.argv) > 1 else "parameters/param_compliance.txt"
parameters = load_parameters(use_file=use_file, filename=param_file)


# ----------------------------
# Mesh generation
# ----------------------------
# test_case = "L_shape"  # could be user input
# msh = load_mesh(test_case, parameters, mesh_folder="mesh")
msh = load_mesh(parameters.mesh_case, parameters, mesh_folder="mesh")
msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim - 1)

# ----------------------------
# Initialize function spaces
# ----------------------------
spaces = init_function_spaces(msh)
V, V_vm, V_ls, Q, V_DG = (
    spaces["V"],
    spaces["V_vm"],
    spaces["V_ls"],
    spaces["Q"],
    spaces["V_DG"],
)


# ----------------------------
# Initialize level set
# ----------------------------
x = ufl.SpatialCoordinate(msh)
if hasattr(parameters, "level_set_init") and hasattr(
    geometry_initialization, parameters.level_set_init
):
    ls_init_func = getattr(geometry_initialization, parameters.level_set_init)
    try:
        ls_ufl = ls_init_func(x, parameters)
    except TypeError:
        ls_ufl = ls_init_func(x)
else:
    # Fallback to defaults
    if parameters.mesh_case == "L_shape":
        ls_ufl = level_set_L_shape(x)
    elif parameters.mesh_case == "rectangle":
        ls_ufl = level_set(x, parameters)
    elif parameters.mesh_case == "3D":
        ls_ufl = level_set_3D(x, parameters)
    else:
        raise ValueError("Test case not implemented")
pts = (
    V_ls.element.interpolation_points()
    if callable(getattr(V_ls.element, "interpolation_points", None))
    else V_ls.element.interpolation_points
)
ls_expr = fem.Expression(ls_ufl, pts)
ls_func = fem.Function(V_ls)
ls_func.interpolate(ls_expr)
ls_func.x.scatter_forward()

# ----------------------------
# Boundary conditions & shift
# ----------------------------
bcs, bc_velocity, ds = initialize_boundary_conditions(
    parameters.mesh_case, msh, V, V_ls, parameters
)
shift = initialize_shift(parameters.mesh_case, msh, parameters)

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
ErsatzSolver = ErsatzElasticSolver(
    ls_func,
    V_ls,
    V,
    ds=ds,
    bc=bcs,
    bc_velocity=bc_velocity,
    parameters=parameters,
    shift=shift,
)
CutFemSolver = CutFEMElasticSolver(
    ls_func,
    V_ls,
    V,
    ds=ds,
    bc=bcs,
    bc_velocity=bc_velocity,
    parameters=parameters,
    problem_topo=problem_topo,
    shift=shift,
)

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
lame_mu, lame_lambda = mechanics_tool.lame_compute(
    parameters.young_modulus, parameters.poisson
)
measure = CutFemSolver.dxq if parameters.cutFEM == 1 else ufl.dx

cost = problem_topo.cost(uh, ph, lame_mu, lame_lambda, measure, parameters)
shape_derivative = problem_topo.shape_derivative_integrand(
    uh, ph, lame_mu, lame_lambda, parameters, measure
)
vm_list = data_manipulation.create_list_vm(
    msh, uh, parameters, lame_mu, lame_lambda, 0, ls_func, V_ls, Q, 0
)

# ----------------------------
# Save initial results
# ----------------------------
time = 0.0
xdmf_file = io.XDMFFile(msh.comm, "res/results.xdmf", "w")
xdmf_file.write_mesh(msh)
velocity_field = fem.Function(V_ls)
velocity_field.x.array[:] = 0.0

for f, name in zip(
    [ls_func, uh, ph, vm_list, velocity_field],
    ["ls_func", "disp", "dual", "vm_list", "velocity"],
):
    f.name = name
    xdmf_file.write_function(f, time)


# ---------- Temporary level-set used during line-search / advection ----------
ls_func_temp = fem.Function(V_ls)
ls_func_temp.x.array[:] = CutFemSolver.level_set.x.array
ls_func_temp.x.scatter_forward()

crit_0 = 1e10
crit = [1e3, 1e6, 1e6, 1e6]  # stagnation criteria history
lagrangian = [1e3, 1e6, 1e6, 1e6]
cv = 0  # 0 if convergence is reached, 1 otherwise.


if rank == 0:
    print(style.RED + "##########################################")
    print(style.RED + "##### Initialization of the problem  #####")
    print(style.RED + "##########################################")
    print(style.WHITE + " ")

# temporary placeholder
xsi_temp = fem.Function(V_ls)

# ---------- Initial vm_list and print ----------
k = 1.0
vm_list = data_manipulation.create_list_vm(
    msh,
    uh,
    parameters,
    lame_mu,
    lame_lambda,
    0,
    CutFemSolver.level_set if parameters.cutFEM == 1 else ls_func_temp,
    V_ls,
    Q,
    0,
)
max_vm = k * np.max(vm_list.x.array[:])
max_vm = comm.allreduce(max_vm, op=MPI.MAX)

# ---------- prepare measures and initial quantities ----------
if parameters.cutFEM == 1:
    measure = CutFemSolver.dxq
    previous_cost = problem_topo.cost(
        uh, ph, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, measure, parameters
    )
    # already computed shape_derivative above
    previous_constraint = problem_topo.constraint(
        uh, lame_mu, lame_lambda, parameters, measure, 0, vm_list
    )
    almMethod.maj_param_constraint_optim(parameters, previous_constraint)
    dual_operator = problem_topo.dual_operator(
        uh,
        CutFemSolver.lame_mu,
        CutFemSolver.lame_lambda,
        parameters,
        msh,
        measure,
        vm_list,
    )
    shape_derivative_integrand_constraint = (
        problem_topo.shape_derivative_integrand_constraint(
            uh, ph, lame_mu, lame_lambda, parameters, ufl.dx, vm_DG=vm_list
        )
    )

else:
    measure = ufl.dx
    previous_cost = problem_topo.cost(
        uh,
        ph,
        ErsatzSolver.lame_mu_fic,
        ErsatzSolver.lame_lambda_fic,
        measure,
        parameters,
    )
    previous_constraint = problem_topo.constraint(
        uh,
        ErsatzSolver.lame_mu_fic,
        ErsatzSolver.lame_lambda_fic,
        parameters,
        measure,
        ErsatzSolver.xsi,
    )
    almMethod.maj_param_constraint_optim(parameters, previous_constraint)
    dual_operator = problem_topo.dual_operator(
        uh,
        ErsatzSolver.lame_mu_fic,
        ErsatzSolver.lame_lambda_fic,
        parameters,
        msh,
        measure,
    )
    shape_derivative_integrand_constraint = (
        problem_topo.shape_derivative_integrand_constraint(
            uh, ph, lame_mu, lame_lambda, parameters, ufl.dx, vm_DG=vm_list
        )
    )

# init counters, parameters
lagrangian_cost_previous = 1e3
lagrangian_cost = 1e3
adv_bool = 1
c_param_HJ = 0.5
n_k = 1
c_k = 1

velocity_field = fem.Function(V_ls)
velocity_field.x.array[:] = CutFemSolver.level_set.x.array * 0

# almMethod.init_param_constraint_optim(previous_constraint, parameters, cost)
resources = prepare_descent(msh, V_ls, parameters)

# ============================================================
# MEMORY OPTIMIZATION: Create reusable fem.Function objects
# ============================================================
solve = fem.Function(V_ls)
solve_temp = fem.Function(V_ls)
ls_new = fem.Function(V_ls)
v_reg = fem.Function(V_ls)
velocity_field_temp = fem.Function(V_ls)

# Pre-compile forms for velocity normalization
L2_form_vel = fem.form(ufl.inner(velocity_field, velocity_field) * ufl.dx)
grad_form_vel = fem.form(
    ufl.inner(ufl.grad(velocity_field), ufl.grad(velocity_field)) * ufl.dx
)

# Create vm_list once and reuse it
vm_calculator = data_manipulation.VonMisesCalculator(
    msh,
    V_ls,
    Q,
    uh,
    lame_mu,
    lame_lambda,
    CutFemSolver.level_set if parameters.cutFEM == 1 else ls_func_temp,
)
vm_list = vm_calculator.compute()

i = 0

# main optimization loop (stop when max iterations or criteria satisfied)


def run_optimization_iteration():
    global c_param_HJ, cv, i, uh, ph, cost, constraint, lagrangian_cost, previous_cost, previous_constraint, lagrangian_cost_previous, time, adv_bool, ls_func_temp, shape_derivative, shape_derivative_integrand_constraint, dual_operator, max_vm, vect_cost, vect_constraint, vect_target_constraint, vm_calculator

    # adapt HJ regularization parameter
    c_param_HJ = opti_tool.adapt_c_HJ(
        c_param_HJ, crit, parameters.tol_cost_func, lagrangian
    )
    if rank == 0:
        print(style.BLUE + "iteration number : ", i)
        print(style.WHITE + "")
        print(f"RAM used: {get_memory_usage() / 1024 / 1024:.3f} GB")

    # update ALM parameters
    almMethod.maj_param_constraint_optim(parameters, previous_constraint)

    # ---------- Descent direction ----------
    v_reg = descent_direction(
        CutFemSolver.level_set,
        msh,
        parameters,
        bc_velocity,
        V_ls,
        previous_constraint,
        shape_derivative_integrand_constraint,
        shape_derivative,
        resources,
    )
    # v_reg interpolation into velocity_field
    if isinstance(v_reg, fem.Function):
        velocity_field.x.array[:] = v_reg.x.array
    else:
        try:
            pts_vel = (
                V_ls.element.interpolation_points()
                if callable(getattr(V_ls.element, "interpolation_points", None))
                else V_ls.element.interpolation_points
            )
            vel_expr = fem.Expression(v_reg, pts_vel)
            velocity_field.interpolate(vel_expr)
        except Exception:
            if rank == 0:
                print(
                    "Warning: descent direction computation failed, setting velocity to zero"
                )
            velocity_field.x.array[:] = 0.0
    velocity_field.x.scatter_forward()

    # Normalize velocity in place
    norm_factor = velocity_normalization(v_reg, parameters.alpha_reg_velocity)
    velocity_field.x.array[:] = v_reg.x.array * norm_factor
    velocity_field.x.scatter_forward()

    # max velocity across ranks
    max_velocity_local = np.max(np.abs(velocity_field.x.array[:]))
    max_velocity = comm.allreduce(max_velocity_local, op=MPI.MAX)

    # ---------- Advection: Hamilton-Jacobi ----------
    solve.x.array[:] = CutFemSolver.level_set.x.array
    solve.x.scatter_forward()
    solve_temp.x.array[:] = solve.x.array
    solve_temp.x.scatter_forward()

    adv_inner_loop = True
    while adv_inner_loop:
        j = 0
        cv = 0
        # reuse ls_func_temp object in-place
        ls_func_temp.x.array[:] = solve.x.array
        ls_func_temp.x.scatter_forward()
        CutFemSolver.level_set.x.array[:] = solve.x.array
        CutFemSolver.level_set.x.scatter_forward()

        while j < parameters.j_max:
            AdvectionSolver.set_level_set(ls_func_temp)
            ls_new_arr = AdvectionSolver.cut_fem_adv(
                velocity_field, (1.0 / adv_bool) * parameters.dt
            ).x.array
            ls_new.x.array[:] = ls_new_arr
            ls_new.x.scatter_forward()

            # copy values into ls_func_temp in-place
            ls_func_temp.x.array[:] = ls_new_arr
            ls_func_temp.x.scatter_forward()
            j += 1

            # periodic reinitialization in-place
            if (j % parameters.freq_reinit) == 0:
                ReinitSolver.reinitializationPC_inplace(
                    ls_func_temp, parameters.step_reinit
                )

        # ---------- Recompute primal/adjoint ----------
        while ((parameters.adapt_time_step + 1) * cv) == 0:
            if parameters.cutFEM == 1:
                uh = CutFemSolver.primal_problem(ls_func_temp, parameters)
                CutFemSolver.update_measures_and_quadratures(ls_func_temp)
                cost = problem_topo.cost(
                    uh,
                    ph,
                    CutFemSolver.lame_mu,
                    CutFemSolver.lame_lambda,
                    CutFemSolver.dxq,
                    parameters,
                )
                shape_derivative = problem_topo.shape_derivative_integrand(
                    uh,
                    ph,
                    CutFemSolver.lame_mu,
                    CutFemSolver.lame_lambda,
                    parameters,
                    CutFemSolver.dxq,
                )

                vm_list_temp = vm_calculator.compute()
                vm_list.x.array[:] = vm_list_temp.x.array
                vm_list.x.scatter_forward()

                constraint = problem_topo.constraint(
                    uh,
                    lame_mu,
                    lame_lambda,
                    parameters,
                    CutFemSolver.dxq,
                    0,
                    vm_list,
                    c_k,
                )
                almMethod.maj_param_constraint_optim_slack(parameters, constraint)

                if parameters.cost_func != "compliance":
                    dual_operator = problem_topo.dual_operator(
                        uh,
                        CutFemSolver.lame_mu,
                        CutFemSolver.lame_lambda,
                        parameters,
                        msh,
                        CutFemSolver.dxq,
                        vm_list,
                        c_k,
                    )
                    CutFemSolver.update_measures_and_quadratures(ls_func_temp)
                    ph = CutFemSolver.adjoint_problem(uh, dual_operator)

                shape_derivative_integrand_constraint = (
                    problem_topo.shape_derivative_integrand_constraint(
                        uh,
                        ph,
                        lame_mu,
                        lame_lambda,
                        parameters,
                        CutFemSolver.dxq,
                        vm_list,
                        c_k,
                    )
                )
            else:
                ErsatzSolver.heaviside_inplace(ls_func_temp, xsi_temp)
                uh, ph = ErsatzSolver.ersatz_solver(ls_func_temp, parameters)
                measure = ufl.dx
                CutFemSolver.update_measures_and_quadratures(ls_func_temp)
                cost = problem_topo.cost(
                    uh,
                    ph,
                    ErsatzSolver.lame_mu_fic,
                    ErsatzSolver.lame_lambda_fic,
                    measure,
                    parameters,
                )
                shape_derivative = problem_topo.shape_derivative_integrand(
                    uh,
                    ph,
                    ErsatzSolver.lame_mu_fic,
                    ErsatzSolver.lame_lambda_fic,
                    parameters,
                    measure,
                )
                constraint = problem_topo.constraint(
                    uh,
                    ErsatzSolver.lame_mu_fic,
                    ErsatzSolver.lame_lambda_fic,
                    parameters,
                    measure,
                    xsi_temp,
                )
                almMethod.maj_param_constraint_optim_slack(parameters, constraint)
                dual_operator = problem_topo.dual_operator(
                    uh,
                    ErsatzSolver.lame_mu_fic,
                    ErsatzSolver.lame_lambda_fic,
                    parameters,
                    msh,
                    measure,
                )

            # compute Lagrangian cost
            lagrangian_cost = opti_tool.lagrangian_cost(cost, constraint, parameters)
            if rank == 0:
                print(style.YELLOW + "cost previous = ", previous_cost)
                print(style.YELLOW + "cost = ", cost)
                print(style.WHITE + "C(Ω) = ", float(constraint))

            cv = (
                1
                if cost < (previous_cost * (1.0 + parameters.tol_cost_func))
                or (parameters.adapt_time_step == 0)
                else 0
            )

        # if i == 1:
        #     constraint_derivative = abs(constraint - previous_constraint) / (max_velocity*parameters.dt * parameters.j_max)
        #     cost_derivative = abs(cost - previous_cost) / (max_velocity*parameters.dt * parameters.j_max)
        #     almMethod.init_param_constraint_optim(constraint_derivative, parameters, cost_derivative, 100)

        parameters.dt, adv_bool = opti_tool.catch_NAN(
            cost, lagrangian_cost, constraint, parameters.dt, adv_bool
        )
        parameters.j_max = (
            opti_tool.adapt_HJ(
                lagrangian_cost,
                lagrangian_cost_previous,
                parameters.j_max,
                parameters.dt,
                parameters,
            )
            if adv_bool < 2
            else 1
        )
        adv_inner_loop = False

    # update iteration history
    crit[3], crit[2], crit[1], crit[0] = (
        crit[2],
        crit[1],
        crit[0],
        abs(cost - previous_cost) / (previous_cost + 1e-30),
    )
    if rank == 0:
        print("criterion of convergence = ", crit[0])

    # accept new level-set
    CutFemSolver.level_set.x.array[:] = ls_func_temp.x.array
    CutFemSolver.level_set.x.scatter_forward()
    ls_func.x.array[:] = ls_func_temp.x.array
    ls_func.x.scatter_forward()
    ErsatzSolver.xsi = xsi_temp
    lagrangian_cost_previous = lagrangian_cost

    # gather results
    # cost and constraint are already reduced in problem.py
    vect_cost.append(cost)
    vect_constraint.append(constraint)
    vect_target_constraint.append(parameters.target_constraint)
    previous_cost = cost
    previous_constraint = constraint

    # Update vm_list in-place using the optimized calculator
    vm_list_temp = vm_calculator.compute()
    vm_list.x.array[:] = vm_list_temp.x.array
    vm_list.x.scatter_forward()

    # Compute and reduce max_vm
    max_vm = np.max(vm_list.x.array[:])
    max_vm = comm.allreduce(max_vm, op=MPI.MAX)

    # Limit history size
    if len(vect_cost) > MAX_HISTORY:
        vect_cost = vect_cost[-MAX_HISTORY:]
        vect_constraint = vect_constraint[-MAX_HISTORY:]
        vect_target_constraint = vect_target_constraint[-MAX_HISTORY:]

    # Collect memory usage from all ranks
    local_mem = get_memory_usage()
    all_mems = comm.gather(local_mem, root=0)

    # write results
    if rank == 0:
        try:
            output_files["cost_func"].write("\n" + str(cost))
            output_files["constraint"].write("\n" + str(constraint))
            output_files["param_lagrangian"].write("\n" + str(lagrangian_cost_previous))
            output_files["max_vm"].write("\n" + str(max_vm))
            output_files["memory"].write("\n" + str(local_mem))
            output_files["memory_per_process"].write(
                "\n" + " ".join(map(str, all_mems))
            )
            if i % 10 == 0:
                for f in output_files.values():
                    f.flush()
        except Exception as e:
            print(f"Error writing results: {e}")

    time += 1.0
    if i % 1 == 0:
        for f, name in zip(
            [ls_func, uh, ph, vm_list, velocity_field],
            ["ls_func", "disp", "dual", "vm_list", "velocity"],
        ):
            f.name = name
            xdmf_file.write_function(f, time)
    i += 1
    gc.collect()


# main optimization loop
while (i < parameters.max_incr) and (
    (abs(crit[0]) > parameters.tol_cost_func)
    or (abs(crit[1]) > parameters.tol_cost_func)
    or (abs(crit[2]) > parameters.tol_cost_func)
    or (abs(crit[3]) > parameters.tol_cost_func)
):
    run_optimization_iteration()

# Close all output files
if rank == 0:
    for f in output_files.values():
        f.close()

# Close XDMF file
xdmf_file.close()

end_time_total = time_module.time()
execution_time = end_time_total - start_time_total

if rank == 0:
    print(
        style.GREEN
        + f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
    )
    print(style.GREEN + "Optimization completed successfully")
