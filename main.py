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
from create_mesh import *
from ersatz_method import *
from cutfem_method import *
from levelSet_tool import *
from extension_regularization import *
from geometry_initialization import *
import almMethod 

import shutil
import os

import mechanics_tool
import data_manipulation
import opti_tool
import problem
import almMethod 

import gmsh

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

type_constraint_vm = "PN"

#"UKS" don't work 
i = 0
###################################
#### parameters initialization ####
###################################

parameters = Parameters()
temp = 1 
if temp == 0:
    parameters.set__paramManually()
else:
    print("Enter the name of your data folder")
    print("- for compliance : param_compliance.txt")
    print("- for Von Mises : param_vonMises.txt")
    print("- for area : param_area.txt")

    name = "param_VonMises.txt" # input()
    #name = "param_compliance.txt"
    parameters.set__paramFolder(name)

###################################
#### 	Mesh generation 	   ####
###################################
print("Choose the test case:")
print(" - for 3D : '3D' ")
print(" - for L shape write : 'L_shape' ")
print(" - for cantilever in 2D : 'rectangle' ")
test_case = "L_shape"  #input()
xdmf_filename =  "mesh/L_VM.xdmf"

if test_case=='rectangle':
    parameters.lx = 2
    parameters.ly = 1
    parameters.lz = 0
    msh = create_mesh_2D(parameters.lx, parameters.ly, int(parameters.lx/parameters.h),int(parameters.ly/parameters.h))
elif test_case=='L_shape':
    parameters.lx = 1
    parameters.ly = 1
    parameters.lz = 0
    with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
        msh, ct, _ = io.gmshio.read_from_msh("mesh/rectangle.msh", MPI.COMM_WORLD, 0, gdim=2)
        xdmf.write_mesh(msh)
elif test_case=='3D':
    parameters.lx = 2
    parameters.ly = 1
    parameters.lz = 1
    msh = create_mesh_3D(parameters.lx, parameters.ly,  parameters.lz, int(parameters.lx/parameters.h),\
        int(parameters.ly/parameters.h), int(parameters.lz/parameters.h))
else :
    print("not implemented test case")

msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim-1)

###################################
### Initialization of the spaces###
###################################
V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
V_vm = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim, )))
V_ls = fem.functionspace(msh, ("Lagrange", 1))
Q = fem.functionspace(msh, ("DG", 0))
V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, )))

# Initialization of the spacial coordinate
x = ufl.SpatialCoordinate(msh)

## Initialization of the level set function
# l.s. function is used to defined the geometry

if test_case == "L_shape":
    ls_func_ufl = level_set_L_shape(x) 
elif test_case == "rectangle":
    ls_func_ufl = level_set(x,parameters) 
elif test_case == "3D":
    ls_func_ufl = level_set_3D(x,parameters)
else :
    print("not implemented test case")


ls_func_expr = fem.Expression(ls_func_ufl, V_ls.element.interpolation_points())
ls_func = Function(V_ls)
ls_func.interpolate(ls_func_expr)


## Initialization of BC   
# Dirichlet condition initialization

def clamped_boundary_cantilever(x):
    return np.isclose(x[0], 0)

def clamped_boundary_L_shape(x):
    return (x[1]>(1. -1e-6))


dim = msh.topology.dim
fdim = msh.topology.dim - 1 #facet dimension


if test_case == "L_shape":
    boundary_facets = mesh.locate_entities_boundary(msh, fdim,clamped_boundary_L_shape)
    u_D = np.array([0,0], dtype=ScalarType)
elif test_case =="3D":
    boundary_facets = mesh.locate_entities_boundary(msh, fdim,clamped_boundary_cantilever)
    u_D = np.array([0,0,0], dtype=ScalarType)
elif test_case == "rectangle":
    boundary_facets = mesh.locate_entities_boundary(msh, fdim,clamped_boundary_cantilever)
    u_D = np.array([0,0], dtype=ScalarType)
else :
    print("not implemented test case")

bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)

# Neumann condition initialization for load traction

def load_marker(x):
    if test_case == "L_shape":
        return np.logical_and(x[0]>(parameters.lx-1e-6),x[1]>0.35)
    
    elif test_case == "3D":
        return np.logical_and(np.isclose(x[0],parameters.lx),np.logical_and(x[1]<(0.7),x[1]>(0.3))) 
    elif test_case == "rectangle":
        return np.logical_and(np.isclose(x[0],parameters.lx),np.logical_and(x[1]<(0.55),x[1]>(0.45))) 
    else:
        print("not implemented test case")



facet_indices, facet_markers = [], []
facets = mesh.locate_entities(msh, fdim, load_marker)
boundary_dofs = fem.locate_dofs_geometrical(V_ls, load_marker) # collect dofs where Dirichlet bc want to be imposed for the velocity field
bc_velocity = fem.dirichletbc(ScalarType(0.), boundary_dofs, V_ls) # dirichlet bc fr the velocity field
facet_indices.append(facets)
facet_markers.append(np.full_like(facets, 2))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)

if test_case=="3D":
    shift = fem.Constant(msh, ScalarType((0., -parameters.strenght, 0.)))
elif test_case == "rectangle" or test_case == "L_shape": 
    shift = fem.Constant(msh, ScalarType((0., -parameters.strenght)))
else :
    print("not implemented test case")


Reinitialization = Reinitialization(ls_func, V_ls=V_ls, l=parameters.l_reinit)
ErsatzMethod = ErsatzMethod(ls_func,V_ls, V, ds = ds, bc = bc, bc_velocity = bc_velocity, parameters = parameters, shift = shift)
CutFemMethod = CutFemMethod(ls_func,V_ls, V, ds = ds, bc = bc, bc_velocity = bc_velocity,  parameters = parameters, shift = shift)

lame_mu,lame_lambda = mechanics_tool.lame_compute(parameters.young_modulus,parameters.poisson)



velocity_field = Function(V_ls)
velocity_field.x.array[:] = CutFemMethod.level_set.x.array*0

###################################
####    Reinitialization       ####
###################################
CutFemMethod.level_set, temp_func = Reinitialization.reinitializationPC(CutFemMethod.level_set,parameters.step_reinit)
               
# ls_predict, temp_func = Reinitialization.predictor(CutFemMethod.level_set)
# CutFemMethod.level_set.x.array[:] = ls_predict.x.array

# num_step = 0
# while (num_step < 3):
#     num_step += 1
#     ls_correct = Reinitialization.corrector(CutFemMethod.level_set)
#     CutFemMethod.level_set.x.array[:] = ls_correct.x.array
#     CutFemMethod.level_set.x.scatter_forward() 
    


#Creation of temporary level set function, wich will be used to actualize level_set function if 
#the direction is a descent direction (ie: J(\Omega_n+1)<J(\Omega_n) ) 
ls_func_temp = Function(V_ls)
ls_func_temp.x.array[:] = CutFemMethod.level_set.x.array
crit_0 = 1e10
crit =  [1e+3,1e+6,1e+6,1e+6] #criterion of stagnation for the previous 3 iterations
lagrangian =  [1e+3,1e+6,1e+6,1e+6] #criterion of stagnation of the lagrangian for the previous 3 iterations
cv = 0 # 0 if convergence is reached ie: J(\Omega_n+1)-J(\Omega_n)>tol_compliance // 1 else 

########################################################
####    Definition of trial and test function       ####
########################################################

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
uh = fem.Function(V)
ph = fem.Function(V)
u_r = ufl.TrialFunction(V_ls)
v_r = ufl.TestFunction(V_ls)
ls_func_n = ufl.TrialFunction(V_ls)
ls_func_test = ufl.TestFunction(V_ls)

print(style.RED+'##########################################')
print(style.RED+'##### Initialization of the problem  #####')
print(style.RED+'##########################################')
print(style.WHITE+" ")

##########################################
## DUAL and PRIMAL problem :
##########################################

if parameters.cost_func=="compliance":
    problem_topo = problem.Compliance_Problem()
elif parameters.cost_func == "VonMises":
    problem_topo = problem.VMLp_Problem()
elif parameters.cost_func == "Area":
    problem_topo = problem.AreaProblem()
else :
    print("problem not implemented")

xsi_temp = Function(V_ls)


if parameters.cutFEM == 1:
    uh = CutFemMethod.primal_problem(ls_func_temp,parameters)
    cost_integrand = problem_topo.cost_integrand(uh,lame_mu,lame_lambda,parameters)
    cost = problem_topo.cost(uh,ph,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,CutFemMethod.dxq,parameters)

    shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,parameters,CutFemMethod.dxq)
    vm_list = data_manipulation.create_list_vm(msh,uh,parameters,lame_mu,lame_lambda,0,CutFemMethod.level_set,V_ls,Q,0)

    constraint = problem_topo.constraint(uh,lame_mu,lame_lambda,parameters,CutFemMethod.dxq,0,vm_list)
    almMethod.maj_param_constraint_optim_slack(parameters,constraint)
    if parameters.cost_func != "compliance" : 
        dual_operator  = problem_topo.dual_operator(uh,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,parameters,msh,CutFemMethod.dxq,vm_list)
        ph = CutFemMethod.adjoint_problem(uh,parameters,ls_func_temp,dual_operator)
    shape_derivative_constraint_integrand = problem_topo.shape_derivative_integrand_constraint(uh,ph,lame_mu,lame_lambda,parameters,CutFemMethod.dxq,vm_list)

else:
    uh, ph = ErsatzMethod.ersatz_solver(ls_func, parameters) 
    cost_integrand = problem_topo.cost_integrand(uh,lame_mu,lame_lambda,parameters)
    cost = dolfinx.fem.assemble_scalar(fem.form(0.5*(2.0*ErsatzMethod.lame_mu_fic  * ufl.inner(mechanics_tool.strain(uh), mechanics_tool.strain(uh))  + ErsatzMethod.lame_lambda_fic *  ufl.inner(ufl.nabla_div(uh), ufl.nabla_div(uh)) )*ufl.dx))

    #problem_topo.cost(uh,ph,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,ufl.dx,parameters)
    shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,parameters,ufl.dx)




k = 1.
vm_list = data_manipulation.create_list_vm(msh,uh,parameters,lame_mu,lame_lambda,0,CutFemMethod.level_set,V_ls,Q,0)
max_vm = k*np.max(vm_list.x.array[:])
print("parameters.elasticity_limit = ",max_vm/parameters.elasticity_limit)


time = 0.
xdmf_ls = io.XDMFFile(msh.comm, "res/level_set.xdmf", "w")
xdmf_ls.write_mesh(msh)
ls_func.name = "ls_func"
xdmf_ls.write_function(ls_func, time)
uh.name = "disp"
xdmf_ls.write_function(uh, time)
ph.name = "dual"
xdmf_ls.write_function(ph, time)
vm_list.name = "vm_list"
xdmf_ls.write_function(vm_list, time)
time += 1

time_ls = 0.
xdmf_ = io.XDMFFile(msh.comm, "res/debogue.xdmf", "w")
xdmf_.write_mesh(msh)
ls_func.name = "ls_func"
xdmf_.write_function(ls_func, time_ls)

if parameters.cutFEM == 1:
    measure = CutFemMethod.dxq
    previous_cost = problem_topo.cost(uh,ph,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,measure,parameters)
    shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,parameters,measure)
    vm_list = data_manipulation.create_list_vm(msh,uh,parameters,lame_mu,lame_lambda,0,CutFemMethod.level_set,V_ls,Q,0)
    
    constraint = problem_topo.constraint(uh,lame_mu,lame_lambda,parameters,measure,0, vm_list)
    almMethod.maj_param_constraint_optim(parameters,constraint)
    dual_operator  = problem_topo.dual_operator(uh,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,parameters,msh,measure,vm_list)
    shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,parameters,CutFemMethod.dxq)             
    shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(uh,ph,lame_mu,lame_lambda,parameters,CutFemMethod.dxq,vm_list)

else:
    measure = ufl.dx
    previous_cost = problem_topo.cost(uh,ph,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,measure,parameters)

    shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,parameters,measure)

    constraint = problem_topo.constraint(uh,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,parameters,measure,ErsatzMethod.xsi)
    almMethod.maj_param_constraint_optim(parameters,constraint)
    dual_operator  = problem_topo.dual_operator(uh,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,parameters,msh,measure)
    shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,parameters,ufl.dx)             
    shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(uh,ph,lame_mu,lame_lambda,parameters,ufl.dx)

lagrangian_cost_previous = 10**3
lagrangian_cost = 10**3
adv_bool = 1
c_param_HJ = 0.5

n_k = 1
c_k = 1

almMethod.init_param_constraint_optim(constraint,parameters,cost)

while (i<parameters.max_incr) and ((abs(crit[0])>parameters.tol_cost_func) or \
    (abs(crit[1])>parameters.tol_cost_func) or ((abs(crit[2])>parameters.tol_cost_func)) or (abs(crit[3])>parameters.tol_cost_func)):
    c_param_HJ = opti_tool.adapt_c_HJ(c_param_HJ,crit,parameters.tol_cost_func,lagrangian)
    print("c = ",c_param_HJ )
    cv = 0
    if adv_bool == 0:
        adv_bool = 1

    print(style.BLUE+"iteration number : ", i )
    print(style.WHITE+"")

    ##########################################
    ## ALM: parameters update
    ##########################################
    almMethod.maj_param_constraint_optim(parameters,constraint)

    ##########################################
    ## Velocity field 
    ##########################################
    if parameters.cutFEM == 1:
        velocity_field = CutFemMethod.descent_direction(CutFemMethod.level_set,parameters,\
                                        constraint,shape_derivative_integrand_constraint,shape_derivative,ErsatzMethod.xsi)
    else: 
        velocity_field = ErsatzMethod.descent_direction(CutFemMethod.level_set,parameters,\
                                        constraint,shape_derivative_integrand_constraint,shape_derivative,ErsatzMethod.xsi)
    
    velocity_field = CutFemMethod.velocity_normalization(velocity_field,parameters.alpha_reg_velocity)

    velocity_expr = fem.Expression(velocity_field, V_ls.element.interpolation_points())
    velocity = fem.Function(V_ls)
    velocity.interpolate(velocity_expr)

    max_velocity = comm.allreduce(np.max(np.abs(velocity.x.array[:])),op=MPI.MAX)

    parameters.dt  =  opti_tool.compliance_adapt_dt(lagrangian_cost,lagrangian_cost_previous,max_velocity,parameters,c_param_HJ)
    print("dt = ",parameters.dt)

    ##########################################
    ## Advection: HJ equation
    ##########################################

    ls_func_n = ufl.TrialFunction(V_ls)
    ls_func_test = ufl.TestFunction(V_ls)
    solve = fem.Function(V_ls)
    solve.x.array[:] = CutFemMethod.level_set.x.array
    solve_temp = fem.Function(V_ls)
    solve_temp.x.array[:] = solve.x.array
    while (adv_bool != 0):
        j = 0
        cv = 0
        ls_func_temp.x.array[:] = solve.x.array
        CutFemMethod.level_set.x.array[:] =  solve.x.array
        time_ls += 1
        ls_func_temp.name = "ls_func"
        xdmf_.write_function(ls_func_temp, time_ls)
        while j< parameters.j_max:

            ls_func_temp = CutFemMethod.cut_fem_adv_temp(ls_func_temp,(1/adv_bool)*parameters.dt, velocity_field)
             
            j += 1
            
            ##########################################
            ## Reinitialization
            ##########################################
            if ((j%parameters.freq_reinit)==0):
                ls_func_temp, temp_func = Reinitialization.reinitializationPC(ls_func_temp,parameters.step_reinit)
                # ls_func_temp = Reinitialization.predictor(ls_func_temp)
                # num_step = 0
                # while (num_step < parameters.step_reinit):
                #     num_step += 1                    
                #     ls_func_temp = Reinitialization.corrector(ls_func_temp)

        ##########################################
        ## Calculus of new solution of 
        ##   dual and primal problem
        ##########################################

        
        while ((parameters.adapt_time_step +1) * cv)==0:
            # if parameters.cutFEM == 1:
            #    # dual_operator  = problem_topo.dual_operator(uh,cutfem_method.lame_mu,cutfem_method.lame_lambda,parameters,msh,measure)
            #     uh, ph = CutFemMethod.cutfem_solver(ls_func_temp,parameters,problem_topo)
            # else:
            #     xsi_temp = ErsatzMethod.heaviside(ls_func_temp)
            #     uh, ph = ErsatzMethod.ersatz_solver(ls_func_temp, parameters) 

            if parameters.cutFEM == 1:
                uh = CutFemMethod.primal_problem(ls_func_temp,parameters)
                cost_integrand = problem_topo.cost_integrand(uh,lame_mu,lame_lambda,parameters)
                CutFemMethod.set_measure_dxq(ls_func_temp)
                cost = problem_topo.cost(uh,ph,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,CutFemMethod.dxq,parameters)
    
                shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,parameters,CutFemMethod.dxq)
                vm_list = data_manipulation.create_list_vm(msh,uh,parameters,lame_mu,lame_lambda,0,CutFemMethod.level_set,V_ls,Q,0)
                max_vm = np.max(vm_list.x.array[:])

                # c_k = (n_k*((max_vm /parameters.elasticity_limit) / previous_cost) + (1-n_k)*c_k)
                constraint = problem_topo.constraint(uh,lame_mu,lame_lambda,parameters,CutFemMethod.dxq,0,vm_list,c_k)
                almMethod.maj_param_constraint_optim_slack(parameters,constraint)
                if parameters.cost_func != "compliance" : 
                    dual_operator  = problem_topo.dual_operator(uh,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,parameters,msh,CutFemMethod.dxq,vm_list,c_k)
                    CutFemMethod.set_measure_dxq(ls_func_temp)
                    ph = CutFemMethod.adjoint_problem(uh,parameters,ls_func_temp,dual_operator)

                shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(uh,ph,lame_mu,lame_lambda,parameters,CutFemMethod.dxq,vm_list,c_k)
            else:
                xsi_temp = ErsatzMethod.heaviside(ls_func_temp)
                uh, ph = ErsatzMethod.ersatz_solver(ls_func_temp, parameters) 

                measure = ufl.dx
                CutFemMethod.set_measure_dxq(ls_func_temp)
                cost = problem_topo.cost(uh,ph,ErsatzMethod.lame_mu_fic ,ErsatzMethod.lame_lambda_fic ,measure,parameters)

                shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,parameters,measure)

                constraint = problem_topo.constraint(uh,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,parameters,measure,ErsatzMethod.xsi)
    
                almMethod.maj_param_constraint_optim_slack(parameters,constraint)
                dual_operator  = problem_topo.dual_operator(uh,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,parameters,msh,measure)
                shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,ErsatzMethod.lame_mu_fic,ErsatzMethod.lame_lambda_fic,parameters,ufl.dx)             
                shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(uh,ph,lame_mu,lame_lambda,parameters,ufl.dx)

            lagrangian_cost = opti_tool.lagrangian_cost(cost,constraint,parameters)
            print(style.YELLOW+"cost previous = ",previous_cost)
            print(style.YELLOW+"cost = ",cost)
            print(style.WHITE+"C(\Omega) = ",float(constraint))

            if cost < (previous_cost *(1+parameters.tol_cost_func)) or (parameters.adapt_time_step==0):
                cv = 1
            else :
                cv = 0 

        ##########################################
        ## Save parameters
        ########################################## 

        lagrangian[3] = lagrangian[2]
        lagrangian[2] = lagrangian[1]
        lagrangian[1]= lagrangian[0]
        lagrangian[0] = abs((lagrangian_cost-lagrangian_cost_previous)/lagrangian_cost)

        time_ls += 1
        ls_func_temp.name = "ls_func"
        xdmf_.write_function(ls_func_temp, time_ls)

        parameters.dt, adv_bool = opti_tool.catch_NAN(cost,lagrangian_cost,constraint,parameters.dt,adv_bool)
        
        if adv_bool<2:
            parameters.j_max = opti_tool.adapt_HJ(lagrangian_cost,lagrangian_cost_previous,parameters.j_max,parameters.dt,parameters)
        else: 
            parameters.j_max = 1

    print("j_max = ", parameters.j_max)

    crit[3] = crit[2]
    crit[2] = crit[1]
    crit[1]= crit[0]
    crit[0] = abs(cost-previous_cost)/previous_cost

    print("criterion of convergence = ", crit[0])
    CutFemMethod.level_set.x.array[:] = ls_func_temp.x.array
    CutFemMethod.level_set.x.scatter_forward() 

    ErsatzMethod.xsi= xsi_temp

    lagrangian_cost_previous = lagrangian_cost
    
    collected_cost = comm.allreduce(cost,op=MPI.SUM)
    vect_cost.append(collected_cost)
    collected_constraint = comm.allreduce(constraint, op=MPI.SUM)
    vect_constraint.append(collected_constraint)
    vect_target_constraint.append(parameters.target_constraint)
    collected_lagrangian_cost_previous = comm.allreduce(lagrangian_cost_previous,op=MPI.SUM)
    previous_cost = cost
    k = 1.
    vm_list = data_manipulation.create_list_vm(msh,uh,parameters,lame_mu,lame_lambda,0,CutFemMethod.level_set,V_ls,Q,0)
    max_vm = k*np.max(vm_list.x.array[:])

    #  c_k = (n_k*((max_vm /parameters.elasticity_limit) / previous_cost) + (1-n_k)*c_k)
    
    if rank == 0:
        folder_cost_func.write("\n"+str(collected_cost))
        folder_constraint.write("\n"+str(collected_constraint))
        folder_param_lagrangian.write("\n"+str(collected_lagrangian_cost_previous))
        folder_max_vm.write("\n"+str(max_vm))


    if parameters.cutFEM == 0:
        lame_mu_fic_expr = fem.Expression(ErsatzMethod.lame_mu_fic, V_ls.element.interpolation_points())
        lame_mu_fic = fem.Function(V_ls)
        lame_mu_fic.interpolate(lame_mu_fic_expr)
        lame_lambda_fic_expr = fem.Expression(ErsatzMethod.lame_lambda_fic, V_ls.element.interpolation_points())
        lame_lambda_fic = fem.Function(V_ls)
        lame_lambda_fic.interpolate(lame_lambda_fic_expr)
        xsi_expr = fem.Expression(xsi_temp, V_ls.element.interpolation_points())
        xsi_temp = fem.Function(V_ls)
        xsi_temp.interpolate(xsi_expr)

    ls_func_temp.name = "ls_func_temp"
    xdmf_ls.write_function(ls_func_temp, time)
    
    uh.name = "disp"
    xdmf_ls.write_function(uh, time)
    ph.name = "dual"
    xdmf_ls.write_function(ph, time)
    temp_func.name = "temp func pred"
    xdmf_ls.write_function(temp_func, time)

    
    velocity_expr = fem.Expression(velocity_field, V_ls.element.interpolation_points())
    velocity = fem.Function(V_ls)
    velocity.interpolate(velocity_expr)
    velocity.name = "velocity"
    xdmf_ls.write_function(velocity, time)


    compliance = - (2.0*lame_mu  * ufl.inner(mechanics_tool.strain(uh), mechanics_tool.strain(ph))  + lame_lambda *  ufl.inner(ufl.nabla_div(uh), ufl.nabla_div(ph)))
        
    compliance_expr = fem.Expression(compliance, V_ls.element.interpolation_points())
    compliance = fem.Function(V_ls)
    compliance.interpolate(compliance_expr)
    compliance.name = "compliance cost "
    xdmf_ls.write_function(compliance, time)

    vm = mechanics_tool.von_mises(uh,lame_mu, lame_lambda, dim)
    vm_expr = fem.Expression(vm, Q.element.interpolation_points())
    vm = fem.Function(Q)
    vm.interpolate(vm_expr)
  
    vm.name = "sigmavm"
    xdmf_ls.write_function(vm, time)

    vm_list.name = "manip"
    xdmf_ls.write_function(vm_list, time)


    
    vm_expr = fem.Expression(vm-vm_list, Q.element.interpolation_points())
    vm_diff = fem.Function(Q)
    vm_diff.interpolate(vm_expr)
    vm_diff.name = "diff"
    xdmf_ls.write_function(vm_diff, time)

    # data_manipulation.save_data_compliance(xdmf_ls,CutFemMethod.level_set, V_ls, xsi_temp, velocity_field, V_ls, uh, V, ph, \
    #     mecanics_tool.von_mises(uh,CutFemMethod.lame_mu,CutFemMethod.lame_lambda,CutFemMethod.dim), Q, time)

    time += 1
    i += 1
