# # Copyright (c) 2026 ONERA and MINES Paris, France
# #
# # All rights reserved.
# #
# # This file is part of OptiCut.
# #
# # Author(s)     : Amina El Bachari
# #
# # Migration note: updated to CutFEMx 0.2 (DOLFINx 0.11)

# import ufl
# from ufl import FacetNormal, Measure, CellDiameter, avg, jump

# import cutfemx
# import cutfemx.petsc as cf_petsc

# from dolfinx import fem, mesh, la
# from dolfinx.fem import petsc as dolfinx_petsc
# from dolfinx.mesh import meshtags, locate_entities, locate_entities_boundary

# from mpi4py import MPI
# import numpy as np

# from petsc4py import PETSc
# from petsc4py.PETSc import ScalarType

# from utils import mechanics_tool


# class CutFEMElasticSolver:
#     r"""CutFEM linear-elasticity solver (CutFEMx 0.2 compatible)."""

#     def __init__(self, level_set, level_set_space, space_displacement, ds, bc,
#                  bc_velocity, parameters, problem_topo, shift):

#         self.level_set = fem.Function(level_set_space)
#         self.level_set.x.array[:] = level_set.x.array
#         self.level_set.x.scatter_forward()

#         self.mesh = self.level_set.function_space.mesh
#         self.space_displacement = space_displacement
#         self.cutFEM = parameters.cutFEM

#         lame_mu, lame_lambda = mechanics_tool.lame_compute(parameters.young_modulus,
#                                                            parameters.poisson)
#         self.V_ls = level_set_space
#         self.cost_func = parameters.cost_func

#         self.lame_mu = lame_mu
#         self.lame_lambda = lame_lambda
#         self.dim = self.mesh.topology.dim

#         self.bc_velocity = bc_velocity
#         self.tdim = self.dim
#         self.shift = shift

#         # Geometric objects
#         self.n = FacetNormal(self.mesh)
#         self.h = CellDiameter(self.mesh)
#         self.bc = bc
#         self.ds = Measure("ds", domain=self.mesh,
#                           subdomain_data=ds.subdomain_data() if hasattr(ds, 'subdomain_data') else ds)

#         # Trial and test functions
#         self.u_trial = ufl.TrialFunction(self.space_displacement)
#         self.v_test  = ufl.TestFunction(self.space_displacement)
#         self.uh = fem.Function(self.space_displacement)
#         self.ph = fem.Function(self.space_displacement)

#         self.gamma_N = 1e3
#         self.gamma   = 1e-5 * (self.lame_mu + self.lame_lambda)

#         self.fast_jit = {"cffi_extra_compile_args": ["-O0", "-w"]}

#         # KSP solvers
#         self.solver_primal = PETSc.KSP().create(self.mesh.comm)
#         self.solver_adjoint = PETSc.KSP().create(self.mesh.comm)
#         self._configure_ksp(self.solver_primal, parameters)
#         self._configure_ksp(self.solver_adjoint, parameters)

#         # ------------------------------------------------------------------
#         # CutFEMx 0.2: CutData creation (only once)
#         # ------------------------------------------------------------------
#         self.order    = 4
#         self.cut_data = cutfemx.cut(self.level_set)

#         self._build_measures()
#         self._build_variational_forms(parameters, problem_topo)

#         if parameters.cost_func != 'compliance':
#             self.p_const   = parameters.p_const
#             self.parameters = parameters
#             self._build_adjoint_forms(parameters, problem_topo)


#     def _configure_ksp(self, solver, parameters):
#         if hasattr(parameters, 'linear_solver') and parameters.linear_solver == "amg":
#             solver.setType(PETSc.KSP.Type.GMRES)
#             pc = solver.getPC()
#             pc.setType("gamg")
#             solver.setTolerances(rtol=1e-6)
#         else:
#             solver.setType(PETSc.KSP.Type.PREONLY)
#             pc = solver.getPC()
#             pc.setType(PETSc.PC.Type.LU)
#             pc.setFactorSolverType("mumps")


#     def _build_measures(self):
#         """Constructs or reconstructs UFL measures from cut_data."""
#         # 1. Volume (phi < 0)
#         self.solid_cells = cutfemx.locate_entities(self.cut_data, "phi<0")
#         self.solid_rules = cutfemx.runtime_quadrature(self.cut_data, "phi<0", self.order)
#         self.dx_solid = ufl.Measure("dx", domain=self.mesh, subdomain_id=0,
#                                     subdomain_data=[self.solid_cells, self.solid_rules])
#         self.dxq = self.dx_solid  # Alias required for main.py

#         # 2. Ghost facets (Ghost penalty)
#         self.ghost_facets = cutfemx.ghost_penalty_facets(self.cut_data, "phi<0")
#         self.dS_ghost = ufl.Measure("dS", domain=self.mesh, subdomain_id=1,
#                                     subdomain_data=self.ghost_facets, metadata={"quadrature_degree": self.order})


#     def _build_variational_forms(self, parameters=None, problem_topo=None):
#         u = self.u_trial
#         v = self.v_test

#         # CutFEM bilinear form
#         self.a_primal = (
#             2.0 * self.lame_mu * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(v)) * self.dx_solid
#             + self.lame_lambda * ufl.inner(ufl.nabla_div(u), ufl.nabla_div(v)) * self.dx_solid
#         )

#         if self.ghost_facets.size > 0:
#             self.a_primal += (
#                 self.gamma * avg(self.h)
#                 * ufl.inner(ufl.jump(ufl.grad(u)), ufl.jump(ufl.grad(v))) * self.dS_ghost
#             )

#         # Neumann right-hand side (uncut outer boundary)
#         self.L_primal = ufl.dot(self.shift, v) * self.ds(2)


#     def _build_adjoint_forms(self, parameters, problem_topo):
#         p     = self.u_trial
#         v_adj = self.v_test

#         self.a_adj = (
#             2.0 * self.lame_mu * ufl.inner(mechanics_tool.strain(p), mechanics_tool.strain(v_adj)) * self.dx_solid
#             + self.lame_lambda * ufl.inner(ufl.nabla_div(p), ufl.nabla_div(v_adj)) * self.dx_solid
#         )

#         if self.ghost_facets.size > 0:
#             self.a_adj += (
#                 avg(self.gamma) * avg(self.h)
#                 * ufl.inner(ufl.jump(ufl.grad(p)), ufl.jump(ufl.grad(v_adj))) * self.dS_ghost
#             )

#         self.L_adj = problem_topo.dual_operator(
#             self.uh, self.lame_mu, self.lame_lambda, parameters, self.mesh, self.dx_solid
#         )


#     def update_measures_and_quadratures(self, level_set, order=4):
#         self.level_set.x.array[:] = level_set.x.array
#         self.level_set.x.scatter_forward()
#         self.order = order

#         cutfemx.update(self.cut_data)
#         self._build_measures()
#         self._build_variational_forms()

#         if hasattr(self, 'a_adj'):
#             self._build_adjoint_forms(getattr(self, 'parameters', None), None)


#     def primal_problem(self, level_set, parameters):
#         self.update_measures_and_quadratures(level_set)
#         bcs_list = self.bc if isinstance(self.bc, list) else [self.bc]

#         self.a_cut_primal = cutfemx.fem.form(self.a_primal, jit_options=self.fast_jit)
#         self.L_std_primal = fem.form(self.L_primal, jit_options=self.fast_jit)

#         if hasattr(self, "A_primal") and self.A_primal is not None:
#             self.A_primal.destroy()
#         self.A_primal = cf_petsc.assemble_matrix(self.a_cut_primal, bcs=bcs_list)
#         self.A_primal.assemble()

#         if hasattr(self, "b_primal") and self.b_primal is not None:
#             self.b_primal.destroy()
#         self.b_primal = dolfinx_petsc.assemble_vector(self.L_std_primal)

#         # Dynamic official CutFEMx lifting on the cut form
#         with self.b_primal.localForm() as b_local:
#             cutfemx.fem.apply_lifting(b_local.array, [self.a_cut_primal], [bcs_list])

#         self.b_primal.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#         dolfinx_petsc.set_bc(self.b_primal, bcs_list)

#         active = cutfemx.fem.active_domain(self.a_cut_primal)
#         cf_petsc.deactivate_outside(self.A_primal, self.b_primal, active)
#         self.A_primal.assemble()

#         self.solver_primal.setOperators(self.A_primal)
#         self.solver_primal.solve(self.b_primal, self.uh.x.petsc_vec)
#         self.uh.x.scatter_forward()

#         return self.uh


#     def adjoint_problem(self, u, level_set, dual_operator):
#         self.update_measures_and_quadratures(level_set)
#         self.uh   = u
#         self.L_adj = dual_operator
#         bcs_list = self.bc if isinstance(self.bc, list) else [self.bc]

#         self.a_cut_adjoint = cutfemx.fem.form(self.a_adj, jit_options=self.fast_jit)
#         self.L_cut_adjoint = cutfemx.fem.form(self.L_adj, jit_options=self.fast_jit)

#         if hasattr(self, "A_adjoint") and self.A_adjoint is not None:
#             self.A_adjoint.destroy()
#         self.A_adjoint = cf_petsc.assemble_matrix(self.a_cut_adjoint, bcs=bcs_list)
#         self.A_adjoint.assemble()

#         if hasattr(self, "b_adjoint") and self.b_adjoint is not None:
#             self.b_adjoint.destroy()

#         if isinstance(self.L_adj, fem.Form):
#             self.b_adjoint = dolfinx_petsc.assemble_vector(self.L_adj)
#         else:
#             self.b_adjoint = cf_petsc.assemble_vector(self.L_cut_adjoint)

#         with self.b_adjoint.localForm() as b_local:
#             cutfemx.fem.apply_lifting(b_local.array, [self.a_cut_adjoint], [bcs_list])

#         self.b_adjoint.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
#         dolfinx_petsc.set_bc(self.b_adjoint, bcs_list)

#         active = cutfemx.fem.active_domain(self.a_cut_adjoint)
#         cf_petsc.deactivate_outside(self.A_adjoint, self.b_adjoint, active)
#         self.A_adjoint.assemble()

#         self.solver_adjoint.setOperators(self.A_adjoint)
#         self.solver_adjoint.solve(self.b_adjoint, self.ph.x.petsc_vec)
#         self.ph.x.scatter_forward()

#         return self.ph


#     def cutfem_solver(self, level_set, parameters, problem_topo=0):
#         self.level_set.x.array[:] = level_set.x.array

#         self.uh = self.primal_problem(level_set, parameters)

#         if parameters.cost_func == "compliance":
#             self.ph.x.array[:] = self.uh.x.array
#             self.ph.x.scatter_forward()
#         else:
#             adjoint = problem_topo.dual_operator(
#                 self.uh, self.lame_mu, self.lame_lambda, parameters, self.mesh, self.dx_solid
#             )
#             self.ph = self.adjoint_problem(self.uh, level_set, adjoint)

#         return self.uh, self.ph


#     def __del__(self):
#         for attr in ("A_primal", "A_adjoint", "b_primal", "b_adjoint",
#                      "solver_primal", "solver_adjoint"):
#             obj = getattr(self, attr, None)
#             if obj is not None:
#                 try:
#                     obj.destroy()
#                 except Exception:
#                     pass


# PETSc version with and without MPI!


# Copyright (c) 2026 ONERA and MINES Paris, France
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari
#
# Migration note: updated to CutFEMx 0.2 (DOLFINx 0.11)
# Stable CutFEMx assembly combined with PETSc KSP solver.

# import ufl
# from ufl import FacetNormal, Measure, CellDiameter, avg, jump

# import cutfemx
# import cutfemx.petsc as cf_petsc

# from dolfinx import fem, mesh, la
# from dolfinx.fem import petsc as dolfinx_petsc
# from dolfinx.mesh import meshtags, locate_entities, locate_entities_boundary

# from mpi4py import MPI
# import numpy as np

# from petsc4py import PETSc
# from petsc4py.PETSc import ScalarType

# from utils import mechanics_tool


# class CutFEMElasticSolver:
#     r"""CutFEM linear-elasticity solver aligned with the official CutFEMx 0.2 tutorial and PETSc KSP."""

#     def __init__(self, level_set, level_set_space, space_displacement, ds, bc,
#                  bc_velocity, parameters, problem_topo, shift):

#         self.level_set = fem.Function(level_set_space)
#         self.level_set.x.array[:] = level_set.x.array
#         self.level_set.x.scatter_forward()

#         self.mesh = self.level_set.function_space.mesh
#         self.space_displacement = space_displacement
#         self.cutFEM = parameters.cutFEM
#         self.parameters = parameters
#         self.problem_topo = problem_topo

#         lame_mu, lame_lambda = mechanics_tool.lame_compute(parameters.young_modulus,
#                                                            parameters.poisson)
#         self.V_ls = level_set_space
#         self.cost_func = parameters.cost_func

#         self.lame_mu = lame_mu
#         self.lame_lambda = lame_lambda
#         self.dim = self.mesh.topology.dim

#         self.bc_velocity = bc_velocity
#         self.shift = shift

#         # Fixed geometric objects
#         self.n = FacetNormal(self.mesh)
#         self.h = CellDiameter(self.mesh)
#         self.bc = bc
#         self.ds = Measure("ds", domain=self.mesh,
#                           subdomain_data=ds.subdomain_data() if hasattr(ds, 'subdomain_data') else ds)

#         # Solution functions
#         self.uh = fem.Function(self.space_displacement)
#         self.ph = fem.Function(self.space_displacement)

#         self.gamma_N = 1e3
#         self.gamma   = 1e-5 * (self.lame_mu + self.lame_lambda)
#         self.order   = 4
#         self.fast_jit = {"cffi_extra_compile_args": ["-O0", "-w"]}

#         # Initialization of PETSc KSP solvers
#         self.solver_primal = PETSc.KSP().create(self.mesh.comm)
#         self.solver_adjoint = PETSc.KSP().create(self.mesh.comm)
#         self._configure_ksp(self.solver_primal)
#         self._configure_ksp(self.solver_adjoint)

#         # Initial CutData
#         self.cut_data = cutfemx.cut(self.level_set)


#     def _configure_ksp(self, solver):
#         """Configures the PETSc linear solver (MUMPS by default for robustness)."""
#         if hasattr(self.parameters, 'linear_solver') and self.parameters.linear_solver == "amg":
#             solver.setType(PETSc.KSP.Type.GMRES)
#             pc = solver.getPC()
#             pc.setType("gamg")
#             solver.setTolerances(rtol=1e-6)
#         else:
#             solver.setType(PETSc.KSP.Type.PREONLY)
#             pc = solver.getPC()
#             pc.setType(PETSc.PC.Type.LU)
#             pc.setFactorSolverType("mumps")


#     def _solve_system(self, a_form, L_form, bcs_list, solver_ksp, solution_func):
#         """Assembles using the stable CutFEMx method and solves via PETSc KSP."""
#         # 1. Stable assembly (from the official tutorial)
#         # A_csr = cutfemx.fem.assemble_matrix(a_form, bcs=bcs_list)
#         # A_csr.scatter_reverse()

#         # b = cutfemx.fem.assemble_vector(L_form)
#         # cutfemx.fem.apply_lifting(b.array, [a_form], [bcs_list])
#         # b.scatter_reverse(la.InsertMode.add)
#         # for bc in bcs_list:
#         #     bc.set(b.array)


#         # a_cut_reg = cutfemx.fem.form(a_form, bcs=bcs_list, jit_options={"cache_dir": "ffcx-forms"})
#         # L_cut_reg = cutfemx.fem.form(L_form)

#         # ------------------------------------------------------------------
#         # Assembly via cutfemx.petsc (returns PETSc.Mat / PETSc.Vec)
#         # ------------------------------------------------------------------
#         b_reg = cf_petsc.assemble_vector(L_form)
#         b_reg.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
#         b_reg.assemble()
#         A_reg = cf_petsc.assemble_matrix(a_form, bcs=bcs_list)
#         A_reg.assemble()

#         # # Solve
#         # ksp = PETSc.KSP().create(self.mesh.comm)
#         # ksp.setType("cg")
#         # pc = ksp.getPC()
#         # pc.setType("hypre")
#         # ksp.setFromOptions()

#         # ksp.setOperators(A_reg)
#         # ksp.setUp()
#         # ksp.setTolerances(rtol=1e-8, atol=1e-12)
#         # ksp.solve(b_reg, solution_func.x.petsc_vec)
#         # solution_func.x.scatter_forward()


#         # Solve with CG + GAMG (robust iterative solver)
#         ksp = PETSc.KSP().create(self.mesh.comm)
#         ksp.setType("cg")
#         pc = ksp.getPC()
#         pc.setType("gamg")                   # GAMG instead of HYPRE
#         ksp.setFromOptions()

#         ksp.setOperators(A_reg)
#         ksp.setUp()
#         ksp.setTolerances(rtol=1e-8, atol=1e-12)
#         ksp.solve(b_reg, solution_func.x.petsc_vec)
#         solution_func.x.scatter_forward()

#         # active = cutfemx.fem.active_domain(a_form)
#         # cutfemx.fem.deactivate_outside(A_csr, b, active)

#         # # 2. Clean conversion of the CSR matrix (SciPy) to a valid PETSc.Mat
#         # scipy_mat = A_csr.to_scipy().tocsr()
#         # mat_petsc = PETSc.Mat().createAIJ(
#         #     size=scipy_mat.shape,
#         #     csr=(scipy_mat.indptr, scipy_mat.indices, scipy_mat.data),
#         #     comm=self.mesh.comm
#         # )
#         # mat_petsc.assemble()

#         # # 3. Creation of the RHS PETSc vector from the assembled b.array
#         # vec_petsc = PETSc.Vec().createWithArray(b.array, comm=self.mesh.comm)

#         # # 4. Resolution with the configured PETSc KSP
#         # solver_ksp.setOperators(mat_petsc)
#         # solver_ksp.solve(vec_petsc, solution_func.x.petsc_vec)
#         # solution_func.x.scatter_forward()

#         # # Memory cleanup
#         # mat_petsc.destroy()
#         # vec_petsc.destroy()

#         return solution_func


#     def primal_problem(self, level_set, parameters):
#         """Exact primal resolution according to the tutorial."""
#         self.level_set.x.array[:] = level_set.x.array
#         self.level_set.x.scatter_forward()

#         # Update of the cut geometry
#         cutfemx.update(self.cut_data)

#         # Tutorial measures
#         solid_cells = cutfemx.locate_entities(self.cut_data, "phi<0")
#         solid_rules = cutfemx.runtime_quadrature(self.cut_data, "phi<0", self.order)
#         dx_solid = ufl.Measure("dx", domain=self.mesh, subdomain_id=0,
#                                subdomain_data=[solid_cells, solid_rules], metadata={"quadrature_degree": self.order})
#         self.dxq = dx_solid  # Alias for main.py

#         ghost_facets = cutfemx.ghost_penalty_facets(self.cut_data, "phi<0")
#         dS_ghost = ufl.Measure("dS", domain=self.mesh, subdomain_id=1,
#                                subdomain_data=ghost_facets)

#         # UFL forms
#         u = ufl.TrialFunction(self.space_displacement)
#         v = ufl.TestFunction(self.space_displacement)
#         gdim = self.mesh.geometry.dim

#         a = ufl.inner(sigma_expr(u, self.lame_mu, self.lame_lambda, gdim), epsilon_expr(v)) * dx_solid
#         if ghost_facets.size > 0:
#             n_facet = FacetNormal(self.mesh)
#             h_avg = avg(CellDiameter(self.mesh))
#             a += (
#                 self.gamma * h_avg
#                 * ufl.inner(ufl.jump(ufl.grad(u), n_facet), ufl.jump(ufl.grad(v), n_facet))
#                 * dS_ghost
#             )

#         L = ufl.dot(self.shift, v) * self.ds(2)
#         bcs_list = self.bc if isinstance(self.bc, list) else [self.bc]

#         a_form = cutfemx.fem.form(a, jit_options=self.fast_jit)
#         L_form = cutfemx.fem.form(L, jit_options=self.fast_jit)

#         self.uh = self._solve_system(a_form, L_form, bcs_list, self.solver_primal, self.uh)
#         return self.uh


#     def adjoint_problem(self, u, level_set, dual_operator):
#         """Adjoint resolution aligned with the same scheme."""
#         self.uh = u
#         cutfemx.update(self.cut_data)

#         solid_cells = cutfemx.locate_entities(self.cut_data, "phi<0")
#         solid_rules = cutfemx.runtime_quadrature(self.cut_data, "phi<0", self.order)
#         dx_solid = ufl.Measure("dx", domain=self.mesh, subdomain_id=0,
#                                subdomain_data=[solid_cells, solid_rules], metadata={"quadrature_degree": self.order})
#         ghost_facets = cutfemx.ghost_penalty_facets(self.cut_data, "phi<0")
#         dS_ghost = ufl.Measure("dS", domain=self.mesh, subdomain_id=1,
#                                subdomain_data=ghost_facets)

#         p = ufl.TrialFunction(self.space_displacement)
#         v_adj = ufl.TestFunction(self.space_displacement)
#         gdim = self.mesh.geometry.dim

#         a_adj = ufl.inner(sigma_expr(p, self.lame_mu, self.lame_lambda, gdim), epsilon_expr(v_adj)) * dx_solid
#         if ghost_facets.size > 0:
#             n_facet = FacetNormal(self.mesh)
#             h_avg = avg(CellDiameter(self.mesh))
#             a_adj += (
#                 self.gamma * h_avg
#                 * ufl.inner(ufl.jump(ufl.grad(p), n_facet), ufl.jump(ufl.grad(v_adj), n_facet))
#                 * dS_ghost
#             )

#         L_adj = dual_operator
#         bcs_list = self.bc if isinstance(self.bc, list) else [self.bc]

#         a_form_adj = cutfemx.fem.form(a_adj, jit_options=self.fast_jit)

#         if isinstance(L_adj, fem.Form):
#             L_form_adj = fem.form(L_adj, jit_options=self.fast_jit)
#         else:
#             L_form_adj = cutfemx.fem.form(L_adj, jit_options=self.fast_jit)

#         self.ph = self._solve_system(a_form_adj, L_form_adj, bcs_list, self.solver_adjoint, self.ph)
#         return self.ph


#     def cutfem_solver(self, level_set, parameters, problem_topo=0):
#         self.level_set.x.array[:] = level_set.x.array
#         self.level_set.x.scatter_forward()

#         self.uh = self.primal_problem(level_set, parameters)

#         if parameters.cost_func == "compliance":
#             self.ph.x.array[:] = self.uh.x.array
#             self.ph.x.scatter_forward()
#         else:
#             if problem_topo.__class__.__name__ in ["VMLp_Problem", "AreaProblem"]:
#                 import utils.mechanics_tool as mechanics_tool
#                 vm_DG = mechanics_tool.project_von_mises(self.uh, self.lame_mu, self.lame_lambda, self.mesh, self.dxq)
#                 adjoint = problem_topo.dual_operator(
#                     self.uh, self.lame_mu, self.lame_lambda, parameters, self.mesh, self.dxq, vm_DG=vm_DG
#                 )
#             else:
#                 adjoint = problem_topo.dual_operator(
#                     self.uh, self.lame_mu, self.lame_lambda, parameters, self.mesh, self.dxq
#                 )
#             self.ph = self.adjoint_problem(self.uh, level_set, adjoint)

#         return self.uh, self.ph

#     def update_measures_and_quadratures(self, level_set, order=4):
#         pass


#     def __del__(self):
#         for attr in ("solver_primal", "solver_adjoint"):
#             solver = getattr(self, attr, None)
#             if solver is not None:
#                 try:
#                     solver.destroy()
#                 except Exception:
#                     pass


# def epsilon_expr(u):
#     return ufl.sym(ufl.grad(u))

# def sigma_expr(u, mu: float, lmbda: float, gdim: int):
#     eps = epsilon_expr(u)
#     return 2.0 * mu * eps + lmbda * ufl.tr(eps) * ufl.Identity(gdim)


# # Copyright (c) 2026 ONERA and MINES Paris, France
# #
# # All rights reserved.
# #
# # This file is part of OptiCut.
# #
# # Author(s)     : Amina El Bachari
# #
# # Migration note: updated to CutFEMx 0.2 (DOLFINx 0.11)
# # Restructured for performance: Persistent KSP solvers & Modular measures.

# import ufl
# from ufl import FacetNormal, Measure, CellDiameter, avg, jump

# import cutfemx
# import cutfemx.petsc as cf_petsc

# from dolfinx import fem, mesh, la
# from dolfinx.fem import petsc as dolfinx_petsc
# from dolfinx.mesh import meshtags, locate_entities, locate_entities_boundary

# from mpi4py import MPI
# import numpy as np

# from petsc4py import PETSc
# from petsc4py.PETSc import ScalarType

# from utils import mechanics_tool


# class CutFEMElasticSolver:
#     r"""CutFEM linear-elasticity solver optimized for TopOpt loop (CutFEMx 0.2)."""

#     def __init__(self, level_set, level_set_space, space_displacement, ds, bc,
#                  bc_velocity, parameters, problem_topo, shift):

#         self.level_set = fem.Function(level_set_space)
#         self.level_set.x.array[:] = level_set.x.array
#         self.level_set.x.scatter_forward()

#         self.mesh = self.level_set.function_space.mesh
#         self.space_displacement = space_displacement
#         self.cutFEM = parameters.cutFEM
#         self.parameters = parameters
#         self.problem_topo = problem_topo

#         lame_mu, lame_lambda = mechanics_tool.lame_compute(parameters.young_modulus,
#                                                            parameters.poisson)
#         self.V_ls = level_set_space
#         self.cost_func = parameters.cost_func

#         self.lame_mu = lame_mu
#         self.lame_lambda = lame_lambda
#         self.dim = self.mesh.topology.dim

#         self.bc_velocity = bc_velocity
#         self.shift = shift

#         # Fixed geometric objects
#         self.n = FacetNormal(self.mesh)
#         self.h = CellDiameter(self.mesh)
#         self.bc = bc
#         self.ds = Measure("ds", domain=self.mesh,
#                           subdomain_data=ds.subdomain_data() if hasattr(ds, 'subdomain_data') else ds)

#         # Solution functions
#         self.uh = fem.Function(self.space_displacement)
#         self.ph = fem.Function(self.space_displacement)

#         self.gamma_N = 1e3
#         self.gamma   = 1e-5 * (self.lame_mu + self.lame_lambda)
#         self.order   = 4
#         self.fast_jit = {"cffi_extra_compile_args": ["-O0", "-w"]}

#         # ------------------------------------------------------------------
#         # 1. Creation of KSP solvers only once (avoids memory leaks)
#         # ------------------------------------------------------------------
#         self.solver_primal = PETSc.KSP().create(self.mesh.comm)
#         self.solver_adjoint = PETSc.KSP().create(self.mesh.comm)
#         self._configure_ksp(self.solver_primal)
#         self._configure_ksp(self.solver_adjoint)

#         # ------------------------------------------------------------------
#         # 2. Initialization of cut data
#         # ------------------------------------------------------------------
#         self.cut_data = cutfemx.cut(self.level_set)
#         self._build_measures()


#     def _configure_ksp(self, solver):
#         """Solver configuration based on a working setup (CG + GAMG)."""
#         solver.setType("cg")
#         pc = solver.getPC()
#         pc.setType("gamg")  # Very robust for CutFEM
#         solver.setFromOptions()
#         solver.setTolerances(rtol=1e-6)


#     def update_measures_and_quadratures(self, level_set):
#         """Method called at each iteration to update the geometry."""
#         self.level_set.x.array[:] = level_set.x.array
#         self.level_set.x.scatter_forward()

#         # Update of the CutFEMx topology
#         cutfemx.update(self.cut_data)

#         # Re-construction of UFL measures
#         self._build_measures()


#     def _build_measures(self):
#         """Isolates the creation of quadratures and measures to clarify the code."""
#         solid_cells = cutfemx.locate_entities(self.cut_data, "phi<0")
#         solid_rules = cutfemx.runtime_quadrature(self.cut_data, "phi<0", self.order)

#         self.dx_solid = ufl.Measure("dx", domain=self.mesh, subdomain_id=0,
#                                     subdomain_data=[solid_cells, solid_rules], metadata={"quadrature_degree": self.order})
#         self.dxq = self.dx_solid  # Alias utilisé par main.py

#         ghost_facets = cutfemx.ghost_penalty_facets(self.cut_data, "phi<0")
#         self.dS_ghost = ufl.Measure("dS", domain=self.mesh, subdomain_id=1,
#                                     subdomain_data=ghost_facets)

#         self.n_facet = FacetNormal(self.mesh)
#         self.h_avg = avg(CellDiameter(self.mesh))


#     def _solve_system(self, a_form, L_form, bcs_list, solver_ksp, solution_func):
#         """Native PETSc assembly and resolution pipeline."""

#         # 1. Vector assembly (identical to the working version)
#         b_reg = cf_petsc.assemble_vector(L_form)
#         b_reg.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
#         dolfinx_petsc.set_bc(b_reg, bcs_list) # Strict application of BCs
#         b_reg.assemble()

#         # 2. Matrix assembly
#         A_reg = cf_petsc.assemble_matrix(a_form, bcs=bcs_list)
#         A_reg.assemble()

#         # (Note: deactivate_outside left commented because the run works without it)
#         # active = cutfemx.fem.active_domain(a_form)
#         # cf_petsc.deactivate_outside(A_reg, b_reg, active)
#         # A_reg.assemble()

#         # 3. Resolution (reuse of the existing KSP!)
#         solver_ksp.setOperators(A_reg)
#         solver_ksp.setUp()
#         solver_ksp.solve(b_reg, solution_func.x.petsc_vec)
#         solution_func.x.scatter_forward()

#         # 4. Vital memory cleanup
#         A_reg.destroy()
#         b_reg.destroy()

#         return solution_func


#     def primal_problem(self, level_set, parameters):
#         """Primal resolution with integrated geometric update."""

#         # 1. Update the Level-Set and measures (dx_solid, dS_ghost...)
#         self.update_measures_and_quadratures(level_set)

#         # 2. Definition of UFL functions
#         u = ufl.TrialFunction(self.space_displacement)
#         v = ufl.TestFunction(self.space_displacement)

#         # 3. Bilinear form (with mechanics_tool)
#         a = (
#             2.0 * self.lame_mu * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(v))
#             + self.lame_lambda * ufl.inner(ufl.nabla_div(u), ufl.nabla_div(v))
#         ) * self.dx_solid

#         if self.dS_ghost.subdomain_data().size > 0:
#             a += (
#                 self.gamma * self.h_avg
#                 * ufl.inner(ufl.jump(ufl.grad(u), self.n_facet), ufl.jump(ufl.grad(v), self.n_facet))
#                 * self.dS_ghost
#             )

#         # 4. Forme linéaire
#         L = ufl.dot(self.shift, v) * self.ds(2)
#         bcs_list = self.bc if isinstance(self.bc, list) else [self.bc]

#         # 5. Compilation and resolution
#         a_form = cutfemx.fem.form(a, jit_options=self.fast_jit)
#         L_form = cutfemx.fem.form(L, jit_options=self.fast_jit)

#         self.uh = self._solve_system(a_form, L_form, bcs_list, self.solver_primal, self.uh)
#         return self.uh

#     def adjoint_problem(self, u, dual_operator):
#         """Adjoint resolution using the mechanics_tool module."""
#         self.uh = u

#         p = ufl.TrialFunction(self.space_displacement)
#         v_adj = ufl.TestFunction(self.space_displacement)

#         # Bilinear form with physics centralized in mechanics_tool
#         a_adj = (
#             2.0 * self.lame_mu * ufl.inner(mechanics_tool.strain(p), mechanics_tool.strain(v_adj))
#             + self.lame_lambda * ufl.inner(ufl.nabla_div(p), ufl.nabla_div(v_adj))
#         ) * self.dx_solid

#         # Ajout Ghost Penalty
#         if self.dS_ghost.subdomain_data().size > 0:
#             a_adj += (
#                 self.gamma * self.h_avg
#                 * ufl.inner(ufl.jump(ufl.grad(p), self.n_facet), ufl.jump(ufl.grad(v_adj), self.n_facet))
#                 * self.dS_ghost
#             )

#         # Second membre Adjoint
#         L_adj = dual_operator
#         bcs_list = self.bc if isinstance(self.bc, list) else [self.bc]

#         # Compilation FFCx
#         a_form_adj = cutfemx.fem.form(a_adj, jit_options=self.fast_jit)

#         if isinstance(L_adj, fem.Form):
#             L_form_adj = fem.form(L_adj, jit_options=self.fast_jit)
#         else:
#             L_form_adj = cutfemx.fem.form(L_adj, jit_options=self.fast_jit)

#         self.ph = self._solve_system(a_form_adj, L_form_adj, bcs_list, self.solver_adjoint, self.ph)
#         return self.ph


#     def cutfem_solver(self, level_set, parameters, problem_topo=0):
#         """Orchestrator called by the optimization loop."""

#         # 1. Solves the primal (which will internally update the level_set and measures)
#         self.uh = self.primal_problem(level_set, parameters)

#         # 2. Solves the adjoint (uses the measures updated by step 1)
#         if parameters.cost_func == "compliance":
#             self.ph.x.array[:] = self.uh.x.array
#             self.ph.x.scatter_forward()
#         else:
#             if problem_topo.__class__.__name__ in ["VMLp_Problem", "AreaProblem"]:
#                 import utils.mechanics_tool as mechanics_tool
#                 vm_DG = mechanics_tool.project_von_mises(self.uh, self.lame_mu, self.lame_lambda, self.mesh, self.dxq)
#                 adjoint = problem_topo.dual_operator(
#                     self.uh, self.lame_mu, self.lame_lambda, parameters, self.mesh, self.dxq, vm_DG=vm_DG
#                 )
#             else:
#                 adjoint = problem_topo.dual_operator(
#                     self.uh, self.lame_mu, self.lame_lambda, parameters, self.mesh, self.dxq
#                 )
#             self.ph = self.adjoint_problem(self.uh, adjoint)

#         return self.uh, self.ph


#     def __del__(self):
#         """Ensures proper destruction of persistent KSP solvers."""
#         for attr in ("solver_primal", "solver_adjoint"):
#             solver = getattr(self, attr, None)
#             if solver is not None:
#                 try:
#                     solver.destroy()
#                 except Exception:
#                     pass


# Copyright (c) 2026 ONERA and MINES Paris, France
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari
#
# Migration note: updated to CutFEMx 0.2 (DOLFINx 0.11)
# Architecture: Persistent solvers, isolated UFL forms, modular measures.

import ufl
from ufl import FacetNormal, Measure, CellDiameter, avg, jump

import cutfemx
import cutfemx.petsc as cf_petsc

from dolfinx import fem, mesh, la
from dolfinx.fem import petsc as dolfinx_petsc
from dolfinx.mesh import meshtags, locate_entities, locate_entities_boundary

from mpi4py import MPI
import numpy as np

from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

from utils import mechanics_tool


class CutFEMElasticSolver:
    r"""CutFEM linear-elasticity solver optimized for TopOpt loop (CutFEMx 0.2)."""

    def __init__(
        self,
        level_set,
        level_set_space,
        space_displacement,
        ds,
        bc,
        bc_velocity,
        parameters,
        problem_topo,
        shift,
    ):

        self.level_set = fem.Function(level_set_space)
        self.level_set.x.array[:] = level_set.x.array
        self.level_set.x.scatter_forward()

        self.mesh = self.level_set.function_space.mesh
        self.space_displacement = space_displacement
        self.cutFEM = parameters.cutFEM
        self.parameters = parameters
        self.problem_topo = problem_topo

        lame_mu, lame_lambda = mechanics_tool.lame_compute(
            parameters.young_modulus, parameters.poisson
        )
        self.V_ls = level_set_space
        self.cost_func = parameters.cost_func

        self.lame_mu = lame_mu
        self.lame_lambda = lame_lambda
        self.dim = self.mesh.topology.dim

        self.bc_velocity = bc_velocity
        self.shift = shift

        # Fixed geometric objects
        self.n = FacetNormal(self.mesh)
        self.h = CellDiameter(self.mesh)
        self.bc = bc
        self.ds = Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=ds.subdomain_data() if hasattr(ds, "subdomain_data") else ds,
        )

        # Trial, test and solution functions (defined only once)
        self.u_trial = ufl.TrialFunction(self.space_displacement)
        self.v_test = ufl.TestFunction(self.space_displacement)
        self.uh = fem.Function(self.space_displacement)
        self.ph = fem.Function(self.space_displacement)

        self.gamma_N = 1e3
        self.gamma = 1e-5 * (self.lame_mu + self.lame_lambda)
        self.order = 4
        self.fast_jit = {"cffi_extra_compile_args": ["-O0", "-w"]}

        # Creation of KSP solvers only once (avoids memory leaks)
        self.solver_primal = PETSc.KSP().create(self.mesh.comm)
        self.solver_adjoint = PETSc.KSP().create(self.mesh.comm)
        self._configure_ksp(self.solver_primal)
        self._configure_ksp(self.solver_adjoint)

        # Initialization of cut data CutFEMx
        self.cut_data = cutfemx.cut(self.level_set)
        self._build_measures()

    def _configure_ksp(self, solver):
        """Configuration of the robust iterative PETSc solver for CutFEM."""
        solver.setType("cg")
        pc = solver.getPC()
        pc.setType("gamg")
        solver.setFromOptions()
        solver.setTolerances(rtol=1e-6)

    def update_measures_and_quadratures(self, level_set):
        """Updates the geometry and reconstructs the integration measures."""
        self.level_set.x.array[:] = level_set.x.array
        self.level_set.x.scatter_forward()

        cutfemx.update(self.cut_data)
        self._build_measures()

    def _build_measures(self):
        """Isolates the creation of quadratures and domains (phi < 0)."""
        solid_cells = cutfemx.locate_entities(self.cut_data, "phi<0")
        solid_rules = cutfemx.runtime_quadrature(self.cut_data, "phi<0", self.order)

        self.dx_solid = ufl.Measure(
            "dx",
            domain=self.mesh,
            subdomain_id=0,
            subdomain_data=[solid_cells, solid_rules],
        )
        self.dxq = self.dx_solid  # Alias utilisé par main.py

        self.ghost_facets = cutfemx.ghost_penalty_facets(self.cut_data, "phi<0")
        self.dS_ghost = ufl.Measure(
            "dS", domain=self.mesh, subdomain_id=1, subdomain_data=self.ghost_facets, metadata={"quadrature_degree": self.order}
        )

        self.n_facet = FacetNormal(self.mesh)
        self.h_avg = avg(CellDiameter(self.mesh))

    # ====================================================================
    # FORMULATION MATHÉMATIQUE (UFL) PURE
    # ====================================================================

    def _build_primal_forms(self):
        """Exclusively defines the physics of the primal problem."""
        u = self.u_trial
        v = self.v_test

        # Forme bilinéaire
        self.a_primal = (
            2.0
            * self.lame_mu
            * ufl.inner(mechanics_tool.strain(u), mechanics_tool.strain(v))
            + self.lame_lambda * ufl.inner(ufl.nabla_div(u), ufl.nabla_div(v))
        ) * self.dx_solid

        if self.ghost_facets.size > 0:
            self.a_primal += (
                self.gamma
                * self.h_avg
                * ufl.inner(
                    ufl.jump(ufl.grad(u), self.n_facet),
                    ufl.jump(ufl.grad(v), self.n_facet),
                )
                * self.dS_ghost
            )

        # Forme linéaire
        self.L_primal = ufl.dot(self.shift, v) * self.ds(2)

    def _build_adjoint_forms(self, dual_operator):
        """Exclusively defines the physics of the adjoint problem."""
        p = self.u_trial
        v_adj = self.v_test

        # Forme bilinéaire (identique au primal)
        self.a_adj = (
            2.0
            * self.lame_mu
            * ufl.inner(mechanics_tool.strain(p), mechanics_tool.strain(v_adj))
            + self.lame_lambda * ufl.inner(ufl.nabla_div(p), ufl.nabla_div(v_adj))
        ) * self.dx_solid

        if self.ghost_facets.size > 0:
            self.a_adj += (
                self.gamma
                * self.h_avg
                * ufl.inner(
                    ufl.jump(ufl.grad(p), self.n_facet),
                    ufl.jump(ufl.grad(v_adj), self.n_facet),
                )
                * self.dS_ghost
            )

        # Forme linéaire
        self.L_adj = dual_operator

    # ====================================================================
    # RESOLUTION (ASSEMBLY AND PETSc)
    # ====================================================================

    # def _solve_system(self, a_form, L_form, bcs_list, solver_ksp, solution_func):
    #     """CutFEMx assembly and PETSc KSP resolution pipeline."""
    #     b_reg = cf_petsc.assemble_vector(L_form)
    #     b_reg.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    #     dolfinx_petsc.set_bc(b_reg, bcs_list)
    #     b_reg.assemble()

    #     A_reg = cf_petsc.assemble_matrix(a_form, bcs=bcs_list)
    #     A_reg.assemble()

    #     solver_ksp.setOperators(A_reg)
    #     solver_ksp.setUp()
    #     solver_ksp.solve(b_reg, solution_func.x.petsc_vec)
    #     solution_func.x.scatter_forward()

    #     A_reg.destroy()
    #     b_reg.destroy()

    #     return solution_func

    def _solve_system(self, a_form, L_form, bcs_list, solver_ksp, solution_func):
        """CutFEMx assembly and 100% MPI-robust PETSc KSP resolution pipeline."""

        # 1. Assemblage du vecteur RHS
        b_reg = cf_petsc.assemble_vector(L_form)

        # --- CORRECTION ICI : Lifting spécifique à CutFEMx ---
        with b_reg.localForm() as b_local:
            # array_w allows writing to local memory for PETSc
            cutfemx.fem.apply_lifting(b_local.array_w, [a_form], [bcs_list])

        b_reg.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
        )

        # set_bc works in a standard way because bcs_list does not contain a UFL form
        dolfinx_petsc.set_bc(b_reg, bcs_list)

        # 2. Matrix assembly
        A_reg = cf_petsc.assemble_matrix(a_form, bcs=bcs_list)
        A_reg.assemble()

        # 3. MPI Deactivation (Avoids NaNs in the void)
        active = cutfemx.fem.active_domain(a_form)
        cf_petsc.deactivate_outside(A_reg, b_reg, active)
        A_reg.assemble()

        # 4. Resolution with existing KSP
        solver_ksp.setOperators(A_reg)
        solver_ksp.setUp()
        solver_ksp.solve(b_reg, solution_func.x.petsc_vec)

        # 5. Update of solution ghosts
        solution_func.x.scatter_forward()

        # 6. Vital memory cleanup
        A_reg.destroy()
        b_reg.destroy()

        return solution_func

    def primal_problem(self, level_set, parameters):
        """Orchestrateur du primal : Géométrie -> UFL -> FFCx -> Solve."""
        self.update_measures_and_quadratures(level_set)
        self._build_primal_forms()

        bcs_list = self.bc if isinstance(self.bc, list) else [self.bc]
        a_form = cutfemx.fem.form(self.a_primal, jit_options=self.fast_jit)
        L_form = cutfemx.fem.form(self.L_primal, jit_options=self.fast_jit)

        self.uh = self._solve_system(
            a_form, L_form, bcs_list, self.solver_primal, self.uh
        )
        return self.uh

    def adjoint_problem(self, u, dual_operator):
        """Adjoint orchestrator: UFL -> FFCx -> Solve."""
        self.uh = u
        self._build_adjoint_forms(dual_operator)

        bcs_list = self.bc if isinstance(self.bc, list) else [self.bc]
        a_form_adj = cutfemx.fem.form(self.a_adj, jit_options=self.fast_jit)

        if isinstance(self.L_adj, fem.Form):
            L_form_adj = fem.form(self.L_adj, jit_options=self.fast_jit)
        else:
            L_form_adj = cutfemx.fem.form(self.L_adj, jit_options=self.fast_jit)

        self.ph = self._solve_system(
            a_form_adj, L_form_adj, bcs_list, self.solver_adjoint, self.ph
        )
        return self.ph

    def cutfem_solver(self, level_set, parameters, problem_topo=0):
        """Main function called by the optimization loop."""
        self.uh = self.primal_problem(level_set, parameters)

        if parameters.cost_func == "compliance":
            self.ph.x.array[:] = self.uh.x.array
            self.ph.x.scatter_forward()
        else:
            if problem_topo.__class__.__name__ in ["VMLp_Problem", "AreaProblem"]:
                import utils.mechanics_tool as mechanics_tool

                vm_DG = mechanics_tool.project_von_mises(
                    self.uh, self.lame_mu, self.lame_lambda, self.mesh, self.dxq
                )
                adjoint = problem_topo.dual_operator(
                    self.uh,
                    self.lame_mu,
                    self.lame_lambda,
                    parameters,
                    self.mesh,
                    self.dxq,
                    vm_DG=vm_DG,
                )
            else:
                adjoint = problem_topo.dual_operator(
                    self.uh,
                    self.lame_mu,
                    self.lame_lambda,
                    parameters,
                    self.mesh,
                    self.dxq,
                )
            self.ph = self.adjoint_problem(self.uh, adjoint)

        return self.uh, self.ph

    def __del__(self):
        """Cleanup of PETSc solvers."""
        for attr in ("solver_primal", "solver_adjoint"):
            solver = getattr(self, attr, None)
            if solver is not None:
                try:
                    solver.destroy()
                except Exception:
                    pass
