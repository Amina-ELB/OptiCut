# Copyright (c) 2025 ONERA and MINES Paris, France
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari
#
# Migration note: updated to CutFEMx 0.2 (DOLFINx 0.11)
#   - Reinitialization: Predictor-Corrector (PC) scheme using CutFEM Nitsche forms
#   - Advection class: standard FEM scheme (no CutFEM)


import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
import matplotlib.pyplot as plt
from ufl import ds, dx, grad, inner, tr, dS

from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

from typing import TYPE_CHECKING

from dolfinx.fem import (
    Constant,
    Function,
    FunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, meshtags
from petsc4py.PETSc import ScalarType
from ufl import (
    FacetNormal,
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    div,
    dot,
    dx,
    grad,
    inner,
    lhs,
    rhs,
    dc,
    FacetNormal,
    CellDiameter,
    dot,
    avg,
    jump,
)
from dolfinx.io import XDMFFile

# CutFEMx 0.2 imports
import cutfemx
import cutfemx.petsc as cf_petsc

from utils import mechanics_tool

############################################
# HJ Reinitialization
############################################


class LevelSet:
    def __init__(self, level_set, space):
        """
        LevelSet class to define a level-set function.

        Parameters
        ----------
        level_set : fem.Function
            Initial level-set function.
        space : fem.FunctionSpace
            Function space of the level-set.
        """
        self.level_set = fem.Function(space, name="phi")
        self.level_set.x.array[:] = level_set.x.array
        self.space = space

    def set_level_set(self, level_set):
        """Update the level-set function."""
        self.level_set.x.array[:] = level_set.x.array
        self.level_set.x.scatter_forward()


class Advection(LevelSet):
    """
    Class to perform level-set advection using a standard FEM scheme (no CutFEM).
    Inherits from LevelSet.
    """

    def __init__(self, level_set, V_ls, dt=1e-3):
        """
        Initialize the Advection solver for a level-set.

        Parameters
        ----------
        level_set : fem.Function
            Initial level-set function.
        V_ls : fem.FunctionSpace
            Function space of the level-set.
        dt : float, optional
            Time step for advection (default: 1e-3).
        """
        super().__init__(level_set, V_ls)

        self.dt = dt
        self.V_ls = V_ls
        self.mesh = V_ls.mesh

        self.phi_n = ufl.TrialFunction(self.V_ls)
        self.phi_test = ufl.TestFunction(self.V_ls)

        self.velocity_field = fem.Function(self.V_ls)
        self.velocity_field.x.array[:] = 1e-3

        self.h = ufl.CellDiameter(self.mesh)
        self.dS = ufl.dS
        self.n = ufl.FacetNormal(self.mesh)
        self.const = 1e-3

        self.fast_jit = {"cffi_extra_compile_args": ["-O0", "-w"]}
        self.dt_const = fem.Constant(self.mesh, PETSc.ScalarType(self.dt))

        a_adv = ufl.dot(self.phi_n, self.phi_test) * ufl.dx
        a_adv += (
            avg(self.const)
            * avg(self.h) ** 3
            * ufl.inner(
                ufl.jump(ufl.grad(self.phi_n), self.n),
                ufl.jump(ufl.grad(self.phi_test), self.n),
            )
            * self.dS
        )
        a_std_adv = fem.form(a_adv, jit_options=self.fast_jit)
        self.A_adv = fem.petsc.assemble_matrix(a_std_adv)
        self.A_adv.assemble()

        L_adv = (
            ufl.dot(self.level_set, self.phi_test) * ufl.dx
            + ufl.dot(
                -self.dt_const
                * self.velocity_field
                * ufl.sqrt(
                    ufl.inner(ufl.grad(self.level_set), ufl.grad(self.level_set))
                ),
                self.phi_test,
            )
            * ufl.dx
        )
        self.L_std_adv = fem.form(L_adv, jit_options=self.fast_jit)
        self.b_adv = fem.petsc.create_vector(self.V_ls)
        fem.petsc.assemble_vector(self.b_adv, self.L_std_adv)
        self.b_adv.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        self.b_adv.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        self.solver_adv = PETSc.KSP().create(self.mesh.comm)
        self.solver_adv.setOperators(self.A_adv)
        self.solver_adv.setType(PETSc.KSP.Type.PREONLY)
        pc = self.solver_adv.getPC()
        pc.setType(PETSc.PC.Type.LU)
        if self.mesh.comm.size > 1:
            pc.setFactorSolverType("mumps")
        self.solver_adv.setUp()

        self.sol_adv = fem.Function(self.V_ls)

    def cut_fem_adv(self, velocity_field, dt):
        r"""Perform one advection step of the level-set function."""
        self.velocity_field.x.array[:] = velocity_field.x.array
        self.velocity_field.x.scatter_forward()

        self.dt = dt
        self.dt_const.value = dt

        with self.b_adv.localForm() as loc:
            loc.set(0.0)
        fem.petsc.assemble_vector(self.b_adv, self.L_std_adv)
        self.b_adv.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        self.b_adv.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        self.solver_adv.solve(self.b_adv, self.sol_adv.x.petsc_vec)
        self.sol_adv.x.scatter_forward()

        self.level_set.x.array[:] = self.sol_adv.x.array
        self.level_set.x.scatter_forward()

        return self.sol_adv


############################################
# Reinitialization — Predictor-Corrector (PC) Scheme
############################################


class Reinitialization(LevelSet):
    r"""Reinitialization class using the Predictor-Corrector (PC) scheme with CutFEM Nitsche penalty.

    Predictor problem:
    Find \phi_p \in V such that:
        -\Delta \phi_p = \psi(x) in D
        \phi_p = 0 on \Gamma
        \nabla \phi_p \cdot n = \psi(x) on \partial D

    Corrector problem (step_reinit iterations):
    Find \phi^{n+1} \in V such that:
        \nabla \cdot (\nabla \phi^{n+1} - n_K) = 0 in D
        \phi^{n+1} = 0 on \Gamma
        (\nabla \phi^{n+1} - n_K) \cdot n = 0 on \partial D
    """

    def __init__(self, level_set, V_ls, l):
        """
        Parameters
        ----------
        level_set : fem.Function
            Initial level-set function.
        V_ls : fem.FunctionSpace
            Function space of the level-set.
        l : float
            Characteristic length parameter.
        """
        super().__init__(level_set, V_ls)
        self.mesh = level_set.function_space.mesh
        self.l = l
        self.V_ls = V_ls
        self.dim = self.mesh.topology.dim
        self.order = 1
        self.gamma_r = 1e4
        self.eps = 1e-6

    def reinitializationPC(self, level_set, step_reinit=5):
        r"""Compute the Predictor-Corrector reinitialization.

        :param fem.Function level_set: The level-set function to reinitialize.
        :param int step_reinit: Number of correction iterations.
        :returns: Reinitialized level-set function.
        """
        self.level_set.x.array[:] = level_set.x.array
        self.level_set.x.scatter_forward()

        msh = self.mesh
        u_r = ufl.TrialFunction(self.V_ls)
        v_r = ufl.TestFunction(self.V_ls)
        h_r = ufl.CellDiameter(msh)

        # 1. Cut the interface from the current level set
        self.level_set.name = "phi"
        cut_data = cutfemx.cut(self.level_set)
        interface_quad = cutfemx.runtime_quadrature(cut_data, "phi=0", self.order)

        # Measures
        dx_full = ufl.Measure("dx", domain=msh)
        ds_bnd = ufl.Measure("ds", domain=msh)
        dsq = ufl.Measure("dx", domain=msh, subdomain_data=interface_quad)

        # 2. Bilinear form: Operator split (Poisson on D + Nitsche penalty on Gamma)
        # Standard Poisson part on whole domain D
        a_poisson_ufl = ufl.inner(ufl.grad(u_r), ufl.grad(v_r)) * dx_full
        a_poisson = fem.form(a_poisson_ufl)
        A_poisson = fem.petsc.assemble_matrix(a_poisson)
        A_poisson.assemble()

        # Nitsche interface part on Gamma
        norm_ls = ufl.sqrt(ufl.inner(ufl.grad(self.level_set), ufl.grad(self.level_set)) + 1e-12)
        n_K = ufl.grad(self.level_set) / norm_ls

        a_nitsche_ufl = (
            - ufl.dot(ufl.grad(u_r), n_K) * v_r * dsq
            - ufl.dot(ufl.grad(v_r), n_K) * u_r * dsq
            + (self.gamma_r / h_r) * u_r * v_r * dsq
        )
        a_nitsche = cutfemx.fem.form(a_nitsche_ufl)
        A_nitsche = cf_petsc.assemble_matrix(a_nitsche)
        A_nitsche.assemble()

        # Combine operators: A = A_poisson + A_nitsche
        A = A_poisson.copy()
        A.axpy(1.0, A_nitsche, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        A.assemble()

        A_poisson.destroy()
        A_nitsche.destroy()

        # Solver KSP setup with matrix A
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        if msh.comm.size > 1:
            pc.setFactorSolverType("mumps")
        ksp.setUp()

        # 3. Predictor step
        sign_ls = self.level_set / ufl.sqrt(self.level_set**2 + self.eps**2)
        L_predict_ufl = (
            ufl.inner(self.l**2 * sign_ls, v_r) * dx_full
            + ufl.dot(sign_ls, v_r) * ds_bnd
        )
        L_predict_form = fem.form(L_predict_ufl)
        b_predict = fem.petsc.create_vector(self.V_ls)
        fem.petsc.assemble_vector(b_predict, L_predict_form)
        b_predict.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        b_predict.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Solve Predictor
        phi_sol = fem.Function(self.V_ls)
        ksp.solve(b_predict, phi_sol.x.petsc_vec)
        phi_sol.x.scatter_forward()
        b_predict.destroy()

        self.level_set.x.array[:] = phi_sol.x.array
        self.level_set.x.scatter_forward()

        # 4. Corrector steps
        for _ in range(step_reinit):
            norm_ls_c = ufl.sqrt(ufl.inner(ufl.grad(self.level_set), ufl.grad(self.level_set)) + 1e-12)
            n_K_c = ufl.grad(self.level_set) / norm_ls_c
            L_correct_ufl = ufl.inner(n_K_c, ufl.grad(v_r)) * dx_full
            L_correct_form = fem.form(L_correct_ufl)
            b_correct = fem.petsc.create_vector(self.V_ls)
            fem.petsc.assemble_vector(b_correct, L_correct_form)
            b_correct.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            b_correct.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            ksp.solve(b_correct, phi_sol.x.petsc_vec)
            phi_sol.x.scatter_forward()
            b_correct.destroy()

            self.level_set.x.array[:] = phi_sol.x.array
            self.level_set.x.scatter_forward()

        # Cleanup
        A.destroy()
        ksp.destroy()

        return self.level_set

    def reinitializationPC_inplace(self, level_set, step_reinit=5):
        """Reinitialize in-place (modifies level_set directly)."""
        res = self.reinitializationPC(level_set, step_reinit)
        level_set.x.array[:] = res.x.array
        level_set.x.scatter_forward()
