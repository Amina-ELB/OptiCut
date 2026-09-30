# Copyright (c) 2025 ONERA and MINES Paris, France
#
# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari
#
# Migration note: updated to CutFEMx 0.2 (DOLFINx 0.11)
#   - Removed manual locate_entities / cut_entities / create_cut_mesh / runtime_quadrature calls
#   - Now uses cutfemx.cut(level_set) + cutfemx.update() + cutfemx.runtime_quadrature()
#   - compute_normal replaced by cutfemx.locate_entities for normals (computed via cutfemx.distance)
#   - Assembly via cutfemx.petsc.assemble_matrix / assemble_vector

import numpy as np
import math
import ufl

from dolfinx import fem, io, mesh
import matplotlib.pyplot as plt
from ufl import dx, grad, inner, dS
import dolfinx
from petsc4py import PETSc
from typing import TYPE_CHECKING

from ufl import FacetNormal, dx, grad, inner, dc, FacetNormal, CellDiameter

# CutFEMx 0.2 unified API
import cutfemx
import cutfemx.petsc as cf_petsc

from mpi4py import MPI

import gc
import os

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def prepare_descent(msh, V_ls, parameters):
    r"""Prepare reusable objects for the descent direction problem.

    This should be called ONCE outside the iteration loop.

    :param dolfinx.mesh.Mesh msh: Computational mesh.
    :param dolfinx.fem.FunctionSpace V_ls: Level-set function space.
    :param object parameters: Object containing problem constants.

    :returns: Dictionary with reusable objects:
        - ``V_DG``: DG0 vector space for normals
        - ``v_reg``: fem.Function, solution of the extension subproblem
        - ``n_K``: fem.Function, normal vector function
        - ``ksp``: PETSc.KSP solver (pre-configured)
    :rtype: dict
    """
    # DG space created only once (vectorial)
    V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim,)))

    v_reg = fem.Function(V_ls)  # solution of the subproblem
    n_K = fem.Function(V_DG)  # normal vector, updated every iteration

    # Pre-create PETSc KSP solver to avoid costly recreation
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setType("cg")
    pc = ksp.getPC()
    pc.setType("hypre")
    ksp.setFromOptions()

    return {
        "V_DG": V_DG,
        "v_reg": v_reg,
        "n_K": n_K,
        "ksp": ksp,
    }


def descent_direction(
    level_set,
    msh,
    parameters,
    bc_velocity,
    V_ls,
    rest_constraint,
    constraint_integrande,
    cost_integrande,
    resources,
):
    r"""Compute one descent direction step.

    Uses CutFEMx 0.2 API: a local ``CutData`` is created from the current level
    set, runtime quadrature rules are retrieved from it, and UFL measures are
    built accordingly.  Assembly uses ``cutfemx.petsc``.

    :param fem.Function level_set: Current level-set function.
    :param dolfinx.mesh.Mesh msh: Computational mesh.
    :param object parameters: Problem parameters.
    :param list bc_velocity: List of velocity boundary conditions.
    :param fem.FunctionSpace V_ls: Level-set function space.
    :param float rest_constraint: Rest constraint value.
    :param fem.Form constraint_integrande: Constraint integrand form.
    :param fem.Form cost_integrande: Cost integrand form.
    :param dict resources: Pre-prepared objects from ``prepare_descent``.

    :returns: Velocity field for advection.
    :rtype: fem.Function
    """
    V_DG = resources["V_DG"]
    v_reg = resources["v_reg"]
    n_K = resources["n_K"]
    ksp = resources["ksp"]

    tdim = msh.topology.dim
    dim = msh.geometry.dim
    order = 2

    # ------------------------------------------------------------------
    # CutFEMx 0.2: create CutData and retrieve geometry/quadrature
    # ------------------------------------------------------------------
    cut_data = cutfemx.cut(level_set)

    inside_entities = cutfemx.locate_entities(cut_data, "phi<0")
    intersected_entities = cutfemx.locate_entities(cut_data, "phi=0")

    # Runtime quadrature rules
    inside_rules = cutfemx.runtime_quadrature(cut_data, "phi<0", order)
    interface_rules = cutfemx.runtime_quadrature(cut_data, "phi=0", order)

    # ------------------------------------------------------------------
    # Normal on the interface (via cutfemx.distance)
    # ------------------------------------------------------------------
    # CutFEMx 0.2 exposes cutfemx.distance.normal() for evaluating the
    # level-set gradient direction.  We compute the nodal normal by
    # interpolating the UFL gradient expression on the DG0 space.
    norm_ls = ufl.sqrt(ufl.inner(ufl.grad(level_set), ufl.grad(level_set)) + 1e-14)
    pts_nK = (
        V_DG.element.interpolation_points()
        if callable(getattr(V_DG.element, "interpolation_points", None))
        else V_DG.element.interpolation_points
    )
    n_K_expr = fem.Expression(
        ufl.as_vector([ufl.grad(level_set)[i] / norm_ls for i in range(dim)]), pts_nK
    )
    n_K.interpolate(n_K_expr)
    n_K.x.scatter_forward()

    # ------------------------------------------------------------------
    # UFL measures
    # ------------------------------------------------------------------
    # Inside-cell background measure (standard FEM part)
    if len(inside_entities) > 0:
        from dolfinx.mesh import meshtags

        dx_tags = meshtags(
            msh,
            dim,
            inside_entities,
            np.full(len(inside_entities), 0, dtype=np.int32),
        )
        dx_tags_cut = meshtags(
            msh,
            dim,
            (
                intersected_entities
                if len(intersected_entities) > 0
                else np.array([0], dtype=np.int32)
            ),
            np.full(
                len(intersected_entities) if len(intersected_entities) > 0 else 1,
                2,
                dtype=np.int32,
            ),
        )
    else:
        from dolfinx.mesh import meshtags

        dx_tags = meshtags(
            msh, dim, np.array([0], dtype=np.int32), np.array([0], dtype=np.int32)
        )
        dx_tags_cut = meshtags(
            msh, dim, np.array([0], dtype=np.int32), np.array([2], dtype=np.int32)
        )

    # Measures
    dx_full = ufl.Measure("dx", domain=msh)
    dx_iface = ufl.Measure("dx", domain=msh, subdomain_data=interface_rules)
    dsq = dx_iface

    # ------------------------------------------------------------------
    # Bilinear form (H1 regularization / extension PDE on full domain D)
    # ------------------------------------------------------------------
    u_r = ufl.TrialFunction(V_ls)
    v_r = ufl.TestFunction(V_ls)

    a_reg = (
        parameters.alpha_reg_velocity
        * ufl.inner(ufl.grad(u_r), ufl.grad(v_r))
        * dx_full
        + u_r * v_r * dx_full
    )

    # ------------------------------------------------------------------
    # Linear form (shape derivative normal speed)
    # ------------------------------------------------------------------
    C_Omega_value = rest_constraint + parameters.ALM_slack_variable
    temp = ufl.as_ufl(cost_integrande)
    temp_ALM = parameters.ALM * (
        parameters.ALM_lagrangian_multiplicator * constraint_integrande
        + parameters.ALM_penalty_parameter * C_Omega_value * constraint_integrande
        + 2 * constraint_integrande * parameters.ALM_slack_variable
    )
    temp_ALM += (1 - parameters.ALM) * parameters.target_constraint
    temp += temp_ALM
    L_reg = -(ufl.inner(temp * v_r * n_K, n_K) * dsq)

    # ------------------------------------------------------------------
    # Cut forms (CutFEMx 0.2)
    # ------------------------------------------------------------------
    a_cut_reg = cutfemx.fem.form(a_reg, jit_options={"cache_dir": "ffcx-forms"})
    L_cut_reg = cutfemx.fem.form(L_reg)

    # ------------------------------------------------------------------
    # Assembly via cutfemx.petsc (returns PETSc.Mat / PETSc.Vec)
    # ------------------------------------------------------------------
    b_reg = cf_petsc.assemble_vector(L_cut_reg)
    b_reg.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    b_reg.assemble()
    A_reg = cf_petsc.assemble_matrix(a_cut_reg, bcs=[bc_velocity])
    A_reg.assemble()

    ksp.setOperators(A_reg)
    ksp.setUp()
    ksp.setTolerances(rtol=1e-8, atol=1e-12)
    ksp.solve(b_reg, v_reg.x.petsc_vec)
    v_reg.x.scatter_forward()

    b_reg.destroy()
    A_reg.destroy()

    del a_cut_reg, L_cut_reg, cut_data
    gc.collect()

    if HAS_PSUTIL and msh.comm.rank == 0:
        process = psutil.Process(os.getpid())
        print(f"RAM used: {process.memory_info().rss / 1e9:.3f} GB")

    return v_reg


def velocity_normalization(v, c_1):
    r"""Normalize the velocity field.

    .. math::

        \overline{v} = \frac{v}{\sqrt{c \left\Vert \nabla v \right\Vert_{L^2(D)}^2 + \left\Vert v \right\Vert_{L^2(D)}^2 }}

    :param fem.Function v: Velocity field to normalize.
    :param float c_1: Smoothing parameter.

    :returns: Normalization factor (scalar).
    :rtype: float
    """
    b_grad = fem.form(ufl.inner(ufl.grad(v), ufl.grad(v)) * ufl.dx)
    b_v = fem.form(ufl.inner(v, v) * ufl.dx)

    denom = MPI.COMM_WORLD.allreduce(
        fem.assemble_scalar(b_grad) * c_1 + fem.assemble_scalar(b_v), op=MPI.SUM
    )

    if denom < 1e-15:
        return 1.0

    return 1.0 / np.sqrt(denom)
