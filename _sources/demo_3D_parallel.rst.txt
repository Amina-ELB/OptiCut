.. _demo3D:

High-Performance 3D Compliance Minimization
=============================================
A major advantage of OptiCut is its native integration with MPI (Message Passing Interface) and PETSc, inherited from the `dolfinx` backend. This enables structural optimization on massive 3D models using parallel high-performance computing (HPC) clusters.

In this tutorial, we demonstrate how to minimize compliance on a 3D cantilever beam and solve the elasticity PDEs across multiple processes using scalable Algebraic Multigrid (AMG) solvers.

Problem definition
---------------------
The continuous shape optimization problem remains identical to the 2D compliance minimization problem. The goal is to minimize the 3D compliance functional subject to a volume constraint:

.. math::
    \begin{aligned}\begin{cases}
    \underset{\Omega\in\mathcal{O}}{\min}\int_{\Omega}\left(\mu\varepsilon(u):\varepsilon(u)+\frac{\lambda}{2}(\nabla\cdot u)^2\right)\text{ }dx\\
    C(\Omega) = \int_{\Omega}dx - \overline{V} = 0
    \end{cases}\end{aligned}


Mesh & MPI Partitioning
~~~~~~~~~~~~~~~~~~~~~~~~
When `mpirun` is invoked, OptiCut automatically partitions the 3D mesh (loaded via Gmsh or generated internally) into sub-domains using standard graph partitioners (e.g., ParMETIS). Each MPI process holds only a fraction of the elements and degrees of freedom (DOFs), significantly reducing per-process memory footprints.

Iterative AMG Solvers
~~~~~~~~~~~~~~~~~~~~~~~~
For massive 3D models, standard LU direct solvers become a memory bottleneck. OptiCut seamlessly switches to iterative solvers when configured. By specifying `linear_solver amg` in the parameters, the code employs the Conjugate Gradient (CG) method preconditioned by Geometric-Algebraic Multigrid (GAMG), achieving near-linear scaling for 3D elasticity.

Implementation
--------------

We use the parameter file `param_compliance_3D.txt`. The relevant mechanical and numerical constants are:

.. code-block:: text
	:linenos:

	cost_func compliance
	mesh_case 3D
	lx 2
	ly 1
	lz 1
	h 0.05
	linear_solver amg
	target_constraint 0.4
	young_modulus 210000000000

Notice that the flag `linear_solver amg` instructs OptiCut to avoid LU factorization.

**Solver Instantiation Snippet**
Behind the scenes, OptiCut configures the robust `GAMG` preconditioner:

.. code-block:: python
	:linenos:
	
	if parameters.linear_solver == "amg":
	    self.primal_solver.setType(PETSc.KSP.Type.CG)
	    self.primal_solver.getPC().setType("gamg")
	    self.primal_solver.setTolerances(rtol=1e-6)


Running in Parallel
-------------------

To launch the 3D optimization across 8 MPI cores, simply use the `mpirun` launcher:

.. code-block:: bash

    mpirun -n 8 python3 main.py parameters/param_compliance_3D.txt

OptiCut's advection solvers (`AdvectionSolver`), level-set routines, and PETSc sparse matrix assemblies inherently communicate the required ghost-node updates across partition boundaries, completely abstracting the parallel complexity from the user script.

Results and Performance
-----------------------

After convergence, the XDMF output files can be stitched and viewed in Paraview. 
To visualize the 3D results:
1. Open **ParaView**.
2. Go to `File > Open` and select `src/res/results.xdmf`.
3. Click **Apply**. 
4. Select the `ls_func` or `disp` variables to visualize the solid structure (e.g., using a Contour filter at `phi=0`).

The optimal topology effectively forms a 3D truss structure bridging the clamped face to the loaded region, minimizing material usage while maintaining strict rigidity.

**Performance and HPC Scaling Note:** 

OptiCut is highly optimized for HPC environments. However, scaling efficiency depends strictly on the *Degrees of Freedom (DOF) per core* ratio. 
As a rule of thumb in parallel finite elements, one needs at least **50,000 to 100,000 DOFs per MPI process** to overcome communication overhead. 

To allow reviewers and users to quickly test the parallel execution on a standard laptop, this tutorial uses a deliberately coarse mesh (`h=0.1`, yielding roughly 15,000 DOFs). For a complete optimization loop (200 iterations) using the CG solver, typical execution times are:

- **1 MPI process:** 367.17 seconds (6.12 minutes)
- **2 MPI processes:** 328.68 seconds (5.48 minutes)
- **4 MPI processes:** 310.56 seconds (5.18 minutes)

While there is a visible speedup, the scaling is not linear. This is a well-known HPC phenomenon: because the local problem size per core is so small, MPI network and memory communication overheads begin to rival the actual computation time. 

To truly leverage the parallel AMG solver and observe massive speedups, the mesh size must be drastically reduced (e.g., `h=0.03`), pushing the DOFs into the millions. Under such heavy workloads, a single core would simply run out of memory (OOM), whereas OptiCut's MPI-partitioned solvers will distribute the memory effectively and solve the problem.

