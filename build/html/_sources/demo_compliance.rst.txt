.. _demo:

Compliance minimization
=========================================
Compliance minimization is a fundamental problem in structural topology optimization. The objective is to optimally distribute a limited amount of material within a given design domain to maximize the overall stiffness (i.e., minimize the compliance) of the structure under prescribed loads.

Running the Tutorial
--------------------

To reproduce this example, open a terminal, navigate to the `src` directory of the OptiCut repository, and execute the main Python script passing the corresponding parameter file:

.. code-block:: bash

    cd src
    python3 main.py parameters/param_compliance.txt

Upon completion, the optimization history is exported in XDMF format. To visualize the evolution of the structure and the cost function:

1. Open **ParaView**.

2. Go to `File > Open` and select the generated file `src/res/results.xdmf`.

3. Click **Apply** in the Properties panel. You can then use the time-player controls at the top to animate the optimization iterations.

Problem definition
---------------------

We seek the optimal shape, :math:`\widetilde{\Omega}\subset D`, that minimizes the compliance for a linear elastic material subject to Dirichlet and Neumann boundary conditions. 
We impose a target area (or volume) on the structure, which introduces an equality constraint.
The continuous optimization problem is formally defined as:

.. math::

    \begin{aligned}\begin{cases}
    \underset{\Omega\in\mathcal{O}}{\min}J(\Omega) & \!\!\!\!=\underset{\Omega\in\mathcal{O}}{\min}\int_{\Omega}\left(\mu\varepsilon(u):\varepsilon(u)+\frac{\lambda}{2}(\nabla\cdot u)^2\right)\text{ }dx\\
    C(\Omega) & \!\!\!\!=0\\
    a\left(u,v\right) & \!\!\!\!=l\left(v\right)
    \end{cases}\end{aligned}

where :math:`u` is the displacement field, solution to the linear elasticity weak form. The volume constraint is defined as:

.. math::

    C(\Omega)=\int_{\Omega}dx - \overline{V}

with :math:`\overline{V}` representing the target volume.  

Shape derivative 
~~~~~~~~~~~~~~~~~~~~~~~~

Using Céa's method (or the Lagrangian approach), the shape derivative of the compliance minimization problem is given by:

.. math::

    \begin{aligned}
    J'(\Omega)(\theta) &= -\int_{\partial\Omega}\theta\cdot n\left[ \mu\varepsilon(u_{\Omega}):\varepsilon(u_{\Omega})+\frac{\lambda}{2}(\nabla\cdot u_{\Omega})^2\right]\text{ }ds
    \end{aligned}
		
where :math:`n` is the outward unit normal to the boundary :math:`\partial\Omega` and :math:`\theta` is the advection velocity field.

To rigorously enforce the volume constraint during the optimization process, we employ the Augmented Lagrangian Method (ALM) (see :ref:`ALM`).
First, we define the modified augmented Lagrangian functional as: 

.. math::

    \begin{aligned}
    \mathcal{J}(\Omega) &= J(\Omega) +\lambda_{ALM} C(\Omega)+\frac{\mu_{ALM}}{2} C^{2}(\Omega).
    \end{aligned}

The shape derivative of this new functional yields:

.. math::

    \begin{aligned}
    \mathcal{J}'(\Omega)(\theta) &= J'(\Omega)(\theta) +\lambda_{ALM} C'(\Omega)(\theta) +\mu_{ALM} C(\Omega)C'(\Omega)(\theta).
    \end{aligned}
		
Using the standard shape optimization framework, the descent direction (normal boundary velocity) is directly deduced as:

.. math::

    v(u_{\Omega}) = 2\mu\varepsilon(u_{\Omega}):\varepsilon(u_{\Omega})+\lambda(\nabla\cdot u_{\Omega})^2 + \lambda_{ALM} + \mu_{ALM} C(\Omega). 

Algorithm
--------------------

The overall optimization process follows a robust iterative scheme combining finite element analysis and level-set advection:

.. code-block:: text

    BEGIN
        Initialize function spaces and the Level-Set function φ
        uh ← Solve Primal problem (linear elasticity): a(u,v) = l(v)
        
        WHILE || J(Ω_{n+1}) - J(Ω_n) || > tol :
            n ← n + 1
            λ_ALM, μ_ALM ← Update ALM parameters
            v ← Compute descent direction on Γ
            v_ext ← Extend and regularize velocity across the domain D (Riesz)
            v_reg ← Normalize the extended velocity field
            
            WHILE adv_NAN ≠ 1 :
                φ_temp ← Advect the level-set function (Hamilton-Jacobi)
                φ_temp ← Periodic Reinitialization of the level-set
                uh ← Solve Primal problem
                dt, j_max, adv_NAN ← Update CFL parameters
                
            φ ← φ_temp
    END

Application 
--------------

We investigate the case of an embedded steel beam measuring :math:`1\text{m}\times 2\text{m}`, subjected to a uniformly distributed tensile load at :math:`\pm 0.5\text{m}` with a magnitude of :math:`g=-0.1 e_{y}\text{ GPa}`.
The numerical parameters characterizing the mechanical model are summarized in :numref:`paramMeca`.
The domain :math:`\Omega \subset D` is initialized as illustrated in :numref:`compliancedomain`, and discretized using a :math:`100\times200` finite element grid.
The optimization aims to reach a target area of :math:`1.2 \text{ m}^{2}` starting from an initial area of :math:`1.64\text{ m}^{2}`. 
The optimal shape obtained is shown in :ref:`finalresCutFEM` (using the CutFEM approach) and in :ref:`finalresErsatz` (using the Ersatz material method). 

.. container:: images-row

	..  container:: centered-figure

		.. _compliancedomain:

		.. figure:: images/demo_compliance/domain.png
			:align: center
			:width: 100%

			Initialization of :math:`\Omega\subset\text{D}`

	..  container:: centered-figure

		.. _mesh:   
		
		.. figure:: images/demo_compliance/mesh.png 
			:width: 100%
			:align: center

			Initialization of the mesh.

.. _paramMeca:

.. table:: Mechanical parameters
		:align: center

		+--------------------+------------+------------+
		| **Parameter**      | **Value**  | **Unit**   | 
		+====================+============+============+
		| E (Young Modulus)  | 210        |  **GPa**   | 
		+--------------------+------------+------------+
		| :math:`\nu`        | 0.33       |            |
		+--------------------+------------+------------+
		| Strength           | 0.1        | **GPa**    |
		+--------------------+------------+------------+

Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below, we demonstrate how OptiCut elegantly solves this problem using its modern object-oriented architecture.

**1. Loading Parameters and Mesh**

First, we load the simulation parameters from the text file and generate the corresponding mesh.

.. code-block:: python
	:linenos:
	
	from utils.config_utils import load_parameters
	from utils.ls_utils import load_mesh
	from utils.spaces_utils import init_function_spaces
	
	parameters = load_parameters(use_file=1, filename="param_compliance.txt")
	msh = load_mesh(parameters.mesh_case, parameters, mesh_folder="mesh")
	msh.topology.create_connectivity(msh.topology.dim, msh.topology.dim-1)

**2. Function Spaces and Level Set Initialization**

We allocate the required finite element function spaces for the level set, the displacement field, and the integration measures.

.. code-block:: python
	:linenos:

	spaces = init_function_spaces(msh)
	V, V_ls, Q, V_DG = spaces["V"], spaces["V_ls"], spaces["Q"], spaces["V_DG"]

	# Initialize the level set geometry
	from utils.ls_utils import init_level_set
	
	ls_func = init_level_set(msh, parameters, parameters.mesh_case)
	ls_func.x.scatter_forward()

.. container:: images-row

	..  container:: centered-figure

		.. _levelSetInit:

		.. figure:: images/demo_compliance/levelSetInit.png
			:alt: Level set initialization
			:align: center
			:width: 100%

			Initialization of the level set function.

	..  container:: centered-figure

		.. _levelSetInitWarp:

		.. figure:: images/demo_compliance/levelSetInitWarp.png
			:alt: Level set warped
			:align: center
			:width: 100%

			Initialization of the level set (warped projection).
		
**3. Boundary Conditions and Solvers**

We specify the Dirichlet and Neumann boundary conditions, and instantiate our specific problem topology (`Compliance_Problem`) alongside the underlying solvers (Ersatz and CutFEM).

.. code-block:: python
	:linenos:

	from fem.boundary_conditions import initialize_boundary_conditions, initialize_shift
	from config.problem import Compliance_Problem
	from solvers.ersatz_elastic_solver import ErsatzElasticSolver
	from solvers.cutfem_elastic_solver import CutFEMElasticSolver
	from levelset.levelSet_tool import Advection, Reinitialization

	bcs, bc_velocity, ds = initialize_boundary_conditions(parameters.mesh_case, msh, V, V_ls, parameters)
	shift = initialize_shift(parameters.mesh_case, msh, parameters)

	problem_topo = Compliance_Problem()

	# Initialize solvers
	AdvectionSolver = Advection(ls_func, V_ls, dt=parameters.dt)
	ReinitSolver = Reinitialization(ls_func, V_ls, l=parameters.l_reinit)
	CutFemSolver = CutFEMElasticSolver(ls_func, V_ls, V, ds, bcs, bc_velocity, parameters, problem_topo, shift)

**4. Primal Problem Resolution**

We compute the displacement field :math:`u_h` by solving the linear elasticity problem. The code elegantly switches between CutFEM and Ersatz based on the parameter flags.

.. code-block:: python
	:linenos:
	
	if parameters.cutFEM == 1:
	    uh, ph = CutFemSolver.cutfem_solver(ls_func, parameters, problem_topo)
	    measure = CutFemSolver.dxq
	else:
	    # Ersatz fallback (not shown for brevity, see main.py)
	    pass

.. container:: images-row

	..  container:: centered-figure

		.. _dispCutFEM:

		.. figure:: images/demo_compliance/dispCutFEM.png
			:align: center
			:width: 100%

			Displacement field (Iteration 0) with CutFEM.

	..  container:: centered-figure

		.. _dispErsatz:

		.. figure:: images/demo_compliance/dispErsatz.png
			:align: center
			:width: 100%
			
			Displacement field (Iteration 0) with Ersatz method.

**5. Shape Derivative and Descent Direction**

The descent direction is derived by computing the shape derivative integrand, followed by Riesz representation and velocity normalization to ensure numerical stability.

.. code-block:: python
	:linenos:

	from levelset.velocity_tools import prepare_descent, descent_direction, velocity_normalization
	
	shape_derivative = problem_topo.shape_derivative_integrand(uh, ph, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, parameters, measure)
	constraint = problem_topo.constraint(uh, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, parameters, measure, 0)
	shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(uh, ph, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, parameters, measure)

	# Update ALM parameters
	from optimization import almMethod
	almMethod.maj_param_constraint_optim(parameters, constraint)

	# Compute and regularize the advection velocity
	resources = prepare_descent(msh, V_ls, parameters)
	v_reg = descent_direction(ls_func, msh, parameters, bc_velocity, V_ls, constraint, shape_derivative_integrand_constraint, shape_derivative, resources)
	
	# Normalize velocity in place
	norm_factor = velocity_normalization(v_reg, parameters.alpha_reg_velocity)
	v_reg.x.array[:] *= norm_factor
	v_reg.x.scatter_forward()

..  container:: centered-figure

	.. _velocity_field:

	.. figure:: images/demo_compliance/velocity_field.png
		:align: center
		:width: 70%

		Regularized velocity field used for the level set advection.
	   
**6. Advection and Reinitialization**

Finally, the boundary is advected by solving the Hamilton-Jacobi equation, and the level set function is periodically reinitialized to maintain its signed distance property.

.. code-block:: python
	:linenos:
	
	from optimization import opti_tool

	# Evaluate cost and lagrangian (required for CFL heuristic)
	cost = problem_topo.cost(uh, ph, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, measure, parameters)
	lagrangian_cost = cost + parameters.ALM_lagrangian_multiplicator * constraint + 0.5 * parameters.ALM_penalty_parameter * constraint**2
	
	# Dynamically adjust time steps for CFL condition
	parameters.dt, adv_bool = opti_tool.catch_NAN(cost, lagrangian_cost, constraint, parameters.dt, 0)
	
	# Advect the boundary using the normalized velocity
	ls_func_advected = AdvectionSolver.cut_fem_adv(v_reg, parameters.dt)
	
	# Periodically reinitialize the level set to a signed distance function
	ReinitSolver.reinitializationPC_inplace(ls_func_advected, parameters.step_reinit)


.. container:: images-row

	..  container:: centered-figure

		.. figure:: images/demo_compliance/levelSetInit.png
			:align: center
			:width: 100%

			Level set before advection.

	..  container:: centered-figure

		.. _levelSetReinit:

		.. figure:: images/demo_compliance/levelSetReinit.png
			:align: center
			:width: 100%

			Reinitialized level set function.

.. _finalresCutFEM:

CutFEM solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

The optimization results using the rigorous CutFEM method, obtained after a fixed limit of 300 iterations, are provided below. The resulting topology strictly satisfies the imposed volume constraints while achieving high-resolution boundary tracking without re-meshing.

.. raw:: html

		<video width="640" height="480" controls>
		    <source src="_static/output_CutFEM.mp4" type="video/mp4">
		    Your browser does not support the video tag.
		</video>
		
.. _finalresErsatz:

Ersatz solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

The analogous results utilizing the fictitious material (Ersatz) approach converge to a remarkably similar geometry. While the Ersatz method exhibits slight boundary diffusion inherent to classical elements, OptiCut's implementation still provides a highly robust continuum approximation.

.. raw:: html

		<video width="640" height="480" controls>
		    <source src="_static/output_Ersatz.mp4" type="video/mp4">
		    Your browser does not support the video tag.
		</video>
