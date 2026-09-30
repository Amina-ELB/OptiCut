.. _demoVM:

Lp norm of Von Mises criteria minimization
===============================================

Running the Tutorial
--------------------

To reproduce this stress-minimization example, navigate to the `src` directory and execute the main script with the Von Mises parameter file:

.. code-block:: bash

    cd src
    python3 main.py parameters/param_vonMises.txt

**Visualization in ParaView:**
The results are saved in `src/res/results.xdmf`. Open this file in ParaView, click **Apply**, and use the time-player controls to animate the geometry. The videos below were generated using exactly this method, showing the simultaneous evolution of the topology and the cost function.

Problem definition
---------------------

In this tutorial, we aim to find the optimal shape, :math:`\widetilde{\Omega}\subset D`, that minimizes the global stress within the structure. To achieve this in a differentiable manner, we minimize the :math:`L^p` norm of the Von Mises stress criterion across the linear elastic material, subject to Dirichlet and Neumann boundary conditions and a strict volume constraint.

The continuous optimization problem is formally defined as:

.. math::
		:label: eq:JuVM

		\begin{cases}
		\underset{\Omega\in\mathcal{O}}{\min}J_{p}(\Omega) & \!\!\!\!
		=\underset{\Omega\in\mathcal{O}}{\min}\left(\int_{\Omega}\left(\frac{\sigma_{\text{VM}}(u)}{\overline{\sigma }}\right)^{p}\text{ }dx\right)^{\frac{1}{p}}\\
		C(\Omega) & \!\!\!\!
		=0\\
		a\left(u,v\right) & 
		\!\!\!\!=l\left(v\right)
		\end{cases}

where :math:`\overline{V}` is the target volume constraint, and :math:`\sigma_{VM}(u)` is the positive scalar Von Mises yield criterion, defined as:

.. math::
	
    	\sigma_{VM}(u)=\sqrt{\frac{2}{3}\left\langle s(u) , s(u)\right \rangle },

with the deviatoric stress tensor :math:`s(u)`:

.. math::
	
		s(u)=\sigma(u)-\frac{1}{3}\text{Tr}(\sigma(u))\text{Id}.

To improve numerical conditioning, :math:`\sigma_{VM}` is normalized by :math:`\overline{\sigma}`, an arbitrary strictly positive scaling factor.

Shape derivative 
~~~~~~~~~~~~~~~~~~~~~~~~

To efficiently solve the continuous constrained problem :eq:`eq:JuVM`, we introduce the auxiliary functional:

.. math::

		\widetilde{J}(\Omega)=\int_{\Omega}\left(\frac{\sigma_{\text{VM}}(u)}{\overline{\sigma }}\right)^{p}\text{ }dx.
		
Using the chain rule, the shape derivative of the actual cost :math:`J(\Omega)` is obtained as:

.. math::

    J'(\Omega)(\theta)=\frac{1}{p}\Bigl(\widetilde{J}(\Omega)\Bigr)^{\frac{1}{p}-1}\widetilde{J}'(\Omega)(\theta)

Through Céa's method (and the introduction of an adjoint state :math:`p_{\Omega}` due to the non-self-adjoint nature of the stress functional), the shape derivative reduces to an integral over the boundary :math:`\partial\Omega`:

.. math::
		:label: eq:6

		\widetilde{J}'(\Omega)(\theta) = \int_{\partial\Omega}\theta\cdot n \Biggl[\Bigl(\frac{\sigma_{\text{VM}}(u_{\Omega})}{\overline{\sigma}}\Bigr)^{p}-2\mu\varepsilon(u_{\Omega}):\varepsilon(p_{\Omega})-\lambda(\nabla\cdot u_{\Omega})(\nabla\cdot p_{\Omega})\Biggr]\text{ }ds

The Augmented Lagrangian Method (ALM) is employed to encapsulate the volume constraints:

.. math::

		\mathcal{J}(\Omega) = J(\Omega) +\lambda_{ALM} C(\Omega)+\frac{\mu_{ALM}}{2} C^{2}(\Omega).

yielding the final augmented shape derivative:

.. math::

		\mathcal{J}'(\Omega)(\theta) = J'(\Omega)(\theta) +\lambda_{ALM} C'(\Omega)(\theta) +\mu_{ALM} C(\Omega)C'(\Omega)(\theta).
		
The descent direction :math:`v(u_{\Omega},p_{\Omega})` directly mirrors the shape derivative boundary integrand, heavily relying on both the primal displacement :math:`u_{\Omega}` and the adjoint dual state :math:`p_{\Omega}`:

.. math::
		:label: velocity_vm

		\begin{aligned}
			v\left(u_{\Omega},p_{\Omega}\right) 
			=& 
			\frac{1}{p}\left[\int_{\Omega}\left(\frac{\sigma_{\text{VM}}\left(u_{\Omega}\right)}{\overline{\sigma}}\right)^{p}dx\right]^{\frac{1}{p}-1}\\
			&\times
			\left[ 
				\left(\frac{\sigma_{\text{VM}}\left(u_{\Omega}\right)}{\overline{\sigma}}\right)^{p}-2\mu\varepsilon\left(u_{\Omega}\right):\varepsilon\left(p_{\Omega}\right)
				-
				\lambda\left(\nabla\cdot u_{\Omega}\right)\left(\nabla\cdot p_{\Omega}\right)
			\right]\\
			&+ \lambda_{ALM} + \mu_{ALM} C(\Omega).
		\end{aligned}

Algorithm
--------------------

Unlike compliance minimization, the stress minimization problem is **non-self-adjoint**. This implies that at every iteration, OptiCut must solve both the primal physical problem and a separate adjoint problem to evaluate the sensitivity.

.. code-block:: text

    BEGIN
        uh ← Solve Primal problem : a(u,v) = l(v)
        ph ← Solve Dual problem : ∂_u J(Ω;v)|_{u=u_Ω} = a(v,p)
        
        WHILE not converged:
            λ_ALM, μ_ALM ← Update ALM parameters
            v ← Compute descent direction using uh and ph
            v_ext ← Extend velocity
            
            WHILE adv_NAN ≠ 1 :
                φ_temp ← Advection
                uh ← Solve Primal problem
                ph ← Solve Dual problem
                
            φ ← φ_temp
    END

Application 
--------------

We investigate an embedded steel "L-shape" beam structure of :math:`1\text{ m} \times 1\text{ m}` subjected to a uniformly distributed tensile load on :math:`\Gamma_{N}`, scaling to :math:`g=-0.1 e_{y}\text{ GPa}`. 
We fix the :math:`L^p` norm exponent to :math:`p=10` and define the scaling factor :math:`\overline{\sigma } = 3`.
The domain :math:`\Omega \subset D` is initialized as shown in :numref:`domainVM` and discretized with a mesh resolution of :math:`0.01\text{ m}` (:numref:`meshVM`).

.. container:: images-row

	..  container:: centered-figure

		.. _domainVM:

		.. figure:: images/VonMises_fic_demo/domain.png
			:align: center
			:width: 100%

			Initialization of :math:`\Omega\subset\text{D}`

	..  container:: centered-figure

		.. _meshVM:   
		
		.. figure:: images/VonMises_fic_demo/mesh.png 
			:width: 100%
			:align: center

			Initialization of the L-shape mesh.


Implementation Highlights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implementation is largely identical to the compliance minimization problem, demonstrating OptiCut's modular design. The key difference lies in initializing the `VMLp_Problem` class and automatically computing the dual (adjoint) problem.

**1. Problem Initialization**

We load the parameters and explicitly request the Von Mises problem topology solver.

.. code-block:: python
	:linenos:
	
	from utils.config_utils import load_parameters
	from config.problem import VMLp_Problem

	parameters = load_parameters(use_file=1, filename="param_VonMises.txt")
	problem_topo = VMLp_Problem()
			
**2. Primal and Dual Problem Resolution**

At each optimization step, OptiCut evaluates the linear elasticity state (Primal), computes the dual operator specific to the Von Mises criterion using automatic differentiation via UFL, and solves the adjoint system (Dual).

.. code-block:: python
	:linenos:
	
	# Solve Primal (Displacement)
	uh, _ = CutFemSolver.cutfem_solver(ls_func, parameters, problem_topo)
	
	from utils import data_manipulation

	# Evaluate Dual Operator specific to the Von Mises topology
	vm_calculator = data_manipulation.VonMisesCalculator(msh, V_ls, spaces["Q"], uh, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, CutFemSolver.level_set)
	vm_list = vm_calculator.compute()
	dual_operator = problem_topo.dual_operator(uh, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, parameters, msh, CutFemSolver.dxq, vm_list)
	
	# Solve Dual (Adjoint State)
	ph = CutFemSolver.adjoint_problem(uh, dual_operator)


.. container:: images-row

	..  container:: centered-figure

		.. _dispCutFEMVM:

		.. figure:: images/VonMises_fic_demo/dispCutFEM.png
			:align: center
			:width: 100%

			Displacement field of the first iteration with CutFEM.

	..  container:: centered-figure

		.. _dualCutFEM:

		.. figure:: images/VonMises_fic_demo/dualCutFEM.png
			:align: center
			:width: 100%

			Adjoint dual state field of the first iteration with CutFEM.


**3. Shape Derivative and Descent**

The velocity field calculation takes into account both `uh` and `ph` seamlessly. OptiCut's unified `descent_direction` function hides the complexity of assembling these combined non-linear boundary integrals.

.. code-block:: python
	:linenos:
	
	from levelset.velocity_tools import prepare_descent, descent_direction, velocity_normalization

	# Evaluate constraint and shape derivative integrands
	constraint = problem_topo.constraint(uh, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, parameters, CutFemSolver.dxq, 0)
	shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(uh, ph, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, parameters, CutFemSolver.dxq)
	shape_derivative = problem_topo.shape_derivative_integrand(uh, ph, CutFemSolver.lame_mu, CutFemSolver.lame_lambda, parameters, CutFemSolver.dxq)
	
	# Regularize and normalize the velocity
	resources = prepare_descent(msh, V_ls, parameters)
	v_reg = descent_direction(ls_func, msh, parameters, bc_velocity, V_ls, constraint, shape_derivative_integrand_constraint, shape_derivative, resources)
	
	# Normalize velocity in place
	norm_factor = velocity_normalization(v_reg, parameters.alpha_reg_velocity)
	v_reg.x.array[:] *= norm_factor
	v_reg.x.scatter_forward()

   
.. _finalresCutFEMVM:

CutFEM Solution for Von Mises
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

The results of the stress-minimization optimization with the CutFEM method exhibit a design strictly distinct from pure compliance minimization. Stress concentrations, specifically near the re-entrant L-shape corner, are actively penalized, leading to a much smoother, rounded inner corner characterized by evenly distributed Von Mises stress contours.

.. raw:: html

		<video width="640" height="480" controls>
		    <source src="_static/output_VM.mp4" type="video/mp4">
		    Your browser does not support the video tag.
		</video>
