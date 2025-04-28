.. _demoVM:

Lp norm of Von Mises criteria minimization
===============================================

Problem definition
---------------------

We seek the optimal shape, :math:`\widetilde{\Omega}\subset D`, that minimizes the Lp norm of Von Mises constraint of the structure for a linear elastic material subject to Dirichlet and Neumann boundary conditions. 
We impose a target area on the structure, which results in the addition of an equality constraint.
The optimization problem is defined as

.. math::
		:label: eq:JuVM


		\begin{cases}
		\underset{\Omega\in\mathcal{O}}{\min}J_{2}(\Omega) & \!\!\!\!
		=\underset{\Omega\in\mathcal{O}}{\min}\left(\int_{\Omega}\left(\frac{\sigma_{\text{VM}}(u)}{\overline{\sigma }}\right)^{p}\text{ }dx\right)^{\frac{1}{p}}\\
		C(\Omega) & \!\!\!\!
		=0\\
		a\left(u,v\right) & 
		\!\!\!\!=l\left(v\right)
		\end{cases}


with :math:`\overline{V}` the target volume, :math:`\sigma_{VM}(u)` a positive scalar value corresponding to Von Mises yield criterion defined as:

.. math::
	
    	\sigma_{VM}(u)=\sqrt{\frac{2}{3}\left\langle s(u) , s(u)\right \rangle },

where

.. math::
	
		s(u)=\sigma(u)-\frac{1}{3}\text{Tr}(\sigma(u))\text{Id}.

Here, :math:`\sigma_{VM}` is normalized by :math:`\overline{\sigma }`, a positive scalar-value.
u is the displacement field solution to PDE equation and the constraint over the area is defined as:

.. math::

	C(\Omega)=\int_{\Omega}dx - \overline{V}

with :math:`\overline{V}` the target volume.  
  
  


Shape derivative 
~~~~~~~~~~~~~~~~~~~~~~~~
For greater convenience to solve :eq:`eq:JuVM` we define the following function

.. math::

		\widetilde{J}(\Omega)=\int_{\Omega}\left(\frac{\sigma_{\text{VM}}(u)}{\overline{\sigma }}\right)^{p}\text{ }dx.

		
Then, shape derivative of :math:`J(\Omega)` can be write by chain rule as:

.. math::


    J'(\Omega)(\theta)=\frac{1}{p}\Bigl(\widetilde{J}(\Omega)\Bigr)^{\frac{1}{p}-1}\widetilde{J}'(\Omega)(\theta)


Using the Céa method the shape derivative is defined as :

.. math::
		:label: eq:6

		\widetilde{J}'(\Omega)(\theta) = \int_{\partial\Omega}\theta\cdot n \Biggl[\Bigl(\frac{\sigma_{\text{VM}}(u_{\Omega})}{\overline{\sigma}}\Bigr)^{p}-2\mu\varepsilon(u_{\Omega}):\varepsilon(p_{\Omega})-\lambda(\nabla\cdot u_{\Omega})(\nabla\cdot p_{\Omega})\Biggr]\text{ }ds



where :math:`n` is the unit normal.


To account for the constraint in the optimization, the ALM (see :ref:`ALM` ) is used.
First, we define the modified cost function as: 

.. math::

		\begin{align}
		\mathcal{J}(\Omega) &= J(\Omega) +\alpha C(\Omega)+\frac{\beta}{2} C^{2}(\Omega).
		\end{align}

Then, we calculate the shape derivative of this new function.
We obtain the following shape derivative:

.. math::


		\begin{align}
		\mathcal{J}'(\Omega)(\theta) &= J'(\Omega)(\theta) +\alpha C'(\Omega)(\theta) +\beta C(\Omega)C'(\Omega)(\theta).
		\end{align}
		

		
Using the shape optimization algorithm, the descent direction is directly defined as:


.. math::
		:label: velocity_vm

		\begin{multline}
			v\left(u_{\Omega},p_{\Omega}\right) 
			= 
			\frac{1}{p}\left[\int_{\Omega}\left(\frac{\sigma_{\text{VM}}\left(u_{\Omega}\right)}{\overline{\sigma}}\right)^{p}dx\right]^{\frac{1}{p}-1}\\
			\times
			\left[ 
				\left(\frac{\sigma_{\text{VM}}\left(u_{\Omega}\right)}{\overline{\sigma}}\right)^{p}-2\mu\varepsilon\left(u_{\Omega}\right):\varepsilon\left(p_{\Omega}\right)
				-
				\lambda\left(\nabla\cdot u_{\Omega}\right)\left(\nabla\cdot p_{\Omega}\right)
			\right]\\
			+ \alpha+\beta C(\Omega).
		\end{multline}




Algorithm
--------------------

.. math::

    \begin{array}{l}
    \textbf{BEGIN} \\
	\quad u_{h} \gets \text{Resolution of Primal problem (elasticity linear): } a\left(u,v\right)=l\left(v\right)\\
    \quad p_{h} \gets \text{Resolution of Dual problem : } \partial_{u} J(\Omega;v)\mid_{u=u_{\Omega}}=a(v,p) \\
	\quad \text{WHILE } \left\Vert J\left(\Omega_{n+1}\right)-J\left(\Omega_{n}\right)\right\Vert <tol \text{:} \\
    \quad \quad n \gets n+1 \\
	\quad \quad \lambda_{ALM}^{n},\mu_{ALM}^{n} \gets \text{ Update ALM parameters}\\
	\quad \quad v \gets \text{Explicit calculation of the velocity, defined on }\Gamma \\
	\quad \quad v_{ext} \gets \text{Extension of the velocity in } D\\
	\quad \quad v_{\text{reg}} \gets \text{Normalization of the velocity} \\
	\quad \quad \text{WHILE  adv_NAN}\neq1 \\
    \quad \quad \quad \phi_{\text{temp}} \gets \text{Advection }\\
	\quad \quad \quad \phi_{\text{temp}} \gets \text{Reinitialization}\\
	\quad \quad \quad u_{h} \gets \text{Resolution of Primal problem (elasticity linear): } a\left(u,v\right)=l\left(v\right)\\
	\quad \quad \quad p_{h} \gets \text{Resolution of Dual problem : } \partial_{u} J(\Omega;v)\mid_{u=u_{\Omega}}=a(v,p) \\
	\quad \quad \quad \text{dt, } j_{\text{max}} \text{, adv_NAN} \gets \text{Update parameters}\\
	\quad \quad \phi \gets \phi_{temp}\\
    \textbf{END}\\
    \end{array}



Application 
--------------

We study the case of non-self-adjoint problem. We consider an embedded steel beam of L shape  of :math:`1\text{ m} \times 1\text{ m}`  subjected to a uniformly distributed tensile load in :math:`\Gamma_{N}`, such that :math:`g=-0.1 e_{y}` GPa. 
We fixed :math:`p=10` for the power of the Lp norm and we defined :math:`\overline{\sigma } = 3`.
The numerical values used as parameters in the mechanical model are detailed in the Table bellow.
The domain :math:`\Omega` included in :math:`D` is initialized as shown in :numref:`domainVM`, and it is discretized with mesh size of :math:`0.01` m as shown in :numref:`meshVM`.
The optimization problem is solved with a target area of :math:`0.3 \text{ m}^{2}` and an initial area of :math:`0.45\text{ m}^{2}`. 
The optimal shape obtained is shown in :ref:`finalresCutFEMVM`. 


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

			Initialization of the mesh.


.. _paramMecaVM:

.. table:: Mechanical parameters
		:align: center


		+--------------------+------------+------------+
		| **Parameter**      | **Value**  | **Unity**  | 
		+====================+============+============+
		| E (Young Modulus)  | 210        |  **GPa**   | 
		+--------------------+------------+------------+
		| :math:`\nu`        | 0.33       |            |
		+--------------------+------------+------------+
		| Strength           | 0.1        | **GPa**    |
		+--------------------+------------+------------+




Initialization of all parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to instantiate the Parameter object and then initialize all of its attributes using the provided data file 
named "param_VonMises.txt".

.. code-block:: python
	:linenos:
	
	parameters = Parameters()
	name = "param_vonMises.txt"
	parameters.set__paramFolder(name)
	

Mesh Generation
~~~~~~~~~~~~~~~~~~~

Then we load the mesh:

.. code-block:: python
	:linenos:
	
	with io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
        msh, ct, _ = io.gmshio.read_from_msh("mesh/L_shape.msh", MPI.COMM_WORLD, 0, gdim=2)
        xdmf.write_mesh(msh)



Initialization of the spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

Next, we generate the spaces for all the functions that we will use.

.. code-block:: python
	:linenos:

	V = fem.functionspace(msh, ("Lagrange", 1, (msh.geometry.dim, )))
	V_vm = fem.functionspace(msh, ("Lagrange", 2, (msh.geometry.dim, )))
	V_ls = fem.functionspace(msh, ("Lagrange", 1))
	Q = fem.functionspace(msh, ("DG", 0))
	V_DG = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, )))

	# Initialization of the spacial coordinate
	x = ufl.SpatialCoordinate(msh)
	
Initialization of the level set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To initialize the level set function, we use the module geometry_initialization. 
In this module, we define all the desired initializations for the level set function and adjust the call based on the desired configuration.

.. code-block:: python
	:linenos:

	import geometry_initialization

	ls_func_ufl = geometry_initialization.level_set_L_shape(x) 
	ls_func_expr = fem.Expression(ls_func_ufl, V_ls.element.interpolation_points())
	ls_func = Function(V_ls)
	ls_func.interpolate(ls_func_expr)




.. container:: images-row

	..  container:: centered-figure

		.. _levelSetInitVM:

		.. figure:: images/VonMises_fic_demo/levelSetInitVM.png
			:alt: Exemple d'image
			:align: center
			:width: 100%

			Initialization of the level set.


	..  container:: centered-figure

		.. _levelSetInitWarpVM:

		.. figure:: images/VonMises_fic_demo/levelSetInitWarpVM.png
			:alt: Exemple d'image
			:align: center
			:width: 100%

			Initialization of the level set warped.
			
		

   
Initialization of the boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define the Dirichlet conditions for the linear elasticity problem.

.. code-block:: python
	:linenos:

	import numpy as np
	from dolfinx import fem, mesh

	fdim = msh.topology.dim - 1 # facet dimension

	def clamped_boundary_cantilever(x):
		return np.isclose(x[0], 0)

	boundary_facets = mesh.locate_entities_boundary(msh, fdim,clamped_boundary_cantilever)
	u_D = np.array([0,0], dtype=ScalarType)

	bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)


We define the Neumann conditions for the linear elasticity problem and we define Dirichlet boundary condition for the velocity field.

.. code-block:: python
	:linenos:

	import numpy as np
	from dolfinx.mesh import meshtags


	def load_marker(x):
		return np.logical_and(x[0]>(parameters.lx-1e-6),x[1]>0.35)
		
	# Boundary condition for the velocity field.
	boundary_dofs = fem.locate_dofs_geometrical(V_ls, load_marker) 
	bc_velocity = fem.dirichletbc(ScalarType(0.), boundary_dofs, V_ls) 

	# Neumann boundary condition for the primal problem. 
	facet_indices, facet_markers = [], []
	facets = mesh.locate_entities(msh, fdim, load_marker)
	facet_indices.append(facets)
	facet_markers.append(np.full_like(facets, 2))

	facet_indices = np.hstack(facet_indices).astype(np.int32)
	facet_markers = np.hstack(facet_markers).astype(np.int32)
	sorted_facets = np.argsort(facet_indices)
	facet_tag = meshtags(msh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
	ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tag)



Instantiation of the objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We instantiate the objects of the following classes: 
	-	Reinitialization (:py:meth:`levelSet_tool.Reinitialization`), 
	-	CutFemMethod (:py:meth:`cutfem_method.CutFemMethod`). 


Then, we initialize the coefficients of Lamé with the function :py:meth:`mechanics_tool.lame_compute`.

.. code-block:: python
	:linenos:

	Reinitialization = Reinitialization(ls_func, V_ls=V_ls, l=parameters.l_reinit)

	CutFemMethod = CutFemMethod(ls_func,V_ls, V, ds = ds, bc = bc, parameters = parameters, shift = shift)

	lame_mu,lame_lambda = mechanics_tool.lame_compute(parameters.young_modulus,parameters.poisson)



Instantiation of Problem class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We instantiate the problem class. 
Here, we aim to minimize the Lp norm of the Von Mises contraint, so the problem is of the VMLp_Problem class. 
Defining the problem object then allows for automating the call of functions to calculate the cost, its integrand, 
the integrand of the cost's derivative, the constraint value, the integrand of the constraint, the integrand of the derivative of the constraint, 
and the dual operator if necessary. See :py:meth:`problem.VMLp_Problem` for clarification. 

.. code-block:: python
	:linenos:

	problem_topo = problem.VMLp_Problem()


Resolution of the primal problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First we solve the primal problem, which corresponds to the linear elasticity problem.

.. code-block:: python
	:linenos:
	
	CutFemMethod.set_measure_dxq(ls_func)
	uh = CutFemMethod.primal_problem(ls_func_temp,parameters)

	cost = problem_topo.cost(uh,ph,lame_mu,lame_lambda,measure,parameters)


..  container:: centered-figure

	.. _dispCutFEMVM:

	.. figure:: images/VonMises_fic_demo/dispCutFEM.png
		:alt: Exemple d'image
		:align: center
		:width: 100%

		Displacement field of the first iteration with CutFEM.




Resolution of the dual problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Then we solve the dual problem by calling the function :py:meth:`cutfem_method.CutFemMethod.adjoint_problem`, the autoamtic differentiation is used,
as describe in :ref:`linearFormLpnorm`. 

.. code-block:: python
	:linenos:
	

	CutFemMethod.set_measure_dxq(ls_func)
	ph = CutFemMethod.dual_problem(ls_func_temp,parameters)


..  container:: centered-figure

	.. _dualCutFEM:

	.. figure:: images/VonMises_fic_demo/dualCutFEM.png
		:alt: Exemple d'image
		:align: center
		:width: 100%

		Displacement field of the first iteration with CutFEM.



Shape derivative computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
	:linenos:
	

	shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,lame_mu,lame_lambda,parameters,measure)

	rest_constraint = problem_topo.constraint(uh,lame_mu,lame_lambda,parameters,measure,(1-parameters.cutFEM)*xsi)

	shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(uh,ph,lame_mu,lame_lambda,parameters,measure)


Update ALM parameters
~~~~~~~~~~~~~~~~~~~~~~~~~

To update the Augmented Lagrangian parameters we call the :py:meth:`almMethod.maj_param_constraint_optim`:

.. code-block:: python
	:linenos:
	
	almMethod.maj_param_constraint_optim(parameters,rest_constraint)

.. note::

	To simplify, the penalization method can also be used by simply initializing parameters.ALM to 0. The penalization value must be chosen very carefully, depending on the problem.


Compute the descent direction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compute the advection velocity, we first call the function :py:meth:`cutfem_method.CutFemMethod.descent_direction`.
Then, we normalize the solution field using an adaptation of the H1 norm by calling the function  :py:meth:`cutfem_method.CutFemMethod.velocity_normalization` . 
Finally, we compute the maximum of the velocity field to determine a time step for advection that satisfies the CFL condition.

.. code-block:: python
	:linenos:
	
	velocity_field = descent_direction(CutFemMethod.level_set,msh,parameters,bc_velocity,V_ls,\
		                            V_DG,rest_constraint,shape_derivative_integrand_constraint,shape_derivative)
	velocity_field = CutFemMethod.velocity_normalization(velocity_field,parameters.alpha_reg_velocity)

	velocity_expr = fem.Expression(velocity_field, V_ls.element.interpolation_points())
	velocity = fem.Function(V_ls)
	velocity.interpolate(velocity_expr)

	max_velocity = comm.allreduce(np.max(np.abs(velocity.x.array[:])),op=MPI.MAX)


..  container:: centered-figure

	.. _velocity_fieldVM:

	.. figure:: images/VonMises_fic_demo/velocity_field.png
		:alt: Exemple d'image
		:align: center
		:width: 100%

		Velocity field of the first iteration.


	   
Advection of the level set function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To advect the level set in the computed descent direction, we call the function :py:meth:`cutfem_method.CutFemMethod.cut_fem_adv`. 
To optimize our code, we first call :py:meth:`opti_tool.adapt_c_HJ`, which optimizes the choice of the parameter c when computing the advection time step. 
This function uses the evolution of the convergence criterion to gradually decrease the value of c. Then, the function :py:meth:`opti_tool.adapt_dt` is 
called to adjust the advection time step.



.. code-block:: python
	:linenos:
	
	c_param_HJ = cost_func.adapt_c_HJ(c_param_HJ,crit,parameters.tol_cost_func,lagrangian)

	parameters.dt  =  cost_func.adapt_dt(lagrangian_cost,lagrangian_cost_previous,max_velocity,parameters,c_param_HJ)

	ls_func_temp = CutFemMethod.cut_fem_adv(ls_func_temp,(1/adv_bool)*parameters.dt, velocity_field)

Reinitialization of the level set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reinitialize the level set using the P.C. method after setting the number of frequency for the 
reinitialization to parameters.step_reinit we call the method :py:meth:`levelSet_tool.Reinitialization.reinitializationPC`

.. code-block:: python
	:linenos:
	
	if ((j%parameters.freq_reinit)==0):
        	ls_func_temp, temp_func = Reinitialization.reinitializationPC(ls_func_temp,parameters.step_reinit)
                

.. note::
	
	The frequency of the reinitialization is the number of iterations of optimization problem we want to do after reinitialized the level set function.
	For this exemple we set it to :math:`1`. 



Update all the parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
	:linenos:
	
	parameters.dt, adv_bool = cost_func.catch_NAN(cost,lagrangian_cost,rest_constraint,parameters.dt,adv_bool)

	if adv_bool<2:
		parameters.j_max = cost_func.vm_adapta_HJ(lagrangian_cost,lagrangian_cost_previous,parameters.j_max,parameters.dt,parameters)
	else: 
		parameters.j_max = 1

	    


.. _finalresCutFEMVM:

CutFEM solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

The results of the optimization with the CutFEM method, obtained after a fixed number of iterations set to 1000, 
are provided below.


.. raw:: html

		<video width="640" height="480" controls>
		    <source src="_static/output_VM.mp4" type="video/mp4">
		    Your browser does not support the video tag.
		</video>
		

