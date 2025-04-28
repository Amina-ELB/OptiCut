.. _demo:

Compliance minimization
=========================================
Minimizing the compliance of a structure is a classic problem that has been the subject of numerous studies. 

Problem definition
---------------------

We seek the optimal shape, :math:`\widetilde{\Omega}\subset D`, that minimizes the compliance of the structure for a linear elastic material subject to Dirichlet and Neumann boundary conditions. 
We impose a target area on the structure, which results in the addition of an equality constraint.
The optimization problem is defined as


.. math::

		\begin{align}\begin{cases}
		\underset{\Omega\in\mathcal{O}}{\min}J(\Omega) & \!\!\!\!=\underset{\Omega\in\mathcal{O}}{\min}\int_{\Omega}(\mu\varepsilon(u):\varepsilon(u)+\frac{\lambda}{2}\nabla\cdot u\nabla\cdot u)\text{ }dx\\
		C(\Omega) & \!\!\!\!=0\\
		a\left(u,v\right) & \!\!\!\!=l\left(v\right)
		\end{cases}\end{align}

u is the displacement field solution to PDE equation and the constraint over the area is defined as:

.. math::

	C(\Omega)=\int_{\Omega}dx - \overline{V}

with :math:`\overline{V}` the target volume.  
  
  


Shape derivative 
~~~~~~~~~~~~~~~~~~~~~~~~

Using the Céa method the shape derivative of the compliance minimization problem is defined as :

.. math::


		\begin{align}
		J'(\Omega)(\theta) &= -\int_{\partial\Omega}\theta\cdot n[ \mu\varepsilon(u_{\Omega}):\varepsilon(u_{\Omega})+\frac{\lambda}{2}(\nabla\cdot u_{\Omega})(\nabla\cdot u_{\Omega})]\text{ }ds
		\end{align}
		
		
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

		v(u_{\Omega}) = 2\mu\varepsilon(u_{\Omega}):\varepsilon(u_{\Omega})+\lambda(\nabla\cdot u_{\Omega})(\nabla\cdot u_{\Omega}) + \alpha+\beta C(\Omega). 



Algorithm
--------------------

.. math::

    \begin{array}{l}
    \textbf{BEGIN} \\
	\quad u_{h} \gets \text{Resolution of Primal problem (elasticity linear): } a\left(u,v\right)=l\left(v\right)\\
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
	\quad \quad \quad \text{dt, } j_{\text{max}} \text{, adv_NAN} \gets \text{Update parameters}\\
	\quad \quad \phi \gets \phi_{temp}\\
    \textbf{END}\\
    \end{array}



Application 
--------------

We study the case of an embedded steel beam of :math:`1\text{m}\times 2\text{m}` subjected to a uniformly distributed tensile load at :math:`\pm 0.5` m such that :math:`g=-0.1 e_{y}` GPa.
The numerical values used as parameters in the mechanical model are detailed in the :numref:`paramMeca`.
The domain :math:`\Omega` included in :math:`D` is initialized as shown in :numref:`compliancedomain`, and it is discretized with a :math:`100\times200` finite element mesh.
The problem is solved with a target area of :math:`1.2 \text{ m}^{2}` and an initial area of :math:`1.64m^{2}`. 
The optimal shape obtained is shown in :ref:`finalresCutFEM` with CutFEM method and in :ref:`finalresErsatz` for Ersatz method. 

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

The first step is to instantiate the Parameter object and then initialize all of its attributes using the provided data file named "param_compliance.txt".

.. code-block:: python
	:linenos:
	
	parameters = Parameters("compliance")
	name = "param_compliance.txt"
	parameters.set__paramFolder(name)
	

Mesh Generation
~~~~~~~~~~~~~~~~~~~

Then to generate the rectangular mesh seen in :numref:`mesh` the function :py:meth:`create_mesh.create_mesh_2D` is called.

.. code-block:: python
	:linenos:
	
	from create_mesh import *
	
	msh = create_mesh_2D(parameters.lx, parameters.ly, int(parameters.lx/parameters.h),int(parameters.ly/parameters.h))



Initialization of the spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~   

Next, we generate the spaces for all the functions that will be used.

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

To initialize the level set function, we use the module :py:meth:`geometry_initialization` and the function :  :py:meth:`geometry_initialization.level_set`. 
In this module, we define all the desired initializations for the level set function and adjust the call based on the desired configuration.

.. code-block:: python
	:linenos:

	import geometry_initialization
	 
	ls_func_ufl = geometry_initialization.level_set(x,parameters) 
	ls_func_expr = fem.Expression(ls_func_ufl, V_ls.element.interpolation_points())
	ls_func = Function(V_ls)
	ls_func.interpolate(ls_func_expr)



.. container:: images-row

	..  container:: centered-figure

		.. _levelSetInit:

		.. figure:: images/demo_compliance/levelSetInit.png
			:alt: Exemple d'image
			:align: center
			:width: 100%

			Initialization of the level set.


	..  container:: centered-figure

		.. _levelSetInitWarp:

		.. figure:: images/demo_compliance/levelSetInitWarp.png
			:alt: Exemple d'image
			:align: center
			:width: 100%

			Initialization of the level set warped.
		
		

   
Initialization of the boundary conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We define the Dirichlet conditions for the linear elasticity problem, denoted :math:`\Gamma_{D}` in :numref:`compliancedomain`.

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


We define the Neumann conditions for the linear elasticity problem, denoted :math:`\Gamma_{N}` in 
:numref:`compliancedomain`. Then, we define Dirichlet boundary condition for the velocity field.

.. code-block:: python
	:linenos:

	import numpy as np
	from dolfinx.mesh import meshtags


	def load_marker(x):
		return np.logical_and(np.isclose(x[0],parameters.lx),np.logical_and(x[1]<(0.55),x[1]>(0.45))) 

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
	-	ErsatzMethod  (:py:meth:`ersatz_method.ErsatzMethod`), 
	-	CutFemMethod (:py:meth:`cutfem_method.CutFemMethod`). 


Then, we initialize the coefficients of Lamé with the function :py:meth:`mechanics_tool.lame_compute`.

.. code-block:: python
	:linenos:

	Reinitialization = Reinitialization(ls_func, V_ls=V_ls, l=parameters.l_reinit)

	ErsatzMethod = ErsatzMethod(ls_func,V_ls, V, ds = ds, bc = bc, parameters = parameters, shift = shift)

	CutFemMethod = CutFemMethod(ls_func,V_ls, V, ds = ds, bc = bc, parameters = parameters, shift = shift)

	lame_mu,lame_lambda = mechanics_tool.lame_compute(parameters.young_modulus,parameters.poisson)



Instantiation of Problem class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We instantiate the problem class. 
Here, we aim to minimize the compliance, so the problem is of the Compliance_Problem class. 
Defining the problem object then allows for automating the call of functions to calculate the cost, its integrand, 
the integrand of the cost's derivative, the constraint value, the integrand of the constraint, the integrand of the derivative of the constraint, 
and the dual operator if necessary. See :py:meth:`problem.Compliance_Problem` for clarification. 

.. code-block:: python
	:linenos:

	problem_topo = problem.Compliance_Problem()


Resolution of linear elasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our code offers two methods for the optimization problem: the ersatz material method and the CutFEM method. 
Here, we demonstrate the implementation of both methods for solving the linear elasticity problem.

.. code-block:: python
	:linenos:
	

	if parameters.cutFEM == 1:
		CutFemMethod.set_measure_dxq(ls_func)
		uh = CutFemMethod.primal_problem(ls_func_temp,parameters)

		measure = CutFemMethod.dxq
	else: 
		xsi = ErsatzMethod.heaviside(ls_func)
		uh = ErsatzMethod.primal_problem(ls_func_temp, parameters) 

		measure = ufl.dx

	cost = problem_topo.cost(uh,ph,lame_mu,lame_lambda,measure,parameters)


.. container:: images-row

	..  container:: centered-figure

		.. _dispCutFEM:

		.. figure:: images/demo_compliance/dispCutFEM.png
			:alt: Exemple d'image
			:align: center
			:width: 100%

			Displacement field of the first iteration with CutFEM.

	..  container:: centered-figure

		.. _dispErsatz:

		.. figure:: images/demo_compliance/dispErsatz.png
			:alt: Exemple d'image
			:align: center
			:width: 100%
			
			Displacement field of the first iteration with Ersatz method.

.. note::

	To remind, the differences CutFEM and Ersatz primal problem are defined in :eq:`eq:20` and in :eq:`eqn:elasticity_weak_form`.

.. note::

	The heaviside function is needed for the ersatz method. The definition is given by :eq:`eqn:smooth_heaviside` and we choose :math:`\varepsilon=0.001`.
	

Shape derivative computation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
	:linenos:
	

	shape_derivative = problem_topo.shape_derivative_integrand(uh,ph,lame_mu,lame_lambda,parameters,measure)

	rest_constraint = problem_topo.constraint(uh,lame_mu,lame_lambda,parameters,measure,(1-parameters.cutFEM)*xsi)
	
	shape_derivative_integrand_constraint = problem_topo.shape_derivative_integrand_constraint(uh,ph,lame_mu,lame_lambda,parameters,measure)

.. note::

	For the compliance minimization the dual solution is automatically initialized as zero function. 
	


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
To compute the advection velocity, we first call on of this two functions :

- :py:meth:`cutfem_method.CutFemMethod.descent_direction`
- :py:meth:`ersatz_method.ErsatzMethod.descent_direction`

depending on the chosen method. 

Next, we normalize the solution field using an adaptation of the H1 norm by calling the function  :py:meth:`cutfem_method.CutFemMethod.velocity_normalization` . 
Finally, we compute the maximum of the velocity field to determine a time step for advection that satisfies the CFL condition.

.. code-block:: python
	:linenos:
	
	velocity_field = problem_topo.descent_direction(CutFemMethod.level_set,msh,parameters,bc_velocity,V_ls,\
		                            V_DG,rest_constraint,shape_derivative_integrand_constraint,shape_derivative)
	velocity_field = CutFemMethod.velocity_normalization(velocity_field,parameters.alpha_reg_velocity)

	velocity_expr = fem.Expression(velocity_field, V_ls.element.interpolation_points())
	velocity = fem.Function(V_ls)
	velocity.interpolate(velocity_expr)

	max_velocity = comm.allreduce(np.max(np.abs(velocity.x.array[:])),op=MPI.MAX)


..  container:: centered-figure

	.. _velocity_field:

	.. figure:: images/demo_compliance/velocity_field.png
		:alt: Exemple d'image
		:align: center
		:width: 70%

		Velocity field of the first iteration.


	   
Advection of the level set function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To advect the level set in the computed descent direction, we call the function :py:meth:`cutfem_method.CutFemMethod.cut_fem_adv`. 
To optimize our code, we first call :py:meth:`opti_tool.adapt_c_HJ`, which optimizes the choice of the parameter c when computing the advection time step. 
This function uses the evolution of the convergence criterion to gradually decrease the value of c. Then, the function :py:meth:`opti_tool.adapt_dt` is 
called to adjust the advection time step.

.. code-block:: python
	:linenos:
	
	c_param_HJ = opti_tool.adapt_c_HJ(c_param_HJ,crit,parameters.tol_cost_func,lagrangian)

	parameters.dt  =  opti_tool.adapt_dt(lagrangian_cost,lagrangian_cost_previous,max_velocity,parameters,c_param_HJ)

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

.. container:: images-row

	..  container:: centered-figure

		.. figure:: images/demo_compliance/levelSetInit.png
			:alt: Exemple d'image
			:align: center
			:width: 100%

			Initialization of the level set.

	..  container:: centered-figure

		.. _levelSetReinit:

		.. figure:: images/demo_compliance/levelSetReinit.png
			:alt: Exemple d'image
			:align: center
			:width: 100%

			Reinitialized level set function.


.. container:: images-row

	..  container:: centered-figure


		.. figure:: images/demo_compliance/levelSetInitWarp.png
			:alt: Exemple d'image
			:align: center
			:width: 100%

			Initialization of the level set.


	..  container:: centered-figure

		.. _levelSetReinitWarp:

		.. figure:: images/demo_compliance/levelSetReinitWarp.png
			:alt: Exemple d'image
			:align: center
			:width: 100%

			Reinitialized level set function.
	   	   

Update all the parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
	:linenos:
	
	parameters.dt, adv_bool = opti_tool.catch_NAN(cost,lagrangian_cost,rest_constraint,parameters.dt,adv_bool)

	if adv_bool<2:
		parameters.j_max = opti_tool.vm_adapta_HJ(lagrangian_cost,lagrangian_cost_previous,parameters.j_max,parameters.dt,parameters)
	else: 
		parameters.j_max = 1


.. _finalresCutFEM:

CutFEM solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

The results of the optimization with the CutFEM method, obtained after a fixed number of iterations set to 300, 
are provided below. The geometry of the domain obtained corresponds to the usual results found in the 
literature. It is observed that the constraint imposed on the volume is well satisfied.



.. raw:: html

		<video width="640" height="480" controls>
		    <source src="_static/output_CutFEM.mp4" type="video/mp4">
		    Your browser does not support the video tag.
		</video>
		



.. _finalresErsatz:

Ersatz solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	

The results of the optimization with the fictitious material method, obtained after a fixed number of 
iterations set to 300, are provided below. The geometry of the domain is very similar to the previous one, 
obtained with CutFEM. It is observed that the constraint imposed on the volume is also well satisfied.




.. raw:: html

		<video width="640" height="480" controls>
		    <source src="_static/output_Ersatz.mp4" type="video/mp4">
		    Your browser does not support the video tag.
		</video>
