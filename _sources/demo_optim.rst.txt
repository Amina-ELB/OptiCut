.. _demoOptim:

Shape optimization method
=========================================
      
Problem definition
---------------------
We seek an optimal shape, :math:`\widetilde{\Omega}\subset\mathbb{R}^{n}`, :math:`d\in\left\{2,3\right\}`, that minimizes a cost function of the structure for a linear elastic material in a fixed working domain :math:`D` subject to Dirichlet and Neumann boundary conditions. 

.. math::

		\widetilde{\Omega}:=\underset{\Omega\in\mathcal{O}\left(D\right)}{\text{argmin}}J\left(\Omega\right)
		
		
with :math:`\mathcal{O}`  a subset of the fixed working domain, :math:`D`. 


The objective function :math:`J` is defined as:

.. math::
		:label: eqn:MinCompliance

		\begin{align}
		J:\mathcal{O}&\rightarrow\mathbb{R}\\
				    \Omega&\rightarrow J(\Omega) = \int_{\Omega}j(u)\text{ }dx
		\end{align}
		
		
where :math:`j` is a function defined from :math:`\Omega` to :math:`\mathbb{R}` and  dependent on the displacement field :math:`u` solution of a PDE.

Inequality and equality constraint can be imposed in the shape optimization problem. 

Function of equality constraint is denoted :math:`C_{1}` and defined by  :math:`C_{1}:\mathcal{O}\rightarrow\mathbb{R}`. 

Function of inequality constraint is denoted :math:`C_{2}` and defined by :math:`C_{2}:\mathcal{O}\rightarrow\mathbb{R}`. 

General shape optimization problem is written:

.. math::
		:label: eqn:Ju_global

		\begin{cases}
		\underset{\Omega\in\mathcal{O}}{\min}J(\Omega) & \!\!\!\!
		=\underset{\Omega\in\mathcal{O}}{\min}\int_{\Omega}j(u)dx \\
		C_{1}(\Omega) & \!\!\!\!=
		0\\
		C_{2}(\Omega) &\!\!\!\!
		<0 \\
		a\left(u,v\right) & \!\!\!\!
		=l\left(v\right) 
		\end{cases} 



Boundaries definition:
~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\Gamma` denote the free boundary, :math:`\partial\Omega=\Gamma\cup\left[\partial D\cap\overline{\Omega}\right]` denote the boundary of :math:`\Omega`. 

One will also distinguish  :math:`\Gamma_{D}` , the part of the boundary where Dirichlet conditions are applied, and :math:`\Gamma_{N}` , the part where Neumann conditions are applied, such that :math:`\Gamma=\Gamma_{D}\cup\Gamma_{N}` and :math:`\Gamma_{D}\cap\Gamma_{N}=\emptyset`. To clarify these definitions, a diagram is given in :numref:`schemaBoundaries`.

..  container:: centered-figure

	.. _schemaBoundaries:

	.. figure:: images/demo_compliance/schema_boundaries.png
		:alt: Exemple d'image
		:align: center
		:width: 50%

		Illustration of the boundaries of the problem definition.
		


.. _ALM:

Augmented lagrangian Method
-------------------------------

Augmented Lagrangian Method is used to solve the constrained optimization problems defined by :eq:`eqn:Ju_global`. Here, we provide a concise overview of the ALM method. We begin by considering the following problem:

.. math::

		\underset{\Omega\in\mathcal{O}}{\min}J(\Omega)\text{ such that  }C(\Omega)=0


where :math:`C(\Omega)=0` represents an equality constraint. 

The problem is reformulated as an equivalent min-max problem:

.. math::

		\underset{\Omega\in\mathcal{O}}{\min}\underset{\alpha\in\mathbb{R}}{\text{ }\max}\left\{J(\Omega)+\alpha C(\Omega)+\frac{\beta}{2}\left|C(\Omega)\right|^{2}\right\} 

where :math:`\alpha` is a Lagrange multiplier, and :math:`\beta>0` is a penalty term. 

The quadratic term helps to stabilize the convergence toward a feasible solution and stabilizes the solution to minimize oscillations.
The min-max problem is solved using a gradient iterative method, in which, the Lagrange multiplier and the penalty parameters are updated at each iteration as follows:

.. math::

		\begin{align}
		\alpha^{n+1}&=\alpha^{n}+\beta C\left(\Omega_{n} \right) \\
		\beta^{n+1}&=\min\left(\hat{\beta},k\beta^{n} \right)
		\end{align}

where :math:`\Omega^{n}` is the domain at iteration :math:`n`, :math:`\hat{\beta}` is the upper limit of the penalty parameter and :math:`k` is a multiplication coefficient.


Céa Method
-----------------

**Céa's method proposed in** :cite:`Cea1986` **enables to overcome the calculation of complex shape derivative terms.**

First, a Lagrangian function is introduced and defined as: 

.. math:: 
		:label: eqn:J_ptn_scelle

		\begin{split}
		   \mathcal{L}:\mathcal{O}\times V\times V_{0} & \rightarrow \mathbb{R} \\
		   (\Omega,u,p) & \mapsto \mathcal{L}(\Omega,u,p)=J(\Omega)-a(u,p)+l(p).
		\end{split}

The minimization problems :eq:`eqn:MinCompliance` without equality and inequality constraints, is equivalent to finding the extremum :math:`\left(\Omega_{\text{min}},u_{\Omega_{\text{min}}},p_{\Omega_{\text{min}}}\right)` of the Lagrangian function, solution of the following optimization problem:
several
Find :math:`\left(\Omega_{\text{min}},u_{\Omega_{\text{min}}},p_{\Omega_{\text{min}}}\right)` such that

.. math::

		\left(\Omega_{\text{min}},u_{\Omega_{\text{min}}},p_{\Omega_{\text{min}}}\right):=\underset{\Omega\subset\mathcal{O}}{\min}\underset{p\in V_{0}}{\text{ }\max} \underset{u\in V}{\text{ }\min} \text{ } \mathcal{L}(\Omega,u,p)


For all :math:`\Omega\in \mathcal{O}`, in :math:`u=u_{\Omega}` (solution of equation :eq:`eqn:elasticity_weak_form`), we have: 

.. math::

		\mathcal{L}(\Omega,u_{\Omega},p)=J(\Omega)\mid_{u=u_{\Omega}}\text{, }\forall p\in V_{0}.

The saddle point of the Lagrangian is determined by the following problems:
Find :math:`u_{\Omega}\in V` such that:

.. math::

    		\partial_{p}\mathcal{L}(\Omega,u_{\Omega},p;\varphi)=-a(u_{\Omega},\varphi)+l(\varphi)=0\text{, }\forall\varphi\in V_{0}.

Find :math:`p_{\Omega}\in V_{0}` such that:

.. math::

    		\partial_{u}\mathcal{L}(\Omega,u_{\Omega},p_{\Omega};\psi)=\partial_{u} J(\Omega;\psi)\mid_{u=u_{\Omega}}-a(\psi,p_{\Omega})=0 \text{, }\forall\psi\in V.


According to :eq:`eqn:J_ptn_scelle` and with the definition of the saddle point :math:`\left(u_{\Omega},p_{\Omega}\right)` the shape derivative of cost function in direction :math:`\theta` is written by composition:

.. math::
		\begin{align}
		J'(\Omega)(\theta)&=\mathcal{L}'_{\Omega}(\Omega,u_{\Omega},p_{\Omega};\theta)\\
		&=\partial_{\Omega}\mathcal{L}(\Omega,u_{\Omega},p_{\Omega};\theta)+\underset{=0}{\underbrace{\partial_{u}\mathcal{L}(\Omega,u_{\Omega},p_{\Omega};u_{\Omega,\theta}^{'})}}+\underset{=0}{\underbrace{\partial_{p}\mathcal{L}\left(\Omega,u_{\Omega},p_{\Omega};p_{\Omega,\theta}^{'}\right)}}\\
		&=\partial_{\Omega}J(\Omega)_{\mid u=u_{\Omega}}-\partial_{\Omega}a(u_{\Omega},p_{\Omega})+\partial_{\Omega}l(p_{\Omega}) 
		\end{align}
		
with :

.. math::
		\begin{align}
		u'_{\Omega,\theta}(x)&=\lim_{t\rightarrow0}\frac{u_{\left(\text{Id}+t\theta\right)(\Omega)}(x)-u_{\Omega}(x)}{t} \quad \text{ the eulerian derivative of }u\text{ in direction }\theta\\
		p'_{\Omega,\theta}(x)&=\lim_{t\rightarrow0}\frac{p_{\left(\text{Id}+t\theta\right)(\Omega)}(x)-p_{\Omega}(x)}{t} \quad \text{ the eulerian derivative of }p\text{ in direction }\theta.
		\end{align}


Mechanical model
-----------------------

In our implementation, we consider a linear elastic isotropic material. In the following, we detail the assumptions and equations of the mechanical model.

Material behavior :
~~~~~~~~~~~~~~~~~~~~~~~~

Assuming the material behavior of the domain :math:`\Omega` is linear isotropic elastic, with Hooke's law we have the following relationship between the stress tensor :math:`\sigma` and the strain tensor :math:`\epsilon` :

.. math::

		\sigma = 2\mu\epsilon+\lambda\text{Tr}\left(\epsilon\right)\text{Id}



where :math:`\lambda` and :math:`\mu` are Lamé moduli of the material.

We seek the displacement of the material, :math:`u`, such that :


.. math::
		:label: eqn:elasticity_form

		\begin{align}
		\begin{cases}
			- \text{div} \sigma(u) & \!\!\!\!=0 \text{ in }\Omega\\
			u& \!\!\!\!=0\text{ on }\Gamma_{D}\\
			\sigma(u)\cdot n & \!\!\!\!=g\text{ on }\Gamma_{N}
		\end{cases}
		\end{align}


.. note::

		We assume small deformations and zero volumetric forces. 


Weak formulation of Linear elasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Find :math:`u_{\Omega}\in V(\Omega)=\left\{ u\in\left[H^{1}(\Omega)\right]^{d}\mid u_{\mid\Gamma_{D}}=u_{D}\right\}`, such that :math:`\forall v\in V_{0}(\Omega)=\left\{ u\in\left[H^{1}(\Omega)\right]^{d}\mid u_{\mid\Gamma_{D}}=0\right\}`


.. math::
		:label: eqn:elasticity_weak_form
		
		a\left(u_{\Omega},v\right)=l\left(v\right)

where for all :math:`u\in V(\Omega)` and :math:`v \in V_{0}(\Omega)` :

.. math::
		
		\begin{align} 
		a\left(u,v\right)&=2\mu\left(\varepsilon(u),\varepsilon\left(v\right)\right)_{L^{2}(\Omega)}+\lambda\left(\nabla\cdot u,\nabla\cdot v\right)_{L^{2}(\Omega)}\\
		l\left(v\right)&=\left(g,v\right)_{L^{2}\left(\Gamma_{N}\right)},
		\end{align}
		
		
with :math:`\varepsilon(u)=\frac{1}{2}\left(\nabla u+\nabla^{t}u\right)`. 




Level set method
--------------------
Level set method is used to describe :math:`\Omega` and to capture its evolution. 


Domain definition
~~~~~~~~~~~~~~~~~~~~~
Domain, :math:`\Omega`, is described by a function :math:`\phi:D\rightarrow\mathbb{R}`, which is

.. math::

		\begin{cases}
		\phi(x)<0 & \text{ if }x\in\Omega, \\
		\phi(x)=0 & \text{ if }x\in\partial\Omega, \\
		\phi(x)>0 & \text{ if }x\in D\setminus\overline{\Omega}.
		\end{cases}
		
..  container:: centered-figure

	.. _schema_ls:

	.. figure:: images/demo_compliance/ls.png
		:alt: Exemple d'image
		:align: center
		:width: 50%

		Domain defined by a level-set signed distance function.


There are several level-set functions to define :math:`\Omega`. However, we are interested in level-set functions with signed distance property to address numerical aspects. 

A level set function with signed distance property with respect to :math:`\phi(x)=0` is defined as:

.. math::
		\begin{align}
		\phi(x) =&
		\begin{cases}
		-d\left(x,\Gamma\right) & \text{ if }x\in\Omega,\\
		d\left(x,\Gamma\right) & \text{ if }x\in D\setminus\overline{\Omega},
		\end{cases}
		\end{align}
		
		
where :math:`d` is the euclidean distance function distance defined as: 

.. math::
	
		\begin{align}
		d\left(x,\Gamma\right)=\underset{y\in\Gamma}{\inf}d\left(x,y\right)\text{ with }\Gamma=\left\{ x\in D\text{, such that }\phi(x)=0\right\}.
		\end{align}

Advection
~~~~~~~~~~~~

To advect :math:`\phi` following the velocity field :math:`\theta_{\text{reg}}` (extended and regularized over the whole domain :math:`D`), we solve a transport equation, defined as:

.. math::
		:label: eqn:HJ_equation
		
		\frac{\partial\phi}{\partial t}+\theta_{\text{reg}}\cdot\nabla\phi=0\text{, }\forall t\in\left[0;T\right].

For motion in the normal direction it's equivalent to solve the following equation: 

.. math::

		\frac{\partial\phi}{\partial t}-v_{\text{reg}}\left|\nabla\phi\right|=0\text{, }\forall t\in\left[0;T\right].


.. note::

		In the context of shape optimization, :math:`t` corresponds to a pseudo-time, a descent parameter in the minimization of the objective function.


Instead of solving the Hamilton-Jacobi equation :eq:`eqn:HJ_equation` using the Finite Difference Method, Finite Element Method is used.

For the computation of the temporal derivative, we adopt the explicit Euler method between :math:`0` and :math:`T` (in an arbitrary fixed number of time steps :math:`\Delta t`) :

.. math::
		:label: eqn:HJ_euler
		
		\frac{\phi^{n+1}-\phi^{n}}{\Delta t}-v_{\text{reg}}\left|\nabla\phi^n\right|=0\text{, }\forall t\in\left[0;T\right].

Here, :math:`\phi_{n}` is the previous iterate, and :math:`n` parameterizes :math:`\Omega_{n}`. To solve the minimization problem :eq:`eqn:MinCompliance`, the descent step :math:`\Delta t` of the gradient algorithm is chosen such that: 

.. math::
		:label: eqn:HJ_descent
		
		\mathcal{J}\left(\Omega_{n+1}\right)<\mathcal{J}\left(\Omega_{n}\right).
		
.. note::

	Moreover, in order to verify the stability conditions of the explicit Euler scheme, the time step must satisfy the following Courant-Friedrichs-Lewy (CFL)  condition:

	.. math::

			\Delta t < c \frac{h}{v_{\text{max}}}

	where  :math:`v_{\text{max}}` is the maximum value of the normal velocity and :math:`c\in\left]0,1\right]` is a chosen parameter.
	

Extension and regularization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. note::
	The definition of the descent direction is ambiguous. The field is only defined on the free boundary. Implementing an extension of :math:`v` is necessary to have a velocity field defined over the entire domain :math:`D`. 
	Moreover, the regularity of the :math:`v` field is not sufficient to ensure the mathematical framework of the notion of shape derivative in Hadamard's sense (as the space :math:`L^{2}\left(\Gamma\right)` is not a subspace of :math:`W^{1,\infty}\left(\mathbb{R},\mathbb{R}\right)`), so a regularization is needed.


In our study, extending and regularizing the velocity field involves solving the following problem:
Find :math:`v'_{\text{reg}}\in H_{\Gamma_{D}}^{1}=\left\{ v\in H^{1}\left(D\right)\text{ such that }v=0\text{ on }\Gamma_{D}\right\}` such that :math:`\forall w\in H_{\Gamma_{D}}^{1}`

.. math::
		:label: eqn:reg_velocity
		
		\alpha\left(\nabla v'_{\text{reg}},\nabla w\right)_{L^{2}\left(D\right)}+\left(v'_{\text{reg}},w\right)_{L^{2}\left(D\right)}=-\mathcal{J}'(\Omega)\left(w\right)

with :math:`\mathcal{J}`  the cost function. 


Next, we define the normalized velocity field:

.. math:: 
		:label: eqn:reg_velocity_2
		
		v_{\text{reg}}=\frac{v'_{\text{reg}}}{\sqrt{\alpha\left\Vert \nabla v'_{\text{reg}}\right\Vert _{L^{2}\left(D\right)}+\left\Vert v'_{\text{reg}}\right\Vert _{L^{2}\left(D\right)}}}

This normalization enables the following equality to hold:

.. math::
    
    \left\Vert v_{\text{reg}}\right\Vert _{H_{\Gamma_{D},\alpha}^{1}\left(D\right)}=\sqrt{\alpha\left\Vert \nabla v_{\text{reg}}\right\Vert _{L^{2}\left(D\right)}+\left\Vert v_{\text{reg}}\right\Vert _{L^{2}\left(D\right)}}=1
    
.. note::

		Then, to respect the small deformation hypothesis of the Hadamard method, we multiply by a constant smaller than 1. Alternatively, we can equivalently choose to use an adaptive time step strategy to ensure convergence.

