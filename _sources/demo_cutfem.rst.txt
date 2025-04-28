.. _demoCutfem:

CutFEM for Immersed geometry discretization
==============================================
The mathematical context of CutFEM method including a detailed description of the implementation aspects is explains in this section.



Definition of mesh parts
--------------------------
The fixed working domain :math:`D` is discretized using a triangular or tetrahedral mesh denoted by :math:`\mathcal{K}_{\;h}`. Let :math:`K` denote the elements in the mesh, let :math:`F` denote the facets in the mesh (i.e. edges in 2D and faces in 3D) and :math:`h` be the element size. Let :math:`\mathcal{F}_{\;h}` denote the set of all facets in the mesh. 
 
We define the set of active elements of the mesh and the set of active facets of the mesh as all 
elements or facets that have at least a part in :math:`\Omega` 

.. math::
		:label: eqn:9

		\mathcal{K}_{\;h,\Omega}:=\left\{ K\in\mathcal{K}_{\;h}\mid\overline{K}\cap\Omega\neq\emptyset\right\} \text{ and }  \mathcal{F}_{h,\Omega}=\left\{ F\in\mathcal{F}_{h}\mid F\cap\Omega\neq\emptyset\right\}.  


The union of all active elements forms the active part of the mesh denoted by

.. math::
		:label: eqn:18
		
		\Omega_{h}=\cup_{K\in\mathcal{K}_{\;h,\Omega}}K.

Note that, :math:`\Omega_{h}` is in general not aligned with :math:`\partial\Omega`. It consists of the smallest set of elements that contains the domain as illustrated by pink elements in :numref:`schemaCutfemFinal`.

Furthermore, we define a set of facets of intersected elements, which will be used for stabilization terms

.. math::
		:label: eqn:10

		\mathcal{F}_{h,G,\Omega}:=\left\{ F\in\mathcal{F}_{h,\Omega}\mid F\cap\overline{\Omega}_{h,N}\neq\emptyset\right\} \text{ where  } \Omega_{h,N}:=\underset{\overline{K}\cap\Gamma \neq\emptyset\text{, }K\in\mathcal{K}_{h,\Omega}}{\cup}K 



Note that these facets are intersected facets as well as facets contained fully inside of :math:`\Omega` belonging to intersected elements. 

..  container:: centered-figure

	.. _schemaCutfemFinal:

	.. figure:: images/cutfem_demo/schema_cutfem_final.png
		:alt: Exemple d'image
		:align: center
		:width: 70%

		Illustration of CutFEM domain and face definitions.



For each face :math:`F\in\mathcal{F}_h`, we denote by :math:`K_{+}` and :math:`K_{-}` the two elements shared by :math:`F` and :math:`n_{F}=n_{\partial K_{+}}` the normal to the face pointing from :math:`K_{+}` to :math:`K_{-}`.

 
Interface discretization
-----------------------------
Given a level set function to describe our evolving domain :math:`\Omega`, we mainly distinguish between the following two immersed interface discretizations: 

- a smooth interface representation
- a sharp interface representation.

Smooth interface representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The boundary of our domain is represented by a smeared out or smoothed interface. The domain described by negative level set values and the domain described by positive level set values are blended by a smoothed Heaviside function :math:`H_{\eta}`. To apply interface terms, the derivative of the Heaviside function is taken to obtain a smoothed delta function. This smoothing approach entails two main drawbacks. Firstly, the smoothed interface region requires resolution with a large number of elements. Secondly, smearing out of the interface reduces accuracy. In this article, we use the following approximation of the characteristic function proposed in :cite:`allaire_book`:

.. math::
		:label: eqn:smooth_heaviside
		
		\forall x\in D\text{, }\chi(x)\simeq H_{\eta}\left(\phi(x)\right)
		\quad \text{ where } \quad 
		H_{\eta}\left(t\right)=\frac{1}{2}\left(1-\frac{t}{\sqrt{t^{2}+\eta^{2}}}\right),


Sharp interface representation: 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a sharp interface representation, the zero contour line of the level set function is reconstructed and used for integration. A sharp interface representation can either by achieved by re-meshing (:cite:`dapogny2023shape`, :cite:`ALLAIRE201422`) to align the mesh with the zero contour line or to "cut" along the zero contour line. The drawback of re-meshing is the cost and complexitiy of re-meshing (especially for parallel computations and 3D) and a major drawback in cutting is the occurrence of "bad" cuts which require stabilization.



Discretization of outside domain :math:`D \setminus \Omega`
---------------------------------------------------------------

We introduce two main approaches to approximate the material outside of the material domain :math:`\Omega`, i.e. :math:`D \setminus \Omega`. 

.. _demoErsatz:

Ersatz material method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Proposed in :cite:`ALLAIRE2004363`, an Ersatz material with small material parameters is introduced in :math:`D \setminus \Omega` to extend the problem formulation from domain :math:`\Omega` to the fixed working domain :math:`D`. Using small material parameters instead of setting material parameters to zero outside of :math:`\Omega` avoids singularities in the stiffness matrix. In practice, a characteristic function :math:`\chi(x)\in\left\{0;1\right\}`, representing the presence or absence of soft material is defined for each :math:`x\in D`. Then we can construct an extension of the elasticity tensor on the entire domain :math:`D`:

.. math::

		A_{\chi}(x):=\alpha\chi(x)+\beta\left(1-\chi(x)\right)\text{ for all }x\in D   

where :math:`\beta` is the elasticity tensor of the solid material and :math:`\alpha=\varepsilon\beta` is the elasticity tensor of the soft material. We set :math:`\varepsilon=10^{-3}`. :math:`\chi(x)` is a characteristic function defined as :

.. math::

		\chi(x) = 
		\begin{cases}
		1 &\mbox{, if } x \in \Omega,\\
		0 &\mbox{, elsewhere}.
		\end{cases}

We use the smoothened characteristic function :eq:`eqn:smooth_heaviside`. 
The Ersatz material method is simple to apply and to implement but it may suffer from a lack of precision in the calculation of mechanical fields, particularly for coarse meshes.


Zero extension outside of active mesh (Deactivation)  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An alternative to using the Ersatz material method is to set all unknowns to zero outside of :math:`\Omega`. In practice, we set the unknowns, in our case the displacement, to zero in all degrees of freedom which do not belong to the active mesh.  

Cutting and Integration
-----------------------------

In our CutFEM approach, we represent the boundary as a sharp interface. As such, we reconstruct the zero level set contour line by computing linear cuts through the mesh elements. We then integrate on those resulting cuts (volume and boundary cuts) as detailed below. For simplicity explanations are given in detail for dimension two.  As mentioned above, we use a level set function, :math:`\phi`, to describe :math:`\Omega`. We approximate :math:`\phi` in a finite element space by interpolation, which we denote as :math:`\phi_h`. This means, using linear Lagrangian elements, :math:`\phi_h` has one value in each vertex of the mesh, which we then use for cutting.

Element Categorization   
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the first step, the sign of the level-set function is checked in each node of the element to determine if the element is intersected by the interface (changing sign), within the domain (:math:`\phi_h(x_i)\leq0, \forall i`), or outside (:math:`\phi_h(x_i)>0, \forall i`). An example is given where the elements in  :numref:`triangleCut` is marked as intersected by the interface, the element in :numref:`triangleIn` is marked as an element of the active domain and the element in  :numref:`triangleOut` is marked as element outside of the active domain.


Interface Reconstruction and Cutting    
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To reconstruct the interface in order to integrate over :math:`\Gamma_K` and :math:`\Omega_K = \Omega_h \cap K`, a simple linear interpolation of level set values is used to calculate the points of intersection between the edges of the element :math:`K` and the interface. Then we use these intersection points to obtain a linear approximation of :math:`\Gamma_K` and :math:`K \cap \Omega_h` (see :numref:`submesh1` and :numref:`submesh2`). 
  



.. container:: images-row

	..  container:: centered-figure

		.. _triangleCut:   
		
		.. figure:: images/cutfem_demo/triangle_cut.png
			:width: 100%
			:align: left

			Element cut by the interface. 


	..  container:: centered-figure
 
		.. _triangleIn:   
			
		.. figure:: images/cutfem_demo/triangle_in.png
			:width: 100%
			:align: center
			
			Element in the active domain.   


	..  container:: centered-figure

		.. _triangleOut:   
		
		.. figure:: images/cutfem_demo/triangle_out.png
			:width: 100%
			:align: right
			

			Element outside the active domain.   







.. container:: images-row

	..  container:: centered-figure

		.. _submesh1:   
		
		.. figure:: images/cutfem_demo/subtriangle_1.png
			:width: 100%
			:align: left

			Triangle cut by the boundary :math:`\Gamma`. 
			


	..  container:: centered-figure

		.. _submesh2:   
				
		.. figure:: images/cutfem_demo/subtriangle_2.png 
			:width: 70%
			:align: center

			Sub-triangulation construction. 


Sub-integration for volume integral 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


We generate the quadrature rules over :math:`\Omega_K` by sub-triangulation. For straight cuts in a triangle, this means either a triangle part or a quadrilateral part which we split into 2 sub-triangles. :numref:`submesh2` shows an example of the sub-triangulation of a triangle element :numref:`submesh1`. 
To generate the integration rule, we map the standard integration rule from a standard reference element onto a cut reference element. 
We define a mapping between the quadrature rule on the reference element and the sub-elements :math:`K_1` and :math:`K_2`, which transforms the quadrature rule on the reference element to a quadrature rule on :math:`K_1` and :math:`K_2` elements. Note that this means we now need to evaluate shape functions and their derivatives in non-standard points inside the reference element (see :numref:`mappingIntegration`).

..  container:: centered-figure

	.. _mappingIntegration:

	.. figure:: images/integration/mapping_integration.png
		:align: center
		:width: 70%

		Illustration of mapping corresponding to sub-triangulation.
	

Sub-integration for interface (or surface) integral     
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To integrate over the interface (or surface) parts :math:`\Gamma_K`, we use a similar technique to generate the quadrature rules. The important difference is here that the element used to generate the quadrature rule has a different dimension toe the reference element in which the quadrature rule is used. This means, we interpret the interface represented by an interval reference element as part of the triangular element. We map the integration rule from the standard reference interval (1D) to part of the triangle (2D). This mapping is mixed dimensional to generate the quadrature rule. An example of the approximation of the :math:`\Gamma_K` integral as part of :math:`K` is given in :numref:`mappingIntegrationFacet`.



..  container:: centered-figure

	.. _mappingIntegrationFacet:


	.. figure:: images/integration/mapping_integration_facet.png
		:align: center
		:width: 70%

		Illustration of mapping corresponding to sub-integration over :math:`\Gamma`.

	
We will henceforth refer to these integrals as cut integrals and their mesh parts as cut meshes. 
