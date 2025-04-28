.. CutOptim documentation master file, created by
   sphinx-quickstart on Mon Feb  3 11:51:37 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OptiCut documentation
========================
..  container:: centered-figure

	.. figure:: images/schema_OptiCut.png
		:alt: Logo
		:align: center
		:width: 100%


OptiCut is a shape optimization framework that combines the **level set** method with the **Cut Finite Element Method** (CutFEM) and the **Ersatz material** approach.

Part :ref:`demoOptim` provides a comprehensive overview of the shape optimization strategy employed, including the mathematical formulation and general methodology.

In Part :ref:`demoCutfem`, we introduce the CutFEM method in a broad sense, emphasizing its fundamental principles and comparing it to the classical fictitious material method, particularly in the context of shape optimization.

Part :ref:`demoCutfemOptim` focuses on the specific adaptations of the CutFEM approach for shape optimization problems for linear elasticity. 

These first three sections together offer a theoretical foundation for understanding the inner workings and design principles of the OptiCut code.

The :ref:`demos` section presents two benchmark problems that illustrate the application and effectiveness of the method, demonstrating the performance and capabilities of OptiCut in practical scenarios.

Finally, the last section :ref:`documentation` contains the complete technical documentation of the code, including usage instructions, input/output specifications, and implementation details.


.. toctree::
   :maxdepth: 3
   :caption: Contents:
   
   ./demo_optim.rst
   ./demo_cutfem.rst
   ./demo_cutfem_optim.rst
   ./demos.rst
   ./documentation.rst
   ./bibliography.rst
