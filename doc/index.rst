OptiCut Documentation
=====================

..  container:: centered-figure

	.. figure:: images/opticut-logo-v2.svg
		:alt: OptiCut Logo
		:align: center
		:width: 70%

.. raw:: html

    <div style="text-align: center; margin-bottom: 30px; color: #333;">
        <span style="font-size: 1.1em;">Amina El Bachari</span>
    </div>

OptiCut is an open-source Python framework for structural shape optimization, combining the **Level-Set** method with the **Cut Finite Element Method** (CutFEM) and the **Ersatz material** approach. Built upon the `FEniCSx <https://fenicsproject.org/>`_ ecosystem and the `CutFEMx <https://github.com/sclaus2/CutFEMx>`_ extension (developed by S. Claus), it completely decouples the computational mesh from the evolving geometry. Natively parallelized using MPI, OptiCut is designed for High-Performance Computing (HPC) environments in computational solid mechanics.

.. raw:: html

    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px; margin-bottom: 20px;">
        <div style="width: 45%;">
            <video width="100%" autoplay loop muted controls style="border: 1px solid #ccc;">
                <source src="_static/compliance.mp4" type="video/mp4">
            </video>
            <div style="text-align: center; margin-top: 5px; font-style: italic; color: #555; font-size: 0.9em;">
                Fig 1. <a href="demo_compliance.html">Compliance Minimization</a>
            </div>
        </div>
        <div style="width: 45%;">
            <video width="100%" autoplay loop muted controls style="border: 1px solid #ccc;">
                <source src="_static/vonMises.mp4" type="video/mp4">
            </video>
            <div style="text-align: center; margin-top: 5px; font-style: italic; color: #555; font-size: 0.9em;">
                Fig 2. <a href="demo_vonMises.html">Von Mises Minimization (Lp Norm)</a>
            </div>
        </div>
    </div>

This documentation is organized into the following main sections:

* **Getting Started**: Installation instructions and a quick start guide to run your first optimization.
* **User Guide**: Detailed instructions on defining problems, setting parameters, and configuring boundary conditions.
* **Tutorials**: Step-by-step benchmark problems illustrating the application and effectiveness of OptiCut.
* **Theory**: A comprehensive overview of the mathematical formulation, including shape optimization, CutFEM, and the Ersatz material method.
* **Developer Guide**: Documentation on the software architecture and instructions for extending the code with new physics.
* **API Reference**: Complete technical documentation of the codebase.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting_started/installation.md
   getting_started/quickstart.md

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   user_guide/problem_definition.md
   user_guide/parameters.md

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   demos.rst

.. toctree::
   :maxdepth: 2
   :caption: Theory
   :hidden:

   demo_optim.rst
   demo_cutfem.rst
   demo_cutfem_optim.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   developer/architecture.md
   developer/extending.md

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   documentation.rst

.. toctree::
   :maxdepth: 1
   :caption: References
   :hidden:

   bibliography.rst
