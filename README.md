# OptiCut: Open Source FEniCSx Framework for Parallel Structural Shape Optimization

> **OptiCut is a cutting-edge, parallel-enabled FEniCSx tool for designing lightweight and high-performance components. It implements advanced optimization algorithms based on immersed boundary methods for numerical simulations in engineering.**

<p align="center">
<img src="./doc/images/schema_OptiCut.png" alt="Diagram illustrating the OptiCut shape optimization process, which combines Level Set, CutFEM, and the Ersatz material method within a parallel FEniCSx framework. The chart shows the evolution of the shape until the optimized solution is reached." width="1000">
</p>

---

## Framework Overview

OptiCut is a **Structural Shape Optimization** framework designed for engineers and researchers in advanced numerical simulation. It is built upon the **FEniCSx** project, leveraging its capabilities for efficient, **parallel computing**.

It provides a powerful solution for structural design by integrating advanced numerical methods:

1.  The **Level Set** method for geometry representation and evolution.
2.  The **Cut Finite Element Method (CutFEM)** for solving equations on non-conforming meshes, compatible with parallel execution.
3.  The **Ersatz Material** approach for simplified modeling.

This combination allows for efficient handling of complex, large-scale optimization problems in **mechanical engineering** and **numerical analysis**.

---

## Key Features and Extensibility

OptiCut is designed with **modularity** and **adaptability** in mind, making it a powerful and flexible tool:

* **3D Capability:** The core implementation is fully compatible with **3D geometries** and simulations, enabling the optimization of complex, real-world structural components.
* **Problem Portability:** The design includes a dedicated **Problem Class**. By modifying or extending this single class, users can easily adapt the framework to solve **other optimization problems** (beyond compliance and stress minimization). This modular structure ensures **high code portability** and reuse.
* **Parallel Computing:** Leverages **FEniCSx** for robust performance on large-scale problems and distributed memory systems.

---

## Documentation and Installation

The **full documentation** for OptiCut, including detailed installation instructions, parallel execution guidance, tutorials, and underlying theory, is available on the official website:

**[Access the Complete OptiCut Technical Documentation (Installation, Tutorials, and Theory)](https://amina-elb.github.io/OptiCut/)**

### Dependencies
The current version of OptiCut is implemented in parallel using the **FEniCSx** environment and relies on the [CutFEMx](https://github.com/sclaus2/CutFEMx) library.

---

## Demonstration Examples

The framework comes with two main demonstrations illustrating its powerful applications:

* **Structural Compliance Minimization (Stiffness Optimization):**
    * Utilizes the Ersatz and CutFEMx methods within a parallel setup to **optimize the overall stiffness** of structures.
* **Minimization of the Lᵖ Norm of the von Mises Stress (Strength):**
    * Applies the CutFEMx method to design components with **improved stress resistance** by minimizing stress peaks.

---

## Contributing

This project is **Open Source**. Please refer to the "Contribution" section in the documentation for more details.
