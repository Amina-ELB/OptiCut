# OptiCut: Open Source Framework for Structural Shape Optimization

> **OptiCut is a cutting-edge tool for designing lightweight and high-performance components. It implements advanced optimization algorithms based on immersed boundary methods for numerical simulations in engineering.**

<p align="center">
<img src="./doc/images/schema_OptiCut.png" alt="Diagram illustrating the OptiCut shape optimization process, which combines Level Set, CutFEM, and the Ersatz material method. The chart shows the evolution of the shape until the optimized solution is reached." width="1000">
</p>

---

## Framework Overview

OptiCut is a **Shape Optimization** framework designed for engineers and researchers in numerical simulation. It provides a powerful solution for structural design by integrating advanced numerical methods.

The technical core of OptiCut combines:

1.  The **Level Set** method for geometry representation and evolution.
2.  The **Cut Finite Element Method (CutFEM)** for solving equations on non-conforming meshes.
3.  The **Ersatz Material** approach for simplified modeling.

This combination allows for efficient handling of complex optimization problems in **mechanical engineering** and **numerical analysis**.

---

## Documentation and Installation

The **full documentation** for OptiCut, including detailed installation instructions, tutorials, and underlying theory, is available on the official website:

**[Access the Complete OptiCut Technical Documentation (Installation, Tutorials, and Theory)](https://amina-elb.github.io/OptiCut/)**

### Dependencies
The current version of OptiCut is implemented using the [CutFEMx](https://github.com/sclaus2/CutFEMx) library.

---

## Demonstration Examples

The framework comes with two main demonstrations illustrating its powerful applications:

* **Structural Compliance Minimization (Stiffness Optimization):**
    * Combined use of the Ersatz and CutFEMx methods to **optimize the overall stiffness** of structures.
    * *SEO Keywords: Compliance Minimization, Structural Design, Stiffness Optimization.*
* **Minimization of the Lᵖ Norm of the von Mises Stress (Strength):**
    * Application of the CutFEMx method to design components with **improved stress resistance** by minimizing stress peaks.
    * *SEO Keywords: Von Mises Stress, L-p Norm Minimization, Strength Optimization.*

---

## Contributing

This project is **Open Source**. Please refer to the "Contribution" section in the documentation for more details.
