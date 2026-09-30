---
title: 'OptiCut: A Level-Set and CutFEM based Shape Optimization Framework in FEniCSx'
tags:
  - Python
  - shape optimization
  - topology optimization
  - CutFEM
  - level-set method
  - FEniCSx
  - finite element method
authors:
  - name: Amina El Bachari
    orcid: 0000-0000-0000-0000 # To be updated
    affiliation: "1, 2"
affiliations:
 - name: ONERA - The French Aerospace Lab, France
   index: 1
 - name: MINES Paris - PSL University, Centre des Matériaux, France
   index: 2
date: 29 September 2026
bibliography: paper.bib
---

# Summary

Structural shape and topology optimization are critical fields in computational mechanics, enabling the design of lightweight and highly performant components. Implicit boundary tracking via the Level-Set method [@Osher1988] has emerged as a powerful technique to manage complex topological changes without requiring explicit mesh morphing. However, imposing boundary conditions accurately on immersed interfaces cutting through background elements remains a major challenge. 

`OptiCut` is a robust, open-source Python framework designed to solve structural shape optimization problems using the Level-Set method combined with the Cut Finite Element Method (CutFEM) [@Burman2015]. Built on top of the modern FEniCSx ecosystem [@Scroggs2022] and the open-source `CutFEMx` extension (https://github.com/sclaus2/CutFEMx), `OptiCut` completely decouples the computational mesh from the evolving geometry. It provides an end-to-end pipeline—from geometry initialization to shape derivatives computation, advection, and reinitialization—all seamlessly integrated into an automated Augmented Lagrangian Method (ALM) optimizer.

# Statement of need

Traditional topology optimization methods (such as SIMP) often suffer from "gray" interface zones and checkerboard instabilities, leading to designs that are difficult to manufacture directly. The Level-Set method solves the crisp boundary problem but introduces the difficulty of integrating physics over cells that are only partially filled with material.

Historically, the "Ersatz material" approach has been used to penalize void regions by assigning them a near-zero stiffness. While easy to implement, this method suffers from severe ill-conditioning and boundary inaccuracies, especially for high-order stress constraints. 

`OptiCut` addresses these shortcomings by offering a fully integrated CutFEM solver. By employing Nitsche's method for Dirichlet conditions and ghost penalty stabilization, CutFEM achieves optimal convergence rates and exact boundary representations on a fixed background mesh. `OptiCut` makes these advanced numerical techniques highly accessible to researchers and engineers. It supports:
- **Cost Functions:** Compliance, Area, and $L^p$-norm Von Mises stress minimization.
- **Solvers:** Both classical Ersatz and state-of-the-art CutFEM solvers for comparative studies.
- **Optimization:** Augmented Lagrangian Method handling equality and inequality constraints.
- **Level-Set Tools:** Hamilton-Jacobi based velocity extension, advection, and reinitialization.

`OptiCut` fills a significant gap in the FEniCSx community by providing a modern, scalable, and heavily tested framework specifically tailored for shape optimization with unfitted finite elements.

# Acknowledgements

This software was developed in collaboration between ONERA and MINES Paris - PSL University.

# References
