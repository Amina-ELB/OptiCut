##############################################################################
###				CODE TOPOLOGY OPTIMIZATION
###	REINITIALIZATION MÉTHOD: EIKONAL EIKONAL equation stabilization supg
###	Velocity extension on all the domain D
###	Advection : use HJ scheme withoud SUPG stabilization
##############################################################################
num_simulation = 0

# ## Implementation
#
# Scaled variable
#Dimension du domaine D
L = lx =  1. #longeur
W = ly =  1. #hauteur
pi = 3.1415


max_i = 10

# The modules that will be used are imported:

# +
import numpy as np

import ufl
# mathematical language for FEM, auto differentiation, python

import matplotlib.pyplot as plt
# meshes, assembly, c++, ython, pybind


from mpi4py import MPI

from typing import TYPE_CHECKING
import pyvista


from mpi4py import MPI

plt.rcParams['text.usetex'] = True
import math
plt.rcParams["font.family"] = "Times New Roman"


with open("0.01/res/lagrangian.txt", 'r') as fic_init:
    b = fic_init.readlines()
    lagrangian001= [float(a.replace('\n',"")) for a in b]
    fic_init.close()
with open("0.001/res/lagrangian.txt", 'r') as fic_init:
    b = fic_init.readlines()
    lagrangian0001= [float(a.replace('\n',"")) for a in b]
    fic_init.close()
with open("0.005/res/lagrangian.txt", 'r') as fic_init:
    b = fic_init.readlines()
    lagrangian0005= [float(a.replace('\n',"")) for a in b]
    fic_init.close()

ite = np.arange(start = 1, stop = (len(lagrangian001)+1))


plt.plot(ite, lagrangian001, 'k-',label =r'$\alpha = 0.01$')
plt.plot(ite, lagrangian0001, 'r-',label =r'$\alpha = 0.001$')
plt.plot(ite, lagrangian0005, 'b-',label =r'$\alpha = 0.005$')
plt.yticks(np.arange(14,23.5,1))
plt.ylabel(r'Value of lagrangian function, $\mathcal{L}\left(\Omega\right)$',fontsize=14)
plt.xlabel(r'iteration step, $n$',fontsize=14)
plt.legend(fontsize=13,loc='upper right')
plt.grid(which="both",linewidth=0.1)
plt.savefig("lagrangian_compared.pdf",bbox_inches="tight")
# # plt.savefig("error_eikonal_N_star.pdf",bbox_inches="tight")
plt.close()
