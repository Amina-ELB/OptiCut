# Copyright (c) 2025 ONERA and MINES Paris, France 

# All rights reserved.
#
# This file is part of OptiCut.
#
# Author(s)     : Amina El Bachari 

from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from random import *
import math


class Parameters:
    """This is the Parameters class.

    :param str name: The Parameters's name.
    """
    def __init__(self, cost_func=0, tol_cost_func = 0, descent_direction_Riesz=0, ALM_lagrangian_multiplicator=0, max_incr=0,\
    lx=0, ly=0, h=0, young_modulus=0, poisson=0, strenght=0, adapt_time_step=0, dt=0, j_max=0, extend_velocity=0,\
    vel_normalization=0, alpha_reg_velocity=0,step_reinit=0, freq_reinit=0, l_reinit=0,\
    cutFEM=0, cut_fem_advection=0, eta=0, elasticity_limit = 0, p_const = 0):
        self.cost_func = cost_func
        self.tol_cost_func = tol_cost_func
        self.p_const = p_const
        self.descent_direction_Riesz = descent_direction_Riesz
        self.ALM_lagrangian_multiplicator=ALM_lagrangian_multiplicator
        self.max_incr=max_incr
        
        self.lx=lx
        self.ly=ly
        self.h=h
        
        self.young_modulus=young_modulus
        self.elasticity_limit = elasticity_limit
        self.poisson=poisson
        self.strenght=strenght
        
        self.adapt_time_step=adapt_time_step
        self.dt=dt
        self.j_max=j_max
        
        self.extend_velocity=extend_velocity
        self.vel_normalization = vel_normalization
        self.alpha_reg_velocity=alpha_reg_velocity
        self.step_reinit=step_reinit	
        self.freq_reinit=freq_reinit
        self.l_reinit=l_reinit
        
        self.cutFEM=cutFEM
        self.cut_fem_advection=cut_fem_advection
        self.eta = eta
        
    def set__paramManually(self):
        print("#################################################")
        print("                Mecanics Parameters:             ")
        print("#################################################")
        print("Young modulus ?  ")
        self.young_modulus = float(input())
        print("#################################################")
        print("Elasticity limit ?  ")
        self.elasticity_limit = float(input())
        print("Poisson coefficient ?  ")
        self.poisson = float(input())
        print("Value of strenght ?  ")
        self.strenght =  float(input())	
        
        print("#################################################")
        print("             Optimization Parameters:            ")
        print("#################################################")
        print("What's the cost funciton to minimize ? (write 'compliance' or 'VonMises')")
        self.cost_func =  input()
        if self.cost_func=='VonMises':
            print("What is the value of p constant")
            self.p_const = int(input())
        print("Use of gradient descent (1), or not (0)")
        self.descent_direction_Riesz =  int(input())
        print("Value of tolerence for compliance criterion ?  ")
        self.tol_cost_func =  float(input())	
        print("Value of ALM_lagrangian_multiplicator ?  ")
        self.ALM_lagrangian_multiplicator =  float(input())		
        print("Number of maximal increment ? ")		
        self.max_incr = int(input())
        
        print("#################################################")
        print("           Structure/Mesh Parameters:            ")
        print("#################################################")
        print("Length ?")
        self.lx = float(input())
        print("Width ?")
        self.ly = float(input())
        print("Mesh size (h) ?")
        self.h = float(input())

        print("#################################################")
        print("           Method Parameters:            ")
        print("#################################################")
        print("Advection method ?")
        print("0: if advection with SUPG stabilization")
        print("1: if simple advection with cutFEM")
        self.cut_fem_advection = int(input())

        print("Adapt time step ?")
        print("0: if no adptation time step")
        print("1: id adapt time step")
        print("veuillez entrer le nombre de simulation que vous souhaitez lancer")
        self.adapt_time_step = int(input())
        
        print("Velocity's extension method ?")
        print("0: if no extension")
        print("1: if extension with resolution of PDE")
        self.extend_velocity = int(input())
        if self.extend_velocity==1:
                print("Value of parameters of regularization of velocity field?")
                self.alpha = float(input())
        print("Velocity's normalization ?")
        print("0: if no normalization")
        print("1: if normalization")
        self.vel_normalization = int(input())
        
        print("Optimization using CutFEM ? :")
        print("1: if optimization using CutFEM method")
        print("0: if not")
        self.cutFEM = int(input())
        
        print("Value of eta in xsi function ? :")
        self.eta = float(input())
        
        print("Reinitialization method ?")
        print("1: 1: HJ stabilized with Ghost penalty")
        print("0: HJ explicit not stabilized")
        print("2: HJ with SUPG stabilization")
        self.cut_fem_advection = int(input())
        
        print("l reinitialization parameter ?(advice l<-lx+ly)")
        self.l = float(input())
               
        print("Frequency of reinitialization ?")
        print("(ie: 1 means reinitialization each incement of advection \n 2 means reainitialization each one increment over 2 ...)")
        self.freq_reinit = int(input())
        print("Number of iteration of reinitialization's scheme?")
        self.step_reinit = int(input())

        
        
#Fonction non utilisé mais fonctionnelle
   
    def set__paramFolder(self,folder_name):
        """Initialization of an object Parameters with a file.

        :param char folder_name: The name of the file with all the parameters data
        
        :returns: The object Parameters.
        :rtype: Parameters
        
        """
        filin = open(folder_name, "r")
        lines = filin.readlines()
        for line in lines:
            word = line.split()          		
            if (word[0]=='ALM_lagrangian_multiplicator') : 
                self.ALM_lagrangian_multiplicator = float(word[1])
            elif(word[0]=='cost_func') : 
                self.cost_func = word[1]
            elif(word[0]=='augmented_lagrangian') : 
                self.augmented_lagrangian = int(word[1])
            elif(word[0]=='uzawa') : 
                self.uzawa = int(word[1])
            elif(word[0]=='ALM') : 
                self.ALM = int(word[1])
            elif(word[0]=='ALM_lagrangian_multiplicator') : 
                self.ALM_lagrangian_multiplicator = float(word[1])
            elif(word[0]=='ALM_penalty_parameter') : 
                self.ALM_penalty_parameter = float(word[1])
            elif(word[0]=='ALM_penalty_limit') : 
                self.ALM_penalty_limit = float(word[1])
            elif(word[0]=='ALM_slack_variable') : 
                self.ALM_slack_variable = float(word[1])
            elif(word[0]=='ALM_penalty_coef_multiplicator') : 
                self.ALM_penalty_coef_multiplicator = float(word[1])
            elif(word[0]=='type_constraint'):
                self.type_constraint = word[1]
            elif(word[0]=='constraint') : 
                self.constraint = word[1]
            elif(word[0]=='target_constraint') : 
                self.target_constraint = float(word[1])
            elif(word[0]=='p_const'):
                self.p_const = int(word[1])
            elif(word[0]=='descent_direction_Riesz') : 
                self.descent_direction_Riesz = int(word[1])
            elif(word[0]=='tol_cost_func') : 
                self.tol_cost_func = float(word[1])
            elif(word[0]=='max_incr') : 
                self.max_incr= int(word[1])
            elif (word[0]=='lx') :
                self.lx= float(word[1])
            elif (word[0]=='ly') :
                self.ly= float(word[1])
            elif (word[0]=='lz') :
                self.lz= float(word[1])
            elif(word[0]=='h') : 
                self.h= float(word[1])
            elif (word[0]=='young_modulus') :
                self.young_modulus= float(word[1])
            elif (word[0]=='elasticity_limit') :
                self.elasticity_limit= float(word[1])
            elif (word[0]=='poisson') :
                self.poisson= float(word[1])
            elif(word[0]=='strenght') : 
                self.strenght= float(word[1])
            elif (word[0]=='adapt_time_step') :
                self.adapt_time_step= int(word[1])
            elif(word[0]=='dt') : 
                self.dt= float(word[1])
            elif (word[0]=='j_max') :
                self.j_max= int(word[1])
            elif (word[0]=='extend_velocity') :
                self.extend_velocity= int(word[1])
            elif (word[0]=='vel_normalization') :
                self.vel_normalization= int(word[1])   
            elif (word[0]=='alpha_reg_velocity') :
                self.alpha_reg_velocity= float(word[1])
            elif(word[0]=='step_reinit') : 
                self.step_reinit= int(word[1])
            elif (word[0]=='freq_reinit') :
                self.freq_reinit= int(word[1])
            elif (word[0]=='l_reinit') :
                self.l_reinit= float(word[1])
            elif(word[0]=='cutFEM') : 
                self.cutFEM= int(word[1])
            elif(word[0]=='eta') : 
                self.eta= float(word[1])
            elif (word[0]=='cut_fem_advection') :
                self.cut_fem_advection= int(word[1])
        filin.close()
