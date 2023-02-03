# -*- coding: utf-8 -*-
"""
Created on Tue May 18 16:32:48 2021

@author: jo28dohe
"""

### testing around with sp and transfer matrices

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
import sympy as sp
from sympy.utilities.lambdify import lambdify,implemented_function
sp.init_printing()
import timeit
import sys
import refractive_index.refractive_index as ri

n=sp.Symbol('n')
l=sp.Symbol('l')
lamb=sp.Symbol('lamb')
n1=sp.Symbol('n1')
n2=sp.Symbol('n2')
n3=sp.Symbol('n3')

n_SiC=2.6   ### bei 800nm https://refractiveindex.info/?shelf=main&book=SiC&page=Wang-4H-o
n_SiO2=1.45   ##https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson


n_Au=ri.n_from_string('Au') 
n_Al=ri.n_from_string('Al') 
n_MoS2_ML=ri.n_from_string('MoS2').get_n(600e-9)
print(n_MoS2_ML)

d_Al=3.5e-9
d_MoS2=0.615e-9
d_SiO2=135e-9
d_vac=2300e-9
d_Au=50e-9

def ref_E(n_1,n_2):
    return((n_1-n_2)/(n_1+n_2))
    
def trans_E(n_1,n_2):
        return(2*n_1/(n_1+n_2))

def P__(n,l,lamb):    
    return sp.simplify(sp.Matrix([[sp.exp(-1j*2*sp.pi*l*n/lamb),0],[0,sp.exp(+1j*2*sp.pi*l*n/lamb)]]))

def D__(n_1,n_2):
    return sp.simplify(1/trans_E(n_1,n_2)*sp.Matrix([[1,ref_E(n_1,n_2)],[ref_E(n_1,n_2),1]]))

def t_from_M__(M): 
    t=1/M[0,0]
    return(t)
    
def r_from_M__(M):
    r=M[1,0]/M[0,0]
    return(r)

def find_in_structure(d_list, distance):
    """
    d_list is list of thicknesses of layers, all of which are finite.

    distance is the distance from the front of the whole multilayer structure
    (i.e., from the start of layer 0.)

    Function returns [layer,z], where:

    * layer is what number layer you're at.
    * z is the distance into that layer.

    For large distance, layer = len(d_list), even though d_list[layer] doesn't
    exist in this case. For negative distance, return [-1, distance]
    """
    if sum(d_list) == np.inf:
        raise ValueError('This function expects finite arguments')
    if distance < 0:
        return [-1, distance]
    layer = 0
    while (layer < len(d_list)) and (distance >= d_list[layer]):
        distance -= d_list[layer]
        layer += 1
    return [layer, distance]

def find_in_structure_with_inf(d_list, distance):
    """
    d_list is list of thicknesses of layers [inf, blah, blah, ..., blah, inf]

    distance is the distance from the front of the whole multilayer structure
    (i.e., from the start of layer 1.)

    Function returns [layer,z], where:

    * layer is what number layer you're at,
    * z is the distance into that layer.

    For distance < 0, returns [0, distance]. So the first interface can be described as
    either [0,0] or [1,0].
    """
    
    print(d_list)
    print(distance)
    if distance < 0:
        return [0, distance]
    [layer, distance_in_layer] = find_in_structure(d_list[1:-1], distance)
    return [layer+1, distance_in_layer]



class sympy_tmm():
    
    """
    an object which corresponds to a layered structure and contains functions for optical parameters
    like Reflectance, Transmission and depth-dependant E-Field
    """

    def __init__(self,n_value_list=[n_SiO2,n_Al,n_SiO2,n_MoS2_ML,1,n_Au,n_SiC],
                d_value_list=[np.inf, d_Al,d_SiO2,d_MoS2,d_vac,d_Au, np.inf],
                d_parameter_list=[4],
                n_parameter_list=[1,5]):
        
        assert len(n_value_list)== len(d_value_list), 'n_value_list and d_value_list must have same lenght!'
        assert len(n_value_list)>1, 'n_value_list must have more than 1 entry'

        self.n_value_list=n_value_list
        self.d_value_list=d_value_list
        self.d_parameter_list=d_parameter_list
        self.n_parameter_list=n_parameter_list
        


        self.n_list=[]
        self.d_list=[]
        self.lamb=sp.Symbol('lamb')
        self.all_parameter_list=[self.lamb]
       
        
        for i in range(len(n_value_list)):
            if i in n_parameter_list:
                self.n_list.append(sp.Symbol('n'+str(i)))
                self.all_parameter_list.append(self.n_list[-1])
            else:
                self.n_list.append(n_value_list[i])
            if i in d_parameter_list:
                self.d_list.append(sp.Symbol('d'+str(i)))
                self.all_parameter_list.append(self.d_list[-1])
            else:
                self.d_list.append(d_value_list[i])
        
        
        
        print(self.n_list,self.d_list)
        ## initialize Transfer Matrix at last boundary:
        self.M__=D__(self.n_list[-2],self.n_list[-1])
        ## save list that contains Transfer matrices of all boundaries from last to first (for E-field calculation)
        self.M__list=[self.M__]
        
        for i in range(len(self.n_value_list)-2):
           self.M__=D__(self.n_list[-3-i],self.n_list[-2-i])*P__(self.n_list[-2-i],self.d_list[-2-i],lamb)*self.M__
           self.M__list.append(self.M__)
        print('Matrix_Multiplication_done')
        
        
          
        self.r=r_from_M__(self.M__)
        self.t=t_from_M__(self.M__)
        self.R=sp.Abs(self.r)**2
        self.T=sp.Abs(self.t)**2*self.n_value_list[-1].real/self.n_value_list[0].real
        
        ## make list of arguments which should be plugged in later
        self.lambdify_args=[self.lamb]
        for index in self.n_parameter_list:
            self.lambdify_args.append(self.n_list[index])
        for index in self.d_parameter_list:
            self.lambdify_args.append(self.d_list[index])
        
        ## lambdify functions for numpy-like evaluation
        self.lam_R=lambdify(self.lambdify_args,self.R)    
        self.lam_T=lambdify(self.lambdify_args,self.T)
        
        print('Total M__, lam_R, lam_T created, all_parameter_list: ',self.all_parameter_list)

    def plot_R_T(self,lamb_min,lamb_max,nlamb=200,d_vac=d_vac):
        lamb_list=np.linspace(lamb_min,lamb_max,nlamb)
        params=n_Al.get_n(lamb_list),n_Au.get_n(lamb_list),d_vac
        plt.plot(lamb_list,self.lam_R(lamb_list,*params),label='R')
        plt.plot(lamb_list,self.lam_T(lamb_list,*params),label='T')
        plt.plot(lamb_list,self.lam_R(lamb_list,*params)+self.lam_T(lamb_list,*params),label='Sum')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def plot_R_d(self,lamb_min,lamb_max,d_min,d_max,n_lamb=200,n_d=200):
        lamb_list=np.linspace(lamb_min,lamb_max,n_lamb)
        d_list=np.linspace(d_min,d_max,n_d)
        plt.figure()
        for i in range(len(d_list)):
            plt.scatter(lamb_list,np.ones(len(lamb_list))*d_list[i],c=self.lam_R(lamb_list,n_Al.get_n(lamb_list),n_Au.get_n(lamb_list),d_list[i]),
                        vmin=0,vmax=1,cmap=cm.nipy_spectral)
        plt.xlim(min(lamb_list),max(lamb_list))
        plt.ylim(min(d_list),max(d_list))
        plt.show()
        
        
    def R_fit_func(self,d,d2,lamb_list):
        R=self.lam_R(lamb_list,d,d2)
        return (R)
        
    def make_lam_E(self,d=d_SiO2+d_Al+d_MoS2/2):
       
        i_layer,d_extra=find_in_structure_with_inf(self.d_value_list, d)
        self.E_r_l_boundary_list=[sp.Matrix([self.t,0])]
        for i in range(len(self.M__list)):
            self.E_r_l_boundary_list.append(self.M__list[i]*self.E_r_l_boundary_list[0])
        
        if self.d_list[i_layer]==np.inf:
            d_prop=-d_extra
        else:
            d_prop=self.d_list[i_layer]-d_extra
        E_r_l_pos=P__(self.n_list[i_layer],d_prop,self.lamb)*self.E_r_l_boundary_list[-1-i_layer]  ## propagation from right side of i_layer_th material to position
        
        print(self.all_parameter_list)
        self.lam_E_r=lambdify(self.lambdify_args,E_r_l_pos[0])
        self.lam_E_l=lambdify(self.lambdify_args,E_r_l_pos[1])
        self.lam_E_ges=lambdify(self.lambdify_args,E_r_l_pos[0]+E_r_l_pos[1])
        print('object now has E_r,E_l,E_ges functions')
        return(self.lam_E_ges)
        
        
    def plot_spectrum_modification(self,d=d_SiO2+d_Al+d_MoS2/2,lamb_min=530e-9,lamb_max=870e-9,d_vac_min=0,d_vac_max=2300e-9):
            
        self.make_lam_E(d=d)
        plt.figure()
        lamb_list=np.linspace(lamb_min,lamb_max,200)
        d_list=np.linspace(d_vac_min,d_vac_max,200)
        for i in range(len(d_list)):
            I_laser=np.abs(self.lam_E_ges(532e-9,n_Al.get_n(532e-9),n_Au.get_n(532e-9),d_list[i]))**2*np.ones(len(lamb_list))
            I_emission=np.abs(self.lam_E_ges(lamb_list,n_Al.get_n(lamb_list),n_Au.get_n(lamb_list),d_list[i]))**2            
            plt.scatter(1e9*lamb_list,1e9*d_list[i]*np.ones(len(lamb_list)),
                        c=I_laser*I_emission,
                        vmin=-0.5,vmax=4,cmap=cm.nipy_spectral)
            
        plt.xlim(1e9*min(lamb_list),1e9*max(lamb_list))
        plt.ylim(1e9*min(d_list),1e9*max(d_list))
        plt.xlabel('wavelength in nm ')
        plt.ylabel('d in nm')
        plt.title('optical enhancement factor')
        plt.show()
        
    def plot_spectrum_modification_to_ax(self,ax,d=d_SiO2+d_Al+d_MoS2/2,lamb_min=530e-9,lamb_max=870e-9,d_vac_min=0,d_vac_max=2300e-9):
            
        self.make_lam_E(d=d)
        lamb_list=np.linspace(lamb_min,lamb_max,200)
        d_list=np.linspace(d_vac_min,d_vac_max,200)
        for i in range(len(d_list)):
            I_laser=np.abs(self.lam_E_ges(532e-9,n_Al.get_n(532e-9),n_Au.get_n(532e-9),d_list[i]))**2*np.ones(len(lamb_list))
            I_emission=np.abs(self.lam_E_ges(lamb_list,n_Al.get_n(lamb_list),n_Au.get_n(lamb_list),d_list[i]))**2            
            ax.scatter(1e9*lamb_list,1e9*d_list[i]*np.ones(len(lamb_list)),
                        c=I_laser*I_emission,
                        vmin=0,vmax=5,cmap=cm.nipy_spectral)
            
        ax.set_xlim(1e9*min(lamb_list),1e9*max(lamb_list))
        ax.set_ylim(1e9*min(d_list),1e9*max(d_list))
        ax.set_xlabel(r'$\lambda$ in nm')
        ax.set_ylabel(r'$d$ in nm')
        ax.set_title('simulated optical enhancement factor')

        
    def plot_E_of_lamb(self,d=d_SiO2+d_Al+d_MoS2/2,lamb_min=530e-9,lamb_max=870e-9,d_vac=d_vac):
            
        
        self.make_lam_E(d=d)
        print('d_value_list: ',self.d_value_list)
        lamb_list=np.linspace(lamb_min,lamb_max,200)
        I_emission=np.abs(self.lam_E_ges(lamb_list,n_Al.get_n(lamb_list),n_Au.get_n(lamb_list),d_vac))**2
        plt.plot(1e9*lamb_list,I_emission)
        plt.xlim(1e9*min(lamb_list),1e9*max(lamb_list))
        plt.show()
    
    def plot_d_dep_E(self,d_vac=d_vac,lamb=600e-9):
        E_list=[]
        d_list=np.linspace(-100e-9,100e-9+np.sum(self.d_value_list[1:-1]))
        for d in d_list:
            self.make_lam_E(d=d)
            E_list.append(np.abs(self.lam_E_ges(lamb,n_Al.get_n(lamb),n_Au.get_n(lamb),d_vac)))
        plt.plot(d_list,E_list)
        plt.show()
        
    def E_of_d_vac(self,d_vac,lamb=532e-9):
        #if 'lam_E_ges' not in self.__dict__:
        #    self.make_lam_E()
        return(self.lam_E_ges(lamb,n_Al.get_n(lamb),n_Au.get_n(lamb),d_vac))
        
        
    def plot_E_of_d_vac(self,lamb=532e-9):
        plt.figure()
        d_list=np.linspace(0,2300e-9,300)
        plt.plot(np.abs(self.E_of_d_vac(d_list))**2,d_list)
        plt.show()
if __name__=='__main__':
    example_tmm=sympy_tmm()
    example_tmm.plot_R_T(630e-9,970e-9,d_vac=10e-9)
    #example_tmm.plot_E_of_lamb(d_vac=800e-9)
    example_tmm.make_lam_E()
    example_tmm.plot_E_of_d_vac()
    example_tmm.plot_spectrum_modification()
    #example_tmm.plot_E(d=d_SiO2+d_Al)
    #example_tmm.plot_R_T(600e-9,900e-9)
    #example_tmm.plot_R_d(600e-9,900e-9,100e-9,2500e-9)