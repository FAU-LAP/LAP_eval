# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 16:39:29 2021

@author: jo28dohe
"""
import numpy as np
import scipy.constants as const
from LAP_eval import refractive_index as ri



def ref_E(n_1,n_2):
    return (n_1-n_2)/(n_1+n_2)
    
def trans_E(n_1,n_2):
    if np.all(n_1<n_2):
        return 2*n_1/(n_1+n_2)
    else: 
        return 2*n_2/(n_1+n_2)

def Planck_lamb(lamb,A,T):
    return 2*const.pi*const.h*const.c**2*A/lamb**5/(np.exp(const.h*const.c/lamb/const.k/T)-1)




def coupling(lamb_list,d,n_Au_list,n_SiC_list,E_0=1,
             extra_imag_Au=0,extra_imag_SiC=-0.02,scatter_param=0.8):
    
    r_Au_list=ref_E(1,n_Au_list)
    r_Au_list = scatter_param*r_Au_list+extra_imag_Au*1j ## with correction for imperfect scattering and refelction phase
    n_SiC_list = ri.n_from_string('SiC').get_n(lamb_list)
    k=2*const.pi/lamb_list
    r_SiC_list = ref_E(1,n_SiC_list)+extra_imag_SiC*1j
    t_SiC_list = trans_E(n_SiC_list,1)+extra_imag_SiC*1j
    return n_SiC_list*np.abs(E_0*t_SiC_list*(1-1/r_SiC_list/(1-(1/((np.exp(2j*k*d)*r_Au_list*r_SiC_list))))))**2

def thermal_radiation_mirror(lamb,d,A,T,n_Au,n_SiC,extra_imag_Au=0,extra_imag_SiC=-0.02,scatter_param=0.8):
    return(Planck_lamb(lamb,A,T)*coupling(lamb,d,n_Au,n_SiC,E_0=1,
                                          extra_imag_Au=extra_imag_Au,extra_imag_SiC=extra_imag_SiC,scatter_param=scatter_param))