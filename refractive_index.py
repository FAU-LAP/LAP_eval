import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os 

package_directory = os.path.dirname(os.path.abspath(__file__))

class n_from_string():
    """
    load refractive index function from text file database. 
    database files retrieved from https://refractiveindex.info 
    """
    
    def __init__(self,material_string):
        self.material_string=material_string
        try:
            path=os.path.join(package_directory,'material_files/'+material_string+'_refractive_index_database.txt')
            lamb_n_k=np.loadtxt(path,skiprows=9).transpose()
            self.lamb_list=lamb_n_k[0]*1e-6
            self.n_func=interp1d(self.lamb_list,lamb_n_k[1],bounds_error=True)
            self.k_func=interp1d(self.lamb_list,lamb_n_k[2],bounds_error=True)
        except:
            try:
                path=os.path.join(package_directory,'material_files/'+material_string+'_refractive_index_tab_separated.txt')
                lamb_n=np.loadtxt(path,skiprows=1).transpose()
                self.lamb_list=lamb_n[0]*1e-6
                self.n_func=interp1d(self.lamb_list,lamb_n[1],bounds_error=True)
                self.k_func=interp1d(self.lamb_list,np.zeros(len(lamb_n[0])),bounds_error=True)
            except: raise ValueError("Could not load material")
        
        
        
    def get_n(self,lamb):
        return(self.n_func(lamb)+1j*self.k_func(lamb))
        
    def plot(self):
        print(self.lamb_list)
        print(self.get_n(1000e-9))
        plt.plot(self.lamb_list*1e9,np.real(self.get_n(self.lamb_list)),label='n',color='blue')
        plt.ylabel('n')
        plt.legend(loc=2)
        plt.xlabel(r'$\lambda$ in nm')
        ax2=plt.twinx()
        plt.plot(self.lamb_list*1e9,np.imag(self.get_n(self.lamb_list)),label='k',color='red')
        ax2.set_ylabel('k')
        plt.legend(loc=1)
        plt.title(self.material_string)
        plt.tight_layout()
        plt.show()
        
def get_materials():
    materials=[ f.split('_refractive_index')[0] 
            for dirpath, dirnames, files in os.walk(os.path.join(package_directory,'material_files'))
            for f in files if (f.endswith('_refractive_index_database.txt') or f.endswith('_refractive_index_tab_separated.txt')) ]
    
    return(materials)

if __name__=='__main__':
    
    print(get_materials())
    n_test=n_from_string('Al2O3')
    n_test.plot()

