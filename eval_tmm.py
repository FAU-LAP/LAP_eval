import sys
import os
import matthias_plot
from sympy_transfer_matrix import sympy_tmm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import sympy as sp
from sympy.utilities.lambdify import lambdify
import re
matplotlib.use('Agg')
sp.init_printing()
n_SiC = 2.6
d_membrane = 500e-9
d_vac = 1000e-9

tmm_model = sympy_tmm(n_value_list=[1,      n_SiC,       1,    n_SiC],
                      d_value_list=[np.inf, d_membrane, d_vac, np.inf],
                      d_parameter_list=[1,2],
                      n_parameter_list=[],
                      )
search_pattern=re.compile(r'^spec_fabry_rough_.*x(.*)_y(.*).txt$')


x=[]
y=[]
distance=[]
i=0
for file in os.listdir():
    search_result = search_pattern.search(file)
    if search_result:
        if i < 1000:
            i+=1
            filepath = search_result[0]
            calib_spec_path = 'Auflicht_kein_Weitfeld_Intensitaet.txt'
            distance.append(matthias_plot.distance_from_spec(filepath, calib_spec_path,
                                                function=tmm_model.R_fit_func, plot=True,
                                                #d_force=1300,
                                                             ))
            x.append(float(search_result[1]))
            y.append(float(search_result[2]))
figfin, axfin = plt.subplots(1)
axfin.scatter(x=np.array(x), y=np.array(y), c=np.array(distance), cmap='viridis', )
#figfin.colorbar()
figfin.savefig('testsavefig.png')
# fig.show()
