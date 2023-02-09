import sys
sys.path.append(r'D:\Fuchs\Promotion\Python\TMM_Membrane\LAP_eval\refractive_index')
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
matplotlib.use('Qt5Agg')
sp.init_printing()
n_SiC = 2.6
d_membrane = 500e-9
d_vac = 1000e-9

tmm_model = sympy_tmm(n_value_list=[1,      n_SiC,       1,    n_SiC],
                      d_value_list=[np.inf, d_membrane, d_vac, np.inf],
                      d_parameter_list=[1,2],
                      n_parameter_list=[],
                      )
# search_pattern=re.compile(r'^spec_fabry_rough_.*.txt$')
# for file in os.listdir('D:\Fuchs\Promotion\Measurements\Morris_Membrane_Distance'):
#     search_result = search_pattern.search(file)
#     if search_result:
#         print(search_result[0])
#         matthias_plot.distance_from_spec(search_result[0], 'Auflicht_kein_Weitfeld_Intensitaet.txt',
#                                                 function=tmm_model.R_fit_func, plot=True,
#                                                 d_force=1400,
#                                                              )


x=[]
y=[]
distance=[]
i=0
folder_path = os.path.abspath(r'D:\Fuchs\Promotion\Measurements\Morris_Membrane_Distance\xy_map_61x61_time17-33-33')
search_pattern = re.compile(r'^spec_fabry_x(.*)_y(.*).txt$')
# search_pattern = re.compile(r'^spec_fabry.*.txt$')
for file in os.listdir(folder_path):
    search_result = search_pattern.search(file)
    if search_result:
        i += 1
        if 0 <= i < 10000:
            print(i)
            if (552 < float(search_result[1]) < 555) and (-80<float(search_result[2])<35):
                filepath = os.path.join(folder_path,search_result[0])
                calib_spec_path = 'Auflicht_kein_Weitfeld_Intensitaet.txt'
                distance.append(matthias_plot.distance_from_spec(filepath, calib_spec_path,
                                                    function=tmm_model.R_fit_func, plot=True,
                                                    d_force=1100,
                                                                 ))
                x.append(554)
                y.append(float(search_result[2]))
figfin, axfin = plt.subplots(1)
scatter = axfin.scatter(x=np.array(x), y=np.array(y), c=np.array(distance), cmap='viridis', marker = 's' )
figfin.colorbar(scatter)
