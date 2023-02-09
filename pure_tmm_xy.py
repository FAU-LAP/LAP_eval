import lmfit
import tmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lmfit import Parameters
import os
import re


matplotlib.use('qt5agg')

# wavelengths = np.linspace(628,970,201)
# angles = np.linspace(0,17.7*np.pi/180,11)
# for wavelength in wavelengths:
#     intensity = 0
#     for angle in angles:
#         result = tmm.coh_tmm('p', [1,2.6,1,2.6], [np.inf,546,1240,np.inf],
#                      th_0=angle, lam_vac=wavelength)
#         intensity += result['R']
#
#     plt.scatter(wavelength, intensity/len(angles), c='blue')

def plot_spec(wavelengths, d_mem, d_vac, ax):
    for wavelength in wavelengths:
        result = tmm.coh_tmm('s', [1,2.6,1,2.6], [np.inf,d_mem,d_vac,np.inf],
                     th_0=0, lam_vac=wavelength)
        intensity = result['R']
        ax.scatter(wavelength, intensity, s=1,c='red')

def tmm_fit(params, wavelengths, measured_intensity):
    def optimization_fun(pars):
        parvals = pars.valuesdict()
        d_mem = parvals['d_mem']
        d_vac = parvals['d_vac']
        amplitude = parvals['amplitude']
        reflectivity = []
        for wavelength in wavelengths:
            result = tmm.coh_tmm('s', [1, 2.6, 1, 2.6], [np.inf, d_mem, d_vac, np.inf],
                                 th_0=0, lam_vac=wavelength)
            reflectivity.append(result['R'])
        reflectivity = np.array(reflectivity)*amplitude
        return np.array((reflectivity-measured_intensity)**2, dtype=float)
    print('running brute')
    brute_params = lmfit.minimize(fcn=optimization_fun, params=params, method='brute', Ns=20, keep=10)
    fit = lmfit.minimize(optimization_fun, brute_params.candidates[0].params, max_nfev=200)
    print(fit.nfev)
    return fit.params.valuesdict()


def get_fit_results(measurement_path, calibration_path, fit=False, ):
    lamb, intensity=np.loadtxt(measurement_path,skiprows=4,unpack=True)
    intensity=intensity-340
    lamb=np.array(lamb)
    calib_spec_path = calibration_path
    calib_data=np.loadtxt(calib_spec_path,skiprows=4,unpack=True)
    lamb_calib,intensity_calib= calib_data[0,:],calib_data[1,:]
    lamb_min = min(lamb[0], lamb_calib[0])
    lamb_max = max(lamb[-1], lamb_calib[-1])

    i_min_lamb = np.argmin(np.abs(lamb - lamb_min))
    i_min_lamb_calib = np.argmin(np.abs(lamb_calib - lamb_min))
    i_max_lamb = np.argmin(np.abs(lamb - lamb_max))
    print(len(np.abs(lamb_calib - lamb_max)))
    i_max_lamb_calib = np.argmin(np.abs(lamb_calib - lamb_max))

    print(i_min_lamb, i_max_lamb, i_min_lamb_calib, i_max_lamb_calib)

    I_normed = intensity[i_min_lamb:i_max_lamb] / intensity_calib[
                                                  i_min_lamb_calib:i_max_lamb_calib]
    # Remove this line to not norm to 0.7 any more!!!!#
    I_normed = I_normed / np.max(I_normed) * 0.8
    lamb_normed = lamb[i_min_lamb:i_max_lamb]


    params = Parameters()
    params.add('d_mem',565,min=550,max=580,vary=False,brute_step=10)
    params.add('d_vac',900,min=400,max=1400,vary=True,brute_step=50)
    params.add('amplitude',1,min=0.1,max=2,vary=False,brute_step=0.5)

    fit_result = tmm_fit(params, lamb_normed, I_normed)
    print(fit_result)
    if fit:
        fig, ax = plt.subplots(1)
        ax.plot(lamb_normed,I_normed)
        ax.set_title(f'{fit_result["d_mem"]=:4.0f} {fit_result["d_vac"]=:4.0f}')
        plot_spec(lamb_normed, fit_result['d_mem'], fit_result['d_vac'], ax)

    print(fit_result['d_mem']+fit_result['d_vac'])
    return fit_result['d_mem'], fit_result['d_vac']

folder_path = os.path.abspath(r'D:\Fuchs\Promotion\Measurements\Morris_Membrane_Distance\xy_map_61x61_time17-33-33')
search_pattern = re.compile(r'^spec_fabry_x(.*)_y(.*).txt$')
# search_pattern = re.compile(r'^spec_fabry.*.txt$')
i = 0
results_mem = []
results_vac = []
x_list = []
y_list = []

for file in os.listdir(folder_path):
    search_result = search_pattern.search(file)
    if search_result:
        i += 1
        if 0 <= i < 20000:
            if (525 < float(search_result[1]) < 555) and (-15 < float(search_result[2]) < 15):
                print(search_result[1], search_result[2])
                filepath = os.path.join(folder_path,search_result[0])
                calib_spec_path = os.path.abspath(r'\\confocal2\Measurement_Data\Morris\FP_morris\161\mirror_for_fabryperot_calibration\iris2.5mm_18-01-10\spec_fabry_x371.45_y-383.4.txt')
                result_mem, result_vac = get_fit_results(filepath, calib_spec_path, fit=False)
                results_mem.append(result_mem)
                results_vac.append(result_vac)
                x_list.append(search_result[1])
                y_list.append(search_result[2])

x_array = np.array(x_list, dtype=float)
y_array = np.array(y_list, dtype=float)
fig1, ax1 = plt.subplots(1)
fig2, ax2 = plt.subplots(1)
scatter1 = ax1.scatter(x_array, y_array, c=results_vac, marker='s')
scatter2 = ax2.scatter(x_array, y_array, c=results_mem, marker='s')
ax1.set_title('Vacume distance')
ax2.set_title('Membrane distance')
fig1.colorbar(scatter1, ax = ax1)
fig2.colorbar(scatter2, ax = ax2)








