import lmfit
import tmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lmfit import Parameters
import load
from joblib import Parallel, delayed
from datetime import datetime
startTime = datetime.now()

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
        result = tmm.coh_tmm('s', [1,1.5,1,2.6,1,2.6], [np.inf,0.5e-3,6e-3,d_mem,d_vac,np.inf],
                     th_0=0, lam_vac=wavelength)
        intensity = result['R']
        ax.scatter(wavelength, intensity, s=1,c='red')




def tmm_fit(params, wavelengths, measured_intensity, parallel):
    def optimization_fun(pars):
        parvals = pars.valuesdict()
        d_mem = parvals['d_mem']
        d_vac = parvals['d_vac']
        amplitude = parvals['amplitude']
        reflectivity = []
        parallel_result = parallel(
                delayed(get_tmm_coh)(wavelength, d_mem, d_vac) for wavelength in
                wavelengths)

        reflectivity = np.array(parallel_result, dtype=float)*amplitude
        return np.array((reflectivity-measured_intensity)**2, dtype=float)
    fit = lmfit.minimize(optimization_fun, params)
    return fit.params.valuesdict()

def get_tmm_coh(wavelength,d_mem, d_vac):
    return tmm.coh_tmm('s', [1, 2.6, 1, 2.6], [np.inf, d_mem, d_vac, np.inf], th_0=0, lam_vac=wavelength)['R']

lamb, intensity=np.loadtxt(r'\\confocal2\Measurement_Data\Morris\FP_morris\PN-27\cantilever_2023-02-08\Q4_16x6um_17-41-42\spec_fabry_x0.45_y0.15.txt',skiprows=4,unpack=True)
intensity=intensity-340
lamb=np.array(lamb)
calib_spec_path = r'\\confocal2\Measurement_Data\Morris\FP_morris\161\mirror_for_fabryperot_calibration\iris2.5mm_18-01-10\spec_fabry_x371.45_y-383.4.txt'
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
I_normed = I_normed / np.max(I_normed)
lamb_normed = lamb[i_min_lamb:i_max_lamb]


params = Parameters()
params.add('d_mem',565,min=450,max=650, vary=True)
params.add('d_vac',1440,min=1000,max=1500, vary=True)
params.add('amplitude',1,min=0.1,max=1.2, vary=True)

fit_result = tmm_fit(params, lamb_normed, I_normed, Parallel(n_jobs=3))
print(fit_result)
fig, ax = plt.subplots(1)
ax.plot(lamb_normed,I_normed)
plot_spec(lamb_normed, fit_result['d_mem'], fit_result['d_vac'], ax)

print(fit_result['d_mem']+fit_result['d_vac'])
print(f'Time for script: {datetime.now() - startTime}')



