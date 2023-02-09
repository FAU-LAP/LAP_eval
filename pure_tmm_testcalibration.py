import lmfit
import tmm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lmfit import Parameters
import load


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

def plot_spec(wavelengths, d_mem, d_vac, amplitude,ax):
    for wavelength in wavelengths:
        result = tmm.coh_tmm('p', [1,2.6,1,2.6], [np.inf,d_mem,d_vac,np.inf],
                     th_0=0, lam_vac=wavelength)
        intensity = result['R']
        ax.scatter(wavelength, intensity*amplitude, s=1,c='red')

def tmm_fit(params, wavelengths, measured_intensity):
    def optimization_fun(pars):
        parvals = pars.valuesdict()
        d_mem = parvals['d_mem']
        d_vac = parvals['d_vac']
        amplitude = parvals['amplitude']
        reflectivity = []
        for wavelength in wavelengths:
            result = tmm.coh_tmm('s', [1, 2.58, 1, 2.58], [np.inf, d_mem, d_vac, np.inf],
                                 th_0=0, lam_vac=wavelength)
            reflectivity.append(result['R'])
        reflectivity = np.array(reflectivity)*amplitude
        return np.array((reflectivity-measured_intensity)**2, dtype=float)
    fit = lmfit.minimize(optimization_fun, params)
    return fit.params.valuesdict()


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
I_normed = I_normed / np.max(I_normed)
lamb_normed = lamb[i_min_lamb:i_max_lamb]


params = Parameters()
params.add('d_mem',565,min=450,max=650, vary=True)
params.add('d_vac',1440,min=1000,max=1500, vary=True)
params.add('amplitude',2,min=0.1,max=3, vary=True)

fit_result = tmm_fit(params, lamb_normed, I_normed)
print(fit_result)
fig, ax = plt.subplots(1)
ax.plot(lamb_normed,I_normed)
plot_spec(lamb_normed, fit_result['d_mem'], fit_result['d_vac'], fit_result['amplitude'],ax)
# plot_spec(lamb_normed, fit_result['d_mem'], fit_result['d_vac'], 1.4,ax)

print(fit_result['d_mem']+fit_result['d_vac'])




