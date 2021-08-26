import numpy as np
import math
import scipy.constants as const
from scipy.interpolate import interp1d
import lmfit
from lmfit import Parameters
import matplotlib.pyplot as plt
import pandas as pd
import copy



    
def remove_cosmics(intensity,factor=1.2):
    intensity=np.array(intensity)
    intensityx = np.append(intensity,[intensity[-1],intensity[-1]])
    intensity1 = np.append(intensity[0],intensityx)
    fraction=intensity[1::]/intensity[0:-1]
    test = np.where(fraction<factor,intensity[1::],(intensity1[0:-4]+intensity1[4::])/2)
    temp=np.insert(test,0,intensity[0],axis=0)
    intensityx = np.append(intensity[0],intensity)
    intensityy = np.append(intensityx,intensity[-1])
    intensity1 = np.append(intensity[0],intensityy)
    fraction2=intensity[0:-1]/intensity[1::]
    return np.insert(np.where(fraction2<factor,temp[0:-1],(intensity1[0:-4]+intensity1[4::])/2),-1,intensity[-1],axis=0)



def plot_brute_leastsquares_results(result, best_vals=True, varlabels=None,
                       output=None, leastsq_fit_result=None):
    """Visualize the result of the brute force grid search.

    The output file will display the chi-square value per parameter and contour
    plots for all combination of two parameters.

    Inspired by the `corner` package (https://github.com/dfm/corner.py).

    Parameters
    ----------
    result : :class:`~lmfit.minimizer.MinimizerResult`
        Contains the results from the :meth:`brute` method.

    best_vals : bool, optional
        Whether to show the best values from the grid search (default is True).

    varlabels : list, optional
        If None (default), use `result.var_names` as axis labels, otherwise
        use the names specified in `varlabels`.

    output : str, optional
        Name of the output PDF file (default is 'None')
    """
    from matplotlib.colors import LogNorm
    import matplotlib.cm as cm
    
    npars = len(result.var_names)
    _fig, axes = plt.subplots(npars, npars)

    if not varlabels:
        varlabels = result.var_names
    if best_vals and isinstance(best_vals, bool):
        best_vals = result.params
    if leastsq_fit_result is not None:
        best_vals_leastsq = leastsq_fit_result.params

    for i, par1 in enumerate(result.var_names):
        for j, par2 in enumerate(result.var_names):

            # parameter vs chi2 in case of only one parameter
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=3)
                axes.set_ylabel(r'$\chi^{2}$')
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls='dashed', color='b')
                if best_vals_leastsq:
                    axes.axvline(best_vals_leastsq[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on top
            elif i == j and j < npars-1:
                if i == 0:
                    axes[0, 0].axis('off')
                ax = axes[i, j+1]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.unique(result.brute_grid[i]),
                        np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        'o', ms=3)
                ax.set_ylabel(r'$\chi^{2}$')
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('right')
                ax.set_xticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='b')
                if best_vals_leastsq:
                    ax.axvline(best_vals_leastsq[par1].value, ls='dashed', color='r')
            # parameter vs chi2 profile on the left
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        np.unique(result.brute_grid[i]), 'o', ms=3)
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i])
                if i != npars-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(r'$\chi^{2}$')
                if best_vals:
                    ax.axhline(best_vals[par1].value, ls='dashed', color='b')
                if best_vals_leastsq:
                    ax.axhline(best_vals_leastsq[par1].value, ls='dashed', color='r')

            # contour plots for all combinations of two parameters
            elif j > i:
                ax = axes[j, i+1]
                red_axis = tuple([a for a in range(npars) if a not in (i, j)])
                X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                   np.unique(result.brute_grid[j]))
                lvls1 = np.linspace(result.brute_Jout.min(),
                                    np.median(result.brute_Jout)/2.0, 20, dtype='int')
                lvls2 = np.linspace(np.median(result.brute_Jout)/2.0,
                                    np.median(result.brute_Jout), 20, dtype='int')
                
                lvls = np.unique(np.concatenate((lvls1, lvls2)))
                
                ax.contourf(X.T, Y.T, np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            lvls, norm=LogNorm(),
                            cmap=cm.nipy_spectral)
                ax.set_yticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='b')
                    ax.axhline(best_vals[par2].value, ls='dashed', color='b')
                    ax.plot(best_vals[par1].value, best_vals[par2].value, 'bs', ms=3)
                if best_vals_leastsq:
                    ax.axvline(best_vals_leastsq[par1].value, ls='dashed', color='r')
                    ax.axhline(best_vals_leastsq[par2].value, ls='dashed', color='r')
                    ax.plot(best_vals_leastsq[par1].value, best_vals_leastsq[par2].value, 'rs', ms=3)
                if j != npars-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel(varlabels[i])
                if j - i >= 2:
                    axes[i, j].axis('off')

    if output is not None:
        plt.savefig(output)
    else: 
        plt.tight_layout()
        plt.show()
   


def brute_leastsquare_fit(fun, x_data, y_data,weight_data=None,p_names=None,p_min_max_steps_dict=None,
                          const_params=[], visualize=False):
    
    """A very robust fit routine inspired from 
    https://lmfit.github.io/lmfit-py/examples/example_brute.html
    that first performs a brute force fit with subsequent least squares fit of 
    best results"""
    
    if p_names == None or p_min_max_steps_dict==None:
        raise Exception ('p_names and p_min_max_steps must be given!'+ 
                         'structure of p_min_max_steps_dict: {"pname0":[min0,max0,brute_steps0]}')
   
    params = Parameters() ### initialize LMfit parameters
    for p_name in p_names:
        min_val=p_min_max_steps_dict[p_name][0]
        max_val=p_min_max_steps_dict[p_name][1]
        steps=p_min_max_steps_dict[p_name][2]
        params.add(p_name,value=min_val,
                   min=min_val,
                   max=max_val,
                   brute_step=(max_val-min_val)/(steps-1))
        
    ### define function to be minimized for fit 
    
    def cost_function_fit(p=params):
            def minimize_fun(pars):
                
                v=pars.valuesdict()
                arglist=[]
                for p_name in p_names:
                    arglist.append(v[p_name])
                
                for const_param in const_params:
                    arglist.append(const_param)
                
                ret=np.array((fun(x_data,*arglist)-y_data),dtype=float)
                if weight_data is not None:
                    ret=ret*np.sqrt(weight_data)
                return(ret)
            brute_result=lmfit.minimize(minimize_fun,params,method='brute',nan_policy='omit')
            best_result=copy.deepcopy(brute_result)
            for candidate in brute_result.candidates[0:5]:
                trial = lmfit.minimize(minimize_fun, params=candidate.params,method='leastsq',nan_policy='omit')
                if trial.chisqr < best_result.chisqr:
                    best_result = trial
            
            return((best_result,brute_result))
            
    best_result,brute_result = cost_function_fit()
    arg_list=[]
    for p_name in p_names:
        arg_list.append(best_result.params.valuesdict()[p_name])
    for const_param in const_params:
        arg_list.append(const_param)
    
        
    if visualize == True:
        plot_brute_leastsquares_results(brute_result,leastsq_fit_result=best_result)
        plt.figure()
        plt.plot(x_data,y_data,label='data',color='blue')
        plt.plot(x_data,fun(x_data,*arg_list),label='Fit',color='red')
        plt.title(best_result.params.valuesdict())
        plt.show()
    return (arg_list[0:len(p_names)])
     
def stitchSpectra(lamb_list,count_list, method="scale", edgeremove=(0, 0), shiftToPositive=False, dlambda=None):
    """
    Stitches the raw spectra together. For this purpose, the following spectra are shifted
    to match the previous spectra in the overlapping region. Afterwards the whole data
    is interpolated on a fixed grid.

    :param str method: stitching method (possible values: scale, shift)
    :param edgeremove: ratio of omitted data at the edges
        (e.g. (0.05, 0.05) and edgetype="symmetric": first 5% and last 5% of data is omitted)
    :type edgeremove: tuple(float, float)
    :param bool shiftToPositive: if True the spectrum is shifted such that min(spectrum) >= 0
    :param float dlambda: custom wavelength steps for interpolation, None for default
    """
    rawData=np.array([np.array(lamb_list),np.array(count_list)])
    rawData=rawData.swapaxes(0,1)
    coefficients = []
    print("Removing edges for stitching:", *edgeremove)
    omitRight = rawData[0].shape[1] - math.floor(rawData[0].shape[1] * edgeremove[1])
    print("Stitching index range is ", 0, omitRight)
    processed = np.array(rawData[0][:, 0:omitRight])  
    if dlambda is None:
        dlambda = math.fabs(processed[0, 1] - processed[0, 0])  ## lambda steps of first spectrum are kept
    for i, spec in enumerate(rawData[1:]):
        omitLeft = math.floor(spec.shape[1] * edgeremove[0])
        omitRight = spec.shape[1] - math.floor(spec.shape[1] * edgeremove[1])
        print("Stitching index range is ", omitLeft, omitRight)
        if i == len(rawData)-2:
            spec = np.array(spec[:, omitLeft:])  ## do not shorten last array at end
        else:
            spec = np.array(spec[:, omitLeft:omitRight]) # shorten middle arrays at both sides
        print("Stitching spectrum in range", np.min(spec[0,]), np.max(spec[0,]))
        # calculate overlap
        overlap = (np.min(spec[0,]), np.max(processed[0,])) 
        #lambdas = np.arange(*overlap, dlambda)
        #leftfun = interp1d(processed[0,], processed[1,])
        #rightfun = interp1d(spec[0,], spec[1,])
        left = np.mean(processed[1, processed[0,] > overlap[0]]) ##mean of counts of overlap
        right = np.mean(spec[1, spec[0,] < overlap[1]])
        if method == "shift":
            # calculate offset in overlap region
            offset = left - right
            print("Stitching offset %s in overlap", offset, *overlap)
            # add shifted spectrum
            spec[1,] = spec[1,] + offset
            coefficients.append(offset)
        elif method == "scale":
            # calculate factor in overlap region
            factor = left/right
            print("Stitching factor"+str(factor)+"  in overlap ", *overlap)
            spec[1,] = spec[1,] * factor
            coefficients.append(factor)
        processed = np.concatenate([processed, spec], axis=1)
    # interpolate data on grid
    interpolated = interp1d(processed[0,], processed[1,])
    lambdas = np.arange(processed[0, 0], processed[0, -1], dlambda)
    specdata = interpolated(lambdas)
    # shift above 0
    if shiftToPositive:
        minimum = np.min(specdata)
        if minimum < 0:
            specdata += math.fabs(minimum)
    
    return (lambdas,specdata,coefficients)




if __name__=='__main__':
    ## generate some example data for fitting and testing routines 
    test_brute_fit=True
    if test_brute_fit:
        def fun(x,a,b,c):
            ret=a*np.sin(b*x)*np.exp(c*x)
            return(ret)
            
        df=pd.DataFrame.from_dict({'x_data':np.linspace(0,10,200)})
        df['y_data']=fun(df['x_data'],1,5,-0.2)
        df['noise']=np.random.normal(scale=0.2,size=200)
        df['y_sim']=df['y_data']+df['noise']
        
        arg_list = brute_leastsquare_fit(fun,df['x_data'],df['y_sim'],p_names=['a','b','c'],
                                         p_min_max_steps_dict={'a':[0,2,40],'b':[0,10,40],'c':[-1,1,40]},
                                         visualize=True)
        df['fitted_function']=fun(df['x_data'],*arg_list)
        df.plot('x_data')
    
    test_stitchSpectra=False
    plt.figure()
    if test_stitchSpectra:
        lamb_list=[]
        counts_list=[]    
        for i in range(4):
            lamb_list.append(np.linspace(500+i*100,700+i*100,256))
            counts_list.append(100*np.sin(lamb_list[-1]/100)**2+30*np.random.rand(256))
            plt.plot(lamb_list[-1],counts_list[-1])
            
        
        lamb,counts,coefficients=stitchSpectra(lamb_list, counts_list)
        plt.plot(lamb,counts,color='black')
        plt.xlabel('stitched lamb')
        plt.ylabel('stitched spectra')
        plt.show()
        
        
   
    
    