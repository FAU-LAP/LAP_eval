import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.constants as const
import matplotlib.ticker as ticker
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import axes3d,Axes3D
from scipy.interpolate import interp2d
import scipy.optimize as optimize
import matplotlib as mpl
import scipy.signal as sig
import lmfit
from lmfit import Model 
from lmfit import Parameters
import copy



import load



def distance_from_spec(spec_path,calib_spec_path,plot=False,input_error=False,d_force=0,d_fit_max='none',d_fit_min='none',function='none'):
    
    
    
    print('bin in distance_calculation')
    lamb, intensity=np.loadtxt(spec_path,skiprows=4,unpack=True)
    intensity=intensity-340
    lamb=np.array(lamb)
    try:
        calib_data=load.opendat(calib_spec_path)
    except: 
        print('Error: Could not load datafile')
    lamb_calib,intensity_calib= np.array(calib_data.data['wavelength in nm']),np.array(calib_data.data['Intensity in au'])-40
    if(plot):
        
        fig,((ax1,ax2),(ax3,ax4))= plt.subplots(nrows=2,ncols=2)
            
        
        ax1.plot(lamb,intensity)
        ax1.set_xlabel(r'$\lambda$ in nm')
        ax1.set_ylabel('intensity in a.u.')
        ax1.set_title('raw data')
       
        ax2.plot(lamb_calib,intensity_calib)
        ax2.set_xlabel(r'$\lambda$ in nm')
        ax2.set_ylabel('intensity in a.u.')
        ax2.set_title('calibration data')
    
    lamb_min=min(lamb[0],lamb_calib[0])
    lamb_max=max(lamb[-1],lamb_calib[-1])
    
    i_min_lamb=np.argmin(np.abs(lamb-lamb_min))
    i_min_lamb_calib=np.argmin(np.abs(lamb_calib-lamb_min))
    i_max_lamb=np.argmin(np.abs(lamb-lamb_max))
    print(len(np.abs(lamb_calib-lamb_max)))
    i_max_lamb_calib=np.argmin(np.abs(lamb_calib-lamb_max))
    
    print(i_min_lamb,i_max_lamb,i_min_lamb_calib,i_max_lamb_calib)
    
    I_normed=intensity[i_min_lamb:i_max_lamb]/intensity_calib[i_min_lamb_calib:i_max_lamb_calib]
    lamb_normed=lamb[i_min_lamb:i_max_lamb]
    if(plot):
        ax3.plot(lamb_normed,I_normed)
        ax3.set_xlabel(r'$\lambda$ in nm')
        ax3.set_ylabel('normed intensity in a.u.')
    
    
    
    ## find peaks in normed spectrum
    win=sig.hann(70)
    I_normed_filtered=sig.convolve(I_normed,win,mode='same')/np.sum(win)
    j_peak,dump = sig.find_peaks(I_normed_filtered)
    j_minima,dump=sig.find_peaks(-I_normed_filtered)
    ## correct peaks by fitting Parabola
    print('bin bei correct peaks')
    
    lamb_max=[]
    for j in j_peak:
        if j>30 and j< len(lamb_normed)-40 and I_normed[j]>np.percentile(I_normed,50):
            
            lamb_max.append(lamb_normed[j])
            
            if(plot):
                ax3.scatter(lamb_max[-1],I_normed_filtered[j],color='red')
    lamb_min=[]
    for j in j_minima:
        if j>30 and j< len(lamb_normed)-40 and I_normed[j]<np.percentile(I_normed,30):
            lamb_min.append(lamb_normed[j])
            
            if(plot):
                ax3.scatter(lamb_min[-1],I_normed_filtered[j],color='blue')
                
    lamb_extrema=np.concatenate((np.array(lamb_max),np.array(lamb_min)))
    ## calculate n and d from min,max, plot d 
    if len(lamb_extrema)>1:
        lamb_ex_max=max(lamb_extrema)
        lamb_ex_min=min(lamb_extrema)
        delta_n=1/2*(len(lamb_extrema)-1)   ## wenn \delta n =1 => len=2
        print('delta_n:',delta_n)
        d_approx=delta_n/2/(1/lamb_ex_min-1/lamb_ex_max)
        try:
            n=int(2*d_approx/lamb_min[-1]) ## benutze Position des langwelligsten Minimums für exakten Abstand
            d=n*lamb_min[-1]/2
        except:
            n=1
            d=0
    
        
    else:
        print('no d and n found...')
        d=1000
        n=0
    
    if d_force !=0:
        d=d_force
    ### do (kind of) Fabry Perot fit
            
    params=Parameters()
    if d_fit_min=='none':
        d_fit_min=d-500
    if d_fit_max=='none':
        d_fit_max=d+500
        
    params.add('d',d,min=d_fit_min,max=d_fit_max,brute_step=20)
    params.add('d2',500,min=480,max=520)
    print('d_start' , d)
    I_min=np.percentile(I_normed,5)
    I_max=np.percentile(I_normed,95)
    F=1
    
    def FabryPerot_min_max(d,I_min,I_max,F,lamb):
    
        return(I_min+(I_max-I_min)/(1-1/(1+F))*(1-1/(1+F*np.sin(2*np.pi*d/lamb)**2)))
           
    
    def cost_function_fit(lamb_list,I_list,I_min,I_max,F,params=params):
        def fun(pars):
            parvals = pars.valuesdict()
            d=parvals['d']
            d2=parvals['d2']
            
            if function=='none':
                ret=np.array((FabryPerot_min_max(d,I_min,I_max,F,lamb_list)-I_list)**2,dtype=float)
            else:
                ret=np.array((function(d*1e-9,d2*1e-9,lamb_list*1e-9,)-I_list)**2,dtype=float)
           
            return(ret)
        brute_result=lmfit.minimize(fun,params,method='brute')
        best_result=copy.deepcopy(brute_result)
        for candidate in brute_result.candidates[0:5]:
            trial = lmfit.minimize(fun, params=candidate.params,method='leastsq')
            if trial.chisqr < best_result.chisqr:
                best_result = trial
        return(best_result.params.valuesdict())
    
    fit_dict=cost_function_fit(lamb_normed,I_normed*np.linspace(1,1.0,len(I_normed)),I_min,I_max,F,params)
    print(fit_dict)
    d=fit_dict['d']
    d2=fit_dict['d2']
    print('return_1_start_'+str(d)+'_return_1_stop')
   
    
    if(plot):  
        if(input_error):
            ax3.set_title('Input error: Spectrum of test data: d= '+'{d:.0f} nm'.format(d=d),color='red')
        else:
            ax3.set_title('calibrated spectrum: d= '+'{d:.0f} nm'.format(d=d))
        
            
        
        ax4.plot(const.c/lamb_normed/1e12,I_normed)
        ax4.set_xlabel(r'$f$ in THz')
        ax4.set_ylabel('normed intensity in a.u.')
        ax1.figure.tight_layout()
        ax1.figure.canvas.draw()
        
        if function == 'none':
            ax3.plot(lamb_normed,FabryPerot_min_max(fit_dict['d'],I_min,I_max,F,lamb_normed))
        else:
            ax3.plot(lamb_normed,function(d*1e-9, d2*1e-9, lamb_normed*1e-9))
    fig.savefig(f'plots/{spec_path}_fits_plot.png')
    plt.close()
      
            
    if plot:
        plt.show()
    print('fitted distance: ',d)
    return(d)
        
def remove_cosmics(intensity,factor=1.2):
    intensity=np.array(intensity)
    fraction=intensity[1:len(intensity)]/intensity[0:len(intensity)-1]
    temp=np.insert(np.where(fraction<factor,intensity[1::],intensity[1::]*np.nan),0,intensity[0],axis=0)
    fraction2=intensity[0:len(intensity)-1]/intensity[1:len(intensity)]
    return(np.insert(np.where(fraction2<factor,temp[0:-1],temp[0:-1]*np.nan),-1,intensity[-1],axis=0))

    

def d(y,x,di=1):
    print('bin in d()')
    if type(di)==int:
        dy=[]
        ## anfang
        for i in np.arange(di):
            dy.append((y[i]-y[0])/(x[i]-x[0]))
        ## Hauptteil
        for i in np.arange(di,len(x)-di):
            dy.append(((y[i+di]-y[i])/(x[i+di]-x[i])+(y[i]-y[i-di])/(x[i]-x[i-di]))/2)
        ## Ende
        for i in np.arange(len(x)-di,len(x)):
            dy.append((y[len(x)-1]-y[i])/(x[len(x)-1]-x[i]))
        dy=np.array(dy)
        return(dy)
    ## wenn di vom typ float ist, soll delta x mindestens di sein.
    if type(di)==float:
        dy=[]
        ## suche jeweils nach unten und nach oben, ob es einen Punkt gibt, der nicht weiter als di entfernt ist.
        for i in np.arange(len(x)):
            dlow=0
            dhigh=0
            dxlow=0
            dylow=0
            dxhigh=0
            dyhigh=0
            for j in np.arange(i-1,-1,-1):
                dxlow=x[i]-x[j]
                if abs(dxlow)>di:
                    dylow=y[i]-y[j]
                    dlow=dylow/dxlow
                    break
            for j in np.arange(i+1,len(x),1):
                dxhigh=x[j]-x[i]
                if abs(dxhigh)>di:
                    dyhigh=y[j]-y[i]
                    dhigh=dyhigh/dxhigh
                    break
            if dlow is not 0 and dhigh is not 0:
                dy.append((dlow+dhigh)/2)
            elif dlow is 0:
                dy.append(dhigh)
            else:
                dy.append(dlow)
        return(np.array(dy))
        
def dfit(y,x,di=1):
    #print('bin in d()')
    x=np.array(x,dtype=float)
    y=np.array(y,dtype=float)
    
    if type(di)==int:
        dy=[]
        ## anfang
        for i in np.arange(di):
            dy.append(np.polyfit(x[0:i+di+1],y[0:i+di+1],1)[0])
            #print('polyfit: ',np.polyfit(x[0:i+di+1],y[0:i+di+1],1))
        ## Hauptteil
        for i in np.arange(di,len(x)-di):
            dy.append(np.polyfit(x[i-di:i+di+1],y[i-di:i+di+1],1)[0])
        ## Ende
        for i in np.arange(len(x)-di,len(x)):
            dy.append(np.polyfit(x[i-di:len(x)],y[i-di:len(x)],1)[0])
        dy=np.array(dy)
        return(dy)
    ## wenn di vom typ float ist, soll delta x mindestens di sein.
    if type(di)==float:
        dy=[]
        ## suche jeweils nach unten und nach oben, ob es einen Punkt gibt, der nicht weiter als di entfernt ist.
        for i in np.arange(len(x)):
            jlow=i
            jhigh=i
            dxlow=0
            dxhigh=0
            for j in np.arange(i,-1,-1):
                dxlow=x[i]-x[j]
                jlow=j
                if abs(dxlow)>di:
                    break
            for j in np.arange(i,len(x),1):
                dxhigh=x[j]-x[i]
                jhigh=j
                if abs(dxhigh)>di:
                    break
            #if i==0 or i == len(x):
                #print('jlow: ',jlow,'jhigh: ',jhigh)
            dy.append(np.polyfit(x[jlow:jhigh+1],y[jlow:jhigh+1],1)[0])
        return(np.array(dy))     
                
## Eine Funktion, die Mittelwert bestimmter Intervalle aus einer angelegten x-y-Spannung errechnet

def interval_mean(x,y,intervals=100):
    xmax=np.max(x)
    xmin=np.min(x)
    interval_length=(xmax-xmin)/intervals
    # build List with values in Intervals:
    x_list=np.linspace(xmin+interval_length/2,xmax-interval_length/2,intervals)
    y_list=[]
    for i in range(len(x_list)):
        y_sublist=np.array([y[j] for j in range(len(x))
                            if x[j]>x_list[i]-interval_length/2
                            and x[j]<x_list[i]+interval_length/2])
        y_list.append(np.mean(y_sublist))
    y_list=np.array(y_list)
    return((x_list,y_list))
        
## eine Funktionen die 0-max-min-0 Kurven über hin-und Rückweg mittelt   (gerade Anzahl benötigt!)        
def n_max_min_n_mean(x,y,istart=100):
    
    a=int((len(x)-istart)/4)

    x=np.array(x)
    y=np.array(y)
    x_list=np.concatenate(((x[3*a+istart:2*a+istart:-1]+x[3*a+istart:4*a+istart:1])/2,
                           (x[istart:a+istart:1]+x[2*a+istart:a+istart:-1])/2))

    y_list=np.concatenate(((y[3*a+istart:2*a+istart:-1]+y[3*a+istart:4*a+istart:1])/2,
                           (y[istart:a+istart:1]+y[2*a+istart:a+istart:-1])/2))
    
    return((x_list,y_list))        
## eine Funktion, die über hin-und rücklauf mittelt, damit ein Kennlinienfeld
##    flach werden kann


    
def mavg(x): ## mirroraverage
    x=np.array(x)
    ## a: erste Haelfte
    a=x[:len(x)/2]
    ## b: zweite Haelfte
    if len(x)%2==0:
        b=x[len(x)-len(x)/2:]
    else:
        b=x[len(x)+1-len(x)/2:]
    ## invertiere zweite Haelfte
    b=b[::-1]
    ## gebe Durchschnitt zufueck
    return((a+b)/2)

## eine Fuinktion, die über n Werte um einen jeweiligen x-Wert mittelt => filter

def avgfilter(x,n):
    print('bin in avgfilter')
    ret=[]
    ## Anfang
    for i in range(n):
        ret.append(sum(x[0:i+1])/(i+1))
    ## Hauptteil
    for i in range(n,len(x)-n):
        ret.append(sum(x[i-n:i+n+1])/(2*n+1))
    ## Ende
    for i in range(len(x)-n,len(x)):
        ret.append(sum(x[i:len(x)])/(len(x)-i))
    return np.array(ret)

    
## eine Funktion, die über n werte mittelt, damit im nächsten Schritt die
## Ableitung richtig ausgewertet werden kann

def pointavg(x,n):
    print(len(x),len(x)/n,-len(x)%n)
    
    x=np.array(x)   
    x=x[0:(len(x)-len(x)%n)]
    summe=x[::n]
    for i in range(1,n):
        summe+=x[i::n]
    return(summe/n)

## eine Funktion, die den richtigen Temperatursensor auswählt

def temp(a,b):
    Temp = []
    for i in range(len(a)):
        if (b[i]<1.6):
            Temp.append(a[i])
        else:
            Temp.append(b[i])
    return(np.array(Temp))

## eine Funktion, die außerhalb eines Bereichs np.nan zurückgibt

def cut(x,a,b):
    return(np.where(np.logical_and(x>=a,x<=b),x,np.nan))

## eine Funktion, die den quadrierten Abstand aller Punkte zu einer Gerade minimiert.

def geom_straight_fit(x,y,m_start=1,tol=1e-6):
    def fun(a):
        delta_y=a[0]*x+a[1]-y   ## Abstand zur Gerade in y-Richtung
        delta_x=(y-a[1])/a[0]-x   ## Abstand zur Gerade in x-Richtung
        abstand_2=delta_x**2*delta_y**2/(delta_x**2+delta_y**2) ## quadrierter Abstand in nähester Richtung
        ret=np.sum(abstand_2) ## Summe aller Quadrierten Abstände
        return(ret)
    return(optimize.minimize(fun,[m_start,0],method='Nelder-Mead',tol=tol))


## Funktion, die Plot-Stile setzt.

def set_style(style='paper_twocolumn'):
    if style=='paper_twocolumn':
        mpl.rcParams['figure.figsize']=3.375,3.375*3/4
        mpl.rcParams['figure.dpi']=600
        mpl.rcParams['font.size']=8
        mpl.rcParams['axes.grid']=True
        mpl.rcParams['axes.labelsize']='medium'
        mpl.rcParams['xtick.labelsize']='small'
        mpl.rcParams['ytick.labelsize']='small'
        mpl.rcParams['figure.subplot.left']=0.15
        mpl.rcParams['figure.subplot.right']=0.97
        mpl.rcParams['figure.subplot.bottom']=0.15
        mpl.rcParams['figure.subplot.top']=0.95
        mpl.rcParams['figure.subplot.hspace']=0.02
        plt.locator_params(nbins=6)

    if style=='paper_twocolumn_double_height':
        mpl.rcParams['figure.figsize']=3.375,3.375*6/4
        mpl.rcParams['figure.dpi']=600
        mpl.rcParams['font.size']=8
        mpl.rcParams['axes.grid']=True
        mpl.rcParams['axes.labelsize']='medium'
        mpl.rcParams['xtick.labelsize']='small'
        mpl.rcParams['ytick.labelsize']='small'
        mpl.rcParams['figure.subplot.left']=0.15
        mpl.rcParams['figure.subplot.right']=0.97
        mpl.rcParams['figure.subplot.bottom']=0.15
        mpl.rcParams['figure.subplot.top']=0.95
        mpl.rcParams['figure.subplot.hspace']=0.02
        plt.locator_params(nbins=6)
        
        
    if style=='paper_twocolumn_half':
        mpl.rcParams['figure.figsize']=3.375/2,3.375*3/8
        mpl.rcParams['figure.dpi']=600
        mpl.rcParams['font.size']=8
        mpl.rcParams['axes.grid']=True
        mpl.rcParams['axes.labelsize']='medium'
        mpl.rcParams['axes.labelpad']=0.5
        mpl.rcParams['xtick.labelsize']='small'
        mpl.rcParams['ytick.labelsize']='small'
        mpl.rcParams['xtick.major.pad']=1
        mpl.rcParams['ytick.major.pad']=0.5
        mpl.rcParams['figure.subplot.left']=0.28
        mpl.rcParams['figure.subplot.right']=0.92
        mpl.rcParams['figure.subplot.bottom']=0.24
        mpl.rcParams['figure.subplot.top']=0.93
        print('rcParams_gesetzt')
        plt.locator_params(nbins=4)


## Funktion, die die Ticks definiert:

def myticks(x,pos):

    if x == 0: return "$0$"
    exponent = int(np.log10(np.abs(x)))
    coeff = x/10**exponent

    return r"${:2.0f} \cdot 10^{{ {:2d} }}$".format(coeff,exponent)

class multiplot:
    def __init__(self,parent,save=False,savedir=''):
        
        ## Testweise alle Variablennamen ausgeben.
        for i in range(len(parent.varboxes)):
            for j in range(len(parent.varboxes[i].varboxes)):           
                print(parent.varboxes[i].varboxes[j].varnamelabel.text())
                
       # self.charfunc = parent.varboxes[0].charfunc
        
        
        
        ## übernehme parent variablen   
        self.parent=parent
        xname=parent.xcombo.currentText()
        yname=parent.ycombo.currentText()
        zname=parent.zcombo.currentText()
        uname=parent.ucombo.currentText()
        vname=parent.vcombo.currentText()
        wname=parent.wcombo.currentText()

        colormap=getattr(cm, parent.colorcombo.currentText())
        
        filenames=parent.files.filepaths
        self.xfuncstring=parent.xfuncstring.text()
        self.yfuncstring=parent.yfuncstring.text()
        self.zfuncstring=parent.zfuncstring.text()
        xlogscale=parent.xlogscalecheck.isChecked()
        
        ## übernehme kennlinien
        for varbox in self.parent.varboxes:
            try:
                setattr(self,varbox.namelabel.text(),varbox.charfunc2)
                print('Kennlinienfunktion '+varbox.namelabel.text()+' geladen')
            except:
                pass
        
        ## tex style benutzen

        #plt.rcParams['text.usetex']=True
        ## plotte x,y , falls keine Angabe in xfuncstring und yfuncstring

        if self.xfuncstring=='':
            self.xfuncstring = 'x'
        if self.yfuncstring=='':
            self.yfuncstring = 'y'
        if self.zfuncstring=='':
            self.zfuncstring = 'z'
       
        ## figure mit plot und colorbar
        fig=plt.figure(figsize=(float(parent.xfigsizeentry.text()),float(parent.yfigsizeentry.text())))
        
        if(parent.colorcheckbox.isChecked()):
            ax1=fig.add_axes([float(parent.leftspaceedit.text()),float(parent.lowerspaceedit.text()),1-float(parent.leftspaceedit.text())-float(parent.rightspaceedit.text()),1-float(parent.lowerspaceedit.text())-float(parent.upperspaceedit.text())])
            ax2=fig.add_axes([0.82,float(parent.lowerspaceedit.text()),0.025,0.75])
        elif(parent.legendcheckbox.isChecked()):
            ax1=fig.add_axes([float(parent.leftspaceedit.text()),float(parent.lowerspaceedit.text()),1-float(parent.leftspaceedit.text())-float(parent.rightspaceedit.text()),1-float(parent.lowerspaceedit.text())-float(parent.upperspaceedit.text())])
        else:
            ax1=fig.add_axes([float(parent.leftspaceedit.text()),float(parent.lowerspaceedit.text()),1-float(parent.leftspaceedit.text())-float(parent.rightspaceedit.text()),1-float(parent.lowerspaceedit.text())-float(parent.upperspaceedit.text())])


        ## berechne maximalen und minimalen Wert für Farbskala
        data=load.opendat(filenames[0])
        ## setze Daten

        x=np.array(data.data[xname])
        y=np.array(data.data[yname])
        z=np.array(data.data[zname])
        u=np.array(data.data[uname])
        v=np.array(data.data[vname])
        w=np.array(data.data[wname])
            
        ## berechne z-Wert
            
        zplot=self.h(x,y,z,u,v,w)
        print(zplot)
        zmean_min=np.mean(zplot)
        zmean_max=zmean_min
        for filename in filenames:
            
            ## lade Rohdaten aus Messung
            data=load.opendat(filename)
            
            ## setze Daten
            x=np.array(data.data[xname])
            y=np.array(data.data[yname])
            z=np.array(data.data[zname])
            u=np.array(data.data[uname])
            v=np.array(data.data[vname])
            w=np.array(data.data[wname])
            
            ## berechne z-Wert
            
            zplot=self.h(x,y,z,u,v,w)
            print(filename)
            zmean=np.mean(zplot)
            if(zmean<zmean_min):
                zmean_min=zmean
            if(zmean>zmean_max):
                zmean_max=zmean

        ## setze Plot Namen
        if self.parent.xlabelline.text():
            xplotname=self.parent.xlabelline.text()
        else:
            xplotname = self.xfuncstring.replace('x',xname).replace('y',yname).replace('z',zname).replace('u',uname).replace('v',vname).replace('w',wname)
        if self.parent.ylabelline.text():
            yplotname=self.parent.ylabelline.text()
        else:
            yplotname = self.yfuncstring.replace('x',xname).replace('y',yname).replace('z',zname).replace('u',uname).replace('v',vname).replace('w',wname)
        if self.parent.zlabelline.text():
            zplotname=self.parent.zlabelline.text()
        else:
            zplotname = self.zfuncstring.replace('x',xname).replace('y',yname).replace('z',zname).replace('u',uname).replace('v',vname).replace('w',wname)
        
        
        for filename in filenames:
            

            ## lade Rohdaten aus Messung
            data=load.opendat(filename)
            
            ## setze Daten
            x=np.array(data.data[xname])
            y=np.array(data.data[yname])
            z=np.array(data.data[zname])
            u=np.array(data.data[uname])
            v=np.array(data.data[vname])
            w=np.array(data.data[wname])
            
            
            ## berechne Funktionen            
            
            xplot=self.f(x,y,z,u,v,w)
            yplot=self.g(x,y,z,u,v,w)
            zplot=self.h(x,y,z,u,v,w)


            ## setze Farben
            zmean=np.mean(zplot)
            z_color_min=zmean_min
            z_color_max=zmean_max
            print(filename,"zmean: ",zmean,'len(v,w)',len(v),len(w),'mean(v,w)',np.mean(v),np.mean(w),'mean temp',np.mean(temp(v,w)))
            if (parent.colorminedit.text() and parent.colormaxedit.text()):
                z_color_min=float(parent.colorminedit.text())
                z_color_max=float(parent.colormaxedit.text())
            if(z_color_min==z_color_max):
                c=colormap(0)
            else:
                c=colormap(((zmean-z_color_min)/(z_color_max-z_color_min))**0.5)
            ## setze Label
            label = parent.legendnameentry.text().split('<')[0]+'%.1f'%zmean+parent.legendnameentry.text().split('>')[1]
            

            ## plotte, wenn zmean im z-Bereich
            plot = False
            if (parent.zminline.text() and parent.zmaxline.text()):
                if (zmean>float(parent.zminline.text()) and zmean<float(parent.zmaxline.text())):
                    plot = True
            else:
                plot=True
            ## entscheidung fertig, jetzt Plot
            if plot:
                try:
                    if parent.stylevar=='scatter':
                        print('scatter')
                        ax1.scatter(xplot,yplot,c=c,label=label)
                    else:
                        ax1.plot(xplot,yplot,c=c,label=label,linewidth=int(parent.linewidthentry.text()))

                    if savedir:
                        print('filename',filename)
                        a=filename.split('\\')
                        if len(a)>2:
                            b=a[-2]
                        else:
                            b=a[-2].split('/')[-1]
                
                        print('a: ',a)
                        load.savedat([xplotname,yplotname,zplotname],[xplot,yplot,zplot],savedir+'/evaluated_'+b+'_'+a[-1])
                except:
                    print('Das konnte nicht geplottet werden')
        ## logscale          
        if xlogscale:
            ax1.set_xscale('log')
        ax1.tick_params(labelsize=int(parent.tickfontsizeentry.text()))
        
        ## set  number of ticks
        if not (parent.xlogscalecheck.isChecked()):
            xloc=plt.MaxNLocator(int(parent.xticksedit.text()))
            ax1.xaxis.set_major_locator(xloc)
        ax1.grid()
        yloc=plt.MaxNLocator(int(parent.yticksedit.text()))
        ax1.yaxis.set_major_locator(yloc)
        #ax1.xaxis.set_major_formatter(ticker.FuncFormatter(myticks))
        #ax1.yaxis.set_major_formatter(ticker.FuncFormatter(myticks))
        ax1.set_xlabel(xplotname,fontsize=int(parent.labelfontsizeentry.text()))
        ax1.set_ylabel(yplotname,fontsize=int(parent.labelfontsizeentry.text()))
        if parent.xmaxline.text() and parent.xminline.text():
            ax1.set_xlim(float(parent.xminline.text()),float(parent.xmaxline.text()))
        if parent.ymaxline.text() and parent.yminline.text():
            ax1.set_ylim(float(parent.yminline.text()),float(parent.ymaxline.text()))
        ax1.ticklabel_format(style='sci',axis='y',scilimits=(-1e-3,1e3),useOffset=False)
        ax1.ticklabel_format(style='sci',axis='y',scilimits=(-1e-3,1e3),useOffset=False)
        #ax1.set_title(zplotname+' min = '+str(zmean_min)+' '+zplotname+' max = '+str(zmean_max))
        ## x-achse invertieren
        if parent.xinvertcheck.isChecked():
            print('invertiere x Achse')
            ax1.invert_xaxis()
        ## plot fertig, jetzt noch colorbar

        if(parent.colorcheckbox.isChecked()):
            norm=mpl.colors.Normalize(vmin=z_color_min,vmax=z_color_max)
            cb=mpl.colorbar.ColorbarBase(ax2,cmap=colormap,norm=norm,orientation='vertical')
            ax2.tick_params(labelsize=int(parent.tickfontsizeentry.text()))
            cb.set_label(zplotname,fontsize=int(parent.labelfontsizeentry.text()))
        ## legende hinzufügen
        if(parent.legendcheckbox.isChecked()):
            handles1,labels1=ax1.get_legend_handles_labels()
            if(parent.invertlegendcheck.isChecked()):
                handles1=handles1[::-1]
                labels1=labels1[::-1]
            leg=ax1.legend(handles1,labels1,bbox_to_anchor=(float(parent.legendxedit.text()),float(parent.legendyedit.text())),loc=2,borderaxespad=0.,fontsize=int(parent.legendfontsizeentry.text()))

            leg.draggable()
        ## plot speichern    
        if parent.saveplot:
            try:
                print(1)
                print(parent.savepath)
                print(parent.savenameedit.text())
                fig.savefig(parent.savepath+'/'+parent.savenameedit.text()+'.pdf',transparent=True)
                print(2)
                fig.savefig(parent.savepath+'/'+parent.savenameedit.text(),transparent=True)
            except:
                print('konnte nicht speichern, wahrscheinlich kein Zielordner angegeben')
        plt.show()
            
     ## Definiere Auswertungsfunktionen

    def f(self,x,y,z,u,v,w):
        try:
            return(eval(self.xfuncstring))
        except:
            print('Da hat etwas nicht geklappt')
    def g(self,x,y,z,u,v,w):
        try:
            return(eval(self.yfuncstring))
        except:
            print('Da hat etwas nicht geklappt')
    def h(self,x,y,z,u,v,w):
        try:
            return(eval(self.zfuncstring))
        except:
            print('Da hat etwas nicht geklappt')

class folderplot:
    def __init__(self,parent):
        
        ## übernehme parent variablen   
        self.parent=parent
        xname=parent.xcombo.currentText()
        yname=parent.ycombo.currentText()
        zname=parent.zcombo.currentText()
        uname=parent.ucombo.currentText()
        vname=parent.vcombo.currentText()
        wname=parent.wcombo.currentText()

        colormap=getattr(cm, parent.colorcombo.currentText())
        
        filenames=parent.files.filepaths
        self.xfuncstring=parent.xfuncstring.text()
        self.yfuncstring=parent.yfuncstring.text()
        self.zfuncstring=parent.zfuncstring.text()
        xlogscale=parent.xlogscalecheck.isChecked()

        ## übernehme Plotgrenzenliste
        try:
            self.rangelist=eval(parent.rangeedit.text())
            print(self.rangelist)
        except:
            print('konnte Plotgrenzen nicht übernehmen, beende Folderplot')
           
        ## übernehme kennlinien
        for varbox in self.parent.varboxes:
            try:
                setattr(self,varbox.namelabel.text(),varbox.charfunc2)
                print('Kennlinienfunktion '+varbox.namelabel.text()+' geladen')
            except:
                pass
        
        
        ## plotte x,y , falls keine Angabe in xfuncstring und yfuncstring

        if self.xfuncstring=='':
            self.xfuncstring = 'x'
        if self.yfuncstring=='':
            self.yfuncstring = 'y'
        if self.zfuncstring=='':
            self.zfuncstring = 'z'

        

        ## berechne maximalen und minimalen Wert für Farbskala
        data=load.opendat(filenames[0])
        ## setze Daten

        x=np.array(data.data[xname])
        y=np.array(data.data[yname])
        z=np.array(data.data[zname])
        u=np.array(data.data[uname])
        v=np.array(data.data[vname])
        w=np.array(data.data[wname])
            
        ## berechne z-Werte 
            
        zplot=self.h(x,y,z,u,v,w)
        zmean_min=np.mean(zplot)
        zmean_max=zmean_min
        self.zdict={}
        for filename in filenames:
            
            ## lade Rohdaten aus Messung
            data=load.opendat(filename)
            
            ## setze Daten
            x=np.array(data.data[xname])
            y=np.array(data.data[yname])
            z=np.array(data.data[zname])
            u=np.array(data.data[uname])
            v=np.array(data.data[vname])
            w=np.array(data.data[wname])

            
            ## berechne z-Wert
            
            zplot=self.h(x,y,z,u,v,w)
            print('Berechne z-Wert von: ',filename)
            zmean=np.mean(zplot)
            if(zmean<zmean_min):
                zmean_min=zmean
            if(zmean>zmean_max):
                zmean_max=zmean
            self.zdict[filename]=zmean
        print('z-Werte fertig berechnet')
        ## setze Plot Namen
        if self.parent.xlabelline.text():
            xplotname=self.parent.xlabelline.text()
        else:
            xplotname = self.xfuncstring.replace('x',xname).replace('y',yname).replace('z',zname).replace('u',uname).replace('v',vname).replace('w',wname)
        if self.parent.ylabelline.text():
            yplotname=self.parent.ylabelline.text()
        else:
            yplotname = self.yfuncstring.replace('x',xname).replace('y',yname).replace('z',zname).replace('u',uname).replace('v',vname).replace('w',wname)
        if self.parent.zlabelline.text():
            zplotname=self.parent.zlabelline.text()
        else:
            zplotname = self.zfuncstring.replace('x',xname).replace('y',yname).replace('z',zname).replace('u',uname).replace('v',vname).replace('w',wname)
        
        

        print('Beginne mit Erstellen der Listen')
        ## erstelle Listen mit filenames für die jeweiligen Intervalle
        self.files_in_range=[]
        for i in range(len(self.rangelist)-1):
            self.files_in_range.append([])
            for filename in self.zdict:
                if self.zdict[filename]>=self.rangelist[i] and self.zdict[filename]<self.rangelist[i+1]:
                    self.files_in_range[-1].append(filename)
        print('Listen fertig erstellt')
        ## erzeuge Plotordner,falls noch nicht vorhanden
        self.folderpath=parent.plotfolderpath
        ## erzeuge Plots
        for i in range(len(self.rangelist)-1):
            fig=plt.figure()
            ax1=fig.add_axes([0.1,0.1,0.75,0.8])
            ax2=fig.add_axes([0.87,0.1,0.025,0.8])
            for filename in self.files_in_range[i]:
                print('filename')
            

                ## lade Rohdaten aus Messung
                data=load.opendat(filename)
            
                ## setze Daten
                x=np.array(data.data[xname])
                y=np.array(data.data[yname])
                z=np.array(data.data[zname])
                u=np.array(data.data[uname])
                v=np.array(data.data[vname])
                w=np.array(data.data[wname])
            
            
                ## berechne Funktionen            
            
                xplot=self.f(x,y,z,u,v,w)
                yplot=self.g(x,y,z,u,v,w)
                zplot=self.h(x,y,z,u,v,w)


               ## setze Farben
                zmean=np.mean(zplot)
                z_color_min=zmean_min
                z_color_max=zmean_max
                print(filename,"zmean: ",zmean,'len(v,w)',len(v),len(w),'mean(v,w)',np.mean(v),np.mean(w),'mean temp',np.mean(temp(v,w)))
                if (parent.colorminedit.text() and parent.colormaxedit.text()):
                    z_color_min=float(parent.colorminedit.text())
                    z_color_max=float(parent.colormaxedit.text())
                if(z_color_min==z_color_max):
                    c=colormap(0)
                else:
                    c=colormap((zmean-z_color_min)/(z_color_max-z_color_min))

                try:
                    ax1.plot(xplot,yplot,c=c)
                except:
                    print('Das konnte nicht geplottet werden')
            if xlogscale:
                ax1.set_xscale('log')
            ax1.grid()
            ax1.set_xlabel(xplotname)
            ax1.set_ylabel(yplotname)
            if parent.xmaxline.text() and parent.xminline.text():
                ax1.set_xlim(float(parent.xminline.text()),float(parent.xmaxline.text()))
            if parent.ymaxline.text() and parent.yminline.text():
                ax1.set_ylim(float(parent.yminline.text()),float(parent.ymaxline.text()))
            ax1.ticklabel_format(style='sci',axis='y',scilimits=(-1e-3,1e3),useOffset=False)
            ax1.ticklabel_format(style='sci',axis='y',scilimits=(-1e-3,1e3),useOffset=False)
            #ax1.set_title(zplotname+' min = '+str(self.rangelist[i])+' '+zplotname+' max = '+str(self.rangelist[i+1]))

            ## zeichne Colorbar

            ## plot fertig, jetzt noch colorbar
            norm=mpl.colors.Normalize(vmin=z_color_min,vmax=z_color_max)
            cb=mpl.colorbar.ColorbarBase(ax2,cmap=colormap,norm=norm,orientation='vertical')
            cb.set_label(zplotname)


            
            fig.savefig(self.folderpath+'/Plot_'+str(i)+'_z_min='+str(self.rangelist[i])+'.png')
            plt.close(fig)

        
     ## Definiere Auswertungsfunktionen

    def f(self,x,y,z,u,v,w):
        try:
            return(eval(self.xfuncstring))
        except:
            print('Da hat etwas nicht geklappt')
    def g(self,x,y,z,u,v,w):
        try:
            return(eval(self.yfuncstring))
        except:
            print('Da hat etwas nicht geklappt')
    def h(self,x,y,z,u,v,w):
        try:
            return(eval(self.zfuncstring))
        except:
            print('Da hat etwas nicht geklappt')
        
class contourplot:
    def __init__(self,parent):

        
        ## übernehme parent variablen   
        self.parent=parent
        xname=parent.xcombo.currentText()
        yname=parent.ycombo.currentText()
        zname=parent.zcombo.currentText()
        filenames=parent.files.filepaths
        self.xfuncstring=parent.xfuncstring.text()
        self.yfuncstring=parent.yfuncstring.text()
        self.zfuncstring=parent.zfuncstring.text()
        
        
        
        ## plotte x,y , falls keine Angabe in xfuncstring und yfuncstring
        print('super: ',parent.xcombo.currentText())
        if self.xfuncstring=='':
            self.xfuncstring = 'x'
        if self.yfuncstring=='':
            self.yfuncstring = 'y'
        if self.zfuncstring=='':
            self.zfuncstring = 'z'
       
    
        fig=plt.figure()
        plt.subplot(111)
        x=[]
        y=[]
        z=[]
        for filename in filenames:
            data=load.opendat(filename)
            ## setze Daten
            for xi in data.data[xname]:
                x.append(xi)
            for yi in data.data[yname]:
                y.append(yi)
            for zi in data.data[zname]:
                z.append(zi)
            ## berechne Funktionen            

        print(len(x),len(y),len(z))
        ax=fig.add_subplot(111,projection='3D')
        ax.scatter(x,y,z)
        

    
