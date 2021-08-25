# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:09:10 2021

@author: jo28dohe
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

### TODO: typical figure sizes 
##- twocolumn paper: 3.375,3.375*3/4  (inches) or 8.57 cm , 8.57*3/4 cm

class paperfigure:
    '''
    returns an object containing figures with typical paper plots.
    width_in_cols=1 cooresponds to 8.6*cm, the column width of PRL
    '''
    
    def __init__(self,width_in_cols=1,aspect_ratio=4/3,
                 override_figsize=None):
        cm_to_in=0.3937
        self.width_in_cols=width_in_cols
        self.width=width_in_cols*8.6*cm_to_in   ## has to be in inch for matplotlib
        self.aspect_ratio=aspect_ratio
        mpl.rcParams['figure.dpi']=600
        mpl.rcParams['axes.grid']=True
        mpl.rcParams['axes.labelsize']='medium'
        mpl.rcParams['xtick.labelsize']='small'
        mpl.rcParams['ytick.labelsize']='small'
        
        if width_in_cols<0.6:
            
            mpl.rcParams['font.size']=8
            plt.locator_params(nbins=4)
            mpl.rcParams['figure.subplot.left']=0.28
            mpl.rcParams['figure.subplot.right']=0.92
            mpl.rcParams['figure.subplot.bottom']=0.24
            mpl.rcParams['figure.subplot.top']=0.93
        else:
            plt.locator_params(nbins=6)
            
            mpl.rcParams['font.size']=9
            mpl.rcParams['figure.subplot.left']=0.15
            mpl.rcParams['figure.subplot.right']=0.97
            mpl.rcParams['figure.subplot.bottom']=0.15
            mpl.rcParams['figure.subplot.top']=0.95
            mpl.rcParams['figure.subplot.hspace']=0.02
        
        self.fig, self.ax =plt.subplots(figsize=(self.width,self.width/self.aspect_ratio))
        
class colorplot(paperfigure):
    '''
    creates a colorplot using plt.scatter()
    x_data shall be a list or pandas DataFrame of lists/arrays/dataframes
    '''
    
    def __init__(self,x_data,y_data,c_data,
                 xlabel=None,ylabel=None,clabel=None,cmap=cm.nipy_spectral,
                 vmin=None,vmax=None,make_cbar=True,cbar_pos='top',
                 **kwargs):
        super().__init__(**kwargs)
        self.cmap=cmap
        ### re-create fig and ax for inclusion of colorbar:
        if make_cbar:
            heights=[0.05,1]
            self.fig, (self.ax_cbar,self.ax) =plt.subplots(nrows=2,ncols=1,
                                                       figsize=(self.width,self.width/self.aspect_ratio),
                                                       gridspec_kw={'height_ratios':heights})
        
        ### make list if only one line of x_data is given:
        if np.shape(x_data)==():
            x_data=[x_data]
            y_data=[y_data]
            c_data=[c_data]
        ### set maximal and minimal color values
        if vmin==None:
            min_list=[np.nanmin(x) for x in c_data]
            self.vmin=np.nanmin(min_list)
        else:
            self.vmin=vmin
        if vmax==None:
            max_list=[np.nanmax(x) for x in c_data]
            self.vmax=np.nanmax(max_list)
        else:
            self.vmax=vmax
            
        
        ### make the colorplot using scatter 
        for x,y,c in zip(x_data,y_data,c_data):
        
            sc=self.ax.scatter(x,y,c=c,cmap=self.cmap,
                            vmin=self.vmin,vmax=self.vmax)
        
        ###  label axes. If no name specified try to use name of xdata
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        else:
            try:
                self.ax.set_xlabel(x_data.name) 
            except:
                pass
                
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        else:
            try:
                self.ax.set_ylabel(y_data.name)
            except:
                pass
        ### switch off grid for colorplots
        self.ax.grid(False)
        
        ## create colorbar
        if make_cbar:
            cbar=plt.colorbar(sc,cax=self.ax_cbar,orientation='horizontal')
            
            if self.width_in_cols < 0.6:
                n_ticks=3
            else:
                n_ticks=5
            cbar.set_ticks(list(np.linspace(self.vmin,self.vmax,n_ticks)))
            
            self.ax_cbar.xaxis.tick_top()
            
            
            if clabel is not None:
                self.ax_cbar.set_title(clabel)
            else:
                try:
                    self.ax_cbar.set_title(c_data.name) 
                except:
                    pass
        self.fig.tight_layout(pad=0.5)
        
class multiline_plot(paperfigure):
    '''
    creates a multi-line plot using plt.plot()
    x_data and y_data shall be a list or pandas DataFrame of lists/arrays/dataframes
    c_data shall be a list of values
    
    rel_vmin,rel_vmax change vmin and vmax relative to min/max if vmin,vmax are None
    '''
    
    def __init__(self,x_data,y_data,c_data,
                 xlabel=None,ylabel=None,cmap=cm.plasma,
                 vmin=None,vmax=None,rel_vmin=1,rel_vmax=1,
                 **kwargs):
        super().__init__(**kwargs) 
        self.cmap=cmap
        ### make list if only one line of x_data is given:
        if np.shape(x_data)==():
            x_data=[x_data]
            y_data=[y_data]
        if np.shape(c_data)==():
            c_data=[c_data]
        ### set maximal and minimal color values
        if vmin==None:
            self.vmin=np.nanmin(c_data)*rel_vmin
        else:
            self.vmin=vmin
        if vmax==None:
            self.vmax=np.nanmax(c_data)*rel_vmax
        else:
            self.vmax=vmax
            
        
        ### make the multiline plot using plt.plot 
        for x,y,c in zip(x_data,y_data,c_data):
        
            color=self.cmap((c-self.vmin)/(self.vmax-self.vmin))
            self.ax.plot(x,y,c=color,label=c)
        
        ###  label axes. If no name specified try to use name of xdata
        if xlabel is not None:
            self.ax.set_xlabel(xlabel)
        else:
            try:
                self.ax.set_xlabel(x_data.name) 
            except:
                pass
                
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        else:
            try:
                self.ax.set_ylabel(y_data.name)
            except:
                pass
        ### switch on grid for line plots
        self.ax.grid(True)
    
if __name__ == '__main__':
    
    test_colorplot=True
    if(test_colorplot):
        ## make list of lists for colorplot testing:
            
        xdata=[]
        ydata=[]
        cdata=[]
        
        for i in range(100):
            xdata.append(np.linspace(0,10,200))
            ydata.append(np.ones(200)*i)
            cdata.append(np.sin(xdata[-1]*10/i))
            
        cplot_halfcolumn=colorplot(xdata,ydata,cdata,xlabel=r'$x$',ylabel=r'$y$',clabel=r'$c$',width_in_cols=0.5,aspect_ratio=0.7)
        cplot_onecolumn=colorplot(xdata,ydata,cdata,xlabel=r'$x$',ylabel=r'$y$',clabel=r'$c$',width_in_cols=1,aspect_ratio=1)
    
    
    
    test_multiline_plot=False
    
    if test_multiline_plot:
        ## make list of lists for colorplot testing:
            
        xdata=[]
        ydata=[]
        cdata=[]
        
        for i in range(10):
            xdata.append(np.linspace(0,10,200))
            ydata.append(np.sin(xdata[-1])*i)
            cdata.append(i)
            
            
        
        
        cplot_halfcolumn=multiline_plot(xdata,ydata,cdata,xlabel=r'$x$',ylabel=r'$y$',width_in_cols=0.5)
        cplot_onecolumn=multiline_plot(xdata,ydata,cdata,xlabel=r'$x$',ylabel=r'$y$',width_in_cols=1)