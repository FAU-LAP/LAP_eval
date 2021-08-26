# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 10:09:10 2021

@author: jo28dohe
"""


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

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
        
def slider_plot(fun, x_data=None, y_data=None,p_names=None,p_min_max_steps_dict=None,
                          const_params=[]):
    ### takes 
    
    if type(x_data) is pd.core.series.Series:
        print('reset x_plot,y_plot')
        x_name=x_data.name
        x_plot=x_data.reset_index()[x_name][1]
        
        y_name=y_data.name
        y_plot=y_data.reset_index()[y_name][1]
        
        print('xlen,ylen:',len(x_plot),len(y_plot))
        
    else:
        x_plot=x_data
        y_plot=y_data
    
    
    
    
    from matplotlib.widgets import Slider, Button
        
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.15, bottom=0.4)
    
    p_list=[]
    for key in p_names:
        min_val,max_val,steps=p_min_max_steps_dict[key]
        p_list.append((max_val+min_val)/2)
   
    func_vals=fun(x_plot,*p_list)
    data_line, = plt.plot(x_plot,y_plot,color='blue')
    func_line, = plt.plot(x_plot, func_vals,color='black', lw=2)
    ax.margins(x=0)
    
    axcolor = 'lightgoldenrodyellow'
    
    slider_ax_dict={}
    slider_dict={}
    h_max=0.35
    h_min=0.1
    h_step=(h_max-h_min)/(len(p_min_max_steps_dict))
    i=1
    for key in p_min_max_steps_dict.keys():
        
        min_val,max_val,steps = p_min_max_steps_dict[key]
        slider_ax_dict[key]=plt.axes([0.15, h_max-i*h_step, 0.65, h_step*0.7], facecolor=axcolor)
        slider_ax_dict[key].set_xlim(min_val,max_val)
        slider_dict[key]=Slider(slider_ax_dict[key],key,min_val,max_val,
                                valinit=(max_val+min_val)/2)
        i+=1
    
    
    def update(val):
        
        p_list=[]
        for key in p_names:
            p_list.append(slider_dict[key].val)
        print(p_list)
        func_line.set_ydata(fun(x_plot,*p_list))
        fig.canvas.draw_idle()
    
    ## connect sliders to update function
    for key in slider_ax_dict.keys():
        slider_dict[key].on_changed(update)
    
    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    
    def reset(event):
        for key in slider_dict.keys():
            slider_dict[key].reset()
    button.on_clicked(reset)
    
    
    ## extra functions if dataframe or list is given to function
    if type(y_data)==pd.core.series.Series:
        y_series=y_data
        print('ydata type is pandas Series, initialize further functions')
        ## reset width of parameter axes
        i=1
        for key in p_min_max_steps_dict.keys():
            slider_ax_dict[key].set_position([0.15, h_max-i*h_step, 0.3, h_step*0.7])
            i+=1
        
        ## initialize index sliders
        print(y_data.index.names)
        index_slider_dict={}
        index_slider_ax_dict={}
        i=1
        for iname in y_data.index.names:
            
            min_val,max_val = np.min(y_data.reset_index()[iname]),np.max(y_data.reset_index()[iname])
            index_slider_ax_dict[iname]=plt.axes([0.6, h_max-i*h_step, 0.3, h_step*0.7], facecolor=axcolor)
            index_slider_ax_dict[iname].set_xlim(min_val,max_val)
            index_slider_dict[iname]=Slider(index_slider_ax_dict[iname],iname,min_val,max_val,
                                    valinit=(max_val+min_val)/2)
            i+=1
            
        def update_index(val):
            
            def find_nearest(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return array[idx]
            
            index_list=[]
            for iname in y_data.index.names:
                index=find_nearest(y_data.reset_index()[iname],index_slider_dict[iname].val)
                index_list.append(index)
            
            x_plot=x_data[tuple(index_list)]
            y_plot=y_data[tuple(index_list)]
            
            data_line.set_xdata(x_plot)
            data_line.set_ydata(y_plot)
                 
            fig.canvas.draw_idle()
                
        for iname in y_data.index.names:
        
            index_slider_dict[iname].on_changed(update_index)
        
    plt.show()
    
if __name__ == '__main__':
    
    test_colorplot=False
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
        
    test_slider_plot=True
    if test_slider_plot:
        def fun(x,a,b,c):
            ret=a*np.sin(b*x)*np.exp(c*x)
            return(ret)
            
        xdata=np.linspace(0,10,200)
        ydata=fun(xdata,1,5,-0.2)
        noise=np.random.normal(scale=0.2,size=200)
        ysim=ydata+noise
        
        slider_plot(fun,xdata,ysim,p_names=['a','b','c'],
                                         p_min_max_steps_dict={'a':[0,2,40],'b':[0,10,40],'c':[-1,1,40]})
        
    test_slider_plot_with_df=False
    if test_slider_plot_with_df:
        def fun(x,a,b,c):
            ret=a*np.sin(b*x)*np.exp(c*x)
            return(ret)
        
        xdata_list=[]
        ydata_list=[]
        v_list=[]
        w_list=[]
        
        
        ## initialize a test dataframe
        for v in np.linspace(0,1):
            for w in np.linspace(0,5):
                v_list.append(v)
                w_list.append(w)
                xdata_list.append(np.linspace(0,10,200))
                ydata=fun(xdata_list[-1],v,w,-0.2)
                noise=np.random.normal(scale=0.2,size=200)
                ydata=ydata+noise
                ydata_list.append(ydata)
        ysim=ydata+noise
        
        df=pd.DataFrame(zip(v_list,w_list,xdata_list,ydata_list),columns=['v','w','xdata','ydata'])
        df.set_index(['v','w'],inplace=True)
        
        slider_plot(fun,df['xdata'],df['ydata'],p_names=['a','b','c'],
                                         p_min_max_steps_dict={'a':[0,2,40],'b':[0,10,40],'c':[-1,1,40]})
        
    