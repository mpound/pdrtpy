#!/usr/bin/env python


#todo: Look into seaborn https://seaborn.pydata.org
# Also https://docs.bokeh.org/en
# especially for coloring and style

import numpy as np
import numpy.ma as ma
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.colors as mpcolors
from matplotlib import ticker
from matplotlib.lines import Line2D

from astropy.io import fits
import astropy.wcs as wcs
import astropy.units as u
from astropy.nddata import NDDataArray, CCDData, NDUncertainty, StdDevUncertainty, VarianceUncertainty, InverseVariance
from astropy.visualization import simple_norm, ZScaleInterval , ImageNormalize
from astropy.visualization.stretch import SinhStretch,  LinearStretch


class PDRPlot:
    def __init__(self,toolbox):
        self._toolbox=toolbox
    
    def plotchisq(self,xaxis,xpix,ypix):
        """Make a line plot of chisq as a function of G0 or n for a given pixel"""
        axes = {"G0":0,"n":1}
        axis = axes[xaxis] #yep key error if you do it wrong
            
              
    def _zscale(self,image):
        norm= ImageNormalize(image.data,ZScaleInterval(),stretch=LinearStretch())
        return norm
    
    def plotReducedChisq(self,cmap='plasma',image=True, contours=True,
                         levels=None,measurements=None):
        self._plot(self_reducedChisq,cmap,image, contours,levels,measurements)
        
    def plotChisq(self,cmap='plasma',image=True, contours=True, 
                  levels=None, measurements=None):
        self._plot(self._chisq,cmap,image, contours,levels,measurements)
       
    def _plot(self,datasource,cmap, image, contours, levels, measurements):
        if type(datasource) == str:
            k = fits.open(datasource)[0]
        else:
            k=datasource.to_hdu()[0]
        min_ = k.data.min()
        max_ = k.data.max()
        ax=plt.subplot(111)
        ax.set_aspect('equal')
        normalizer = simple_norm(k.data, min_cut=min_,max_cut=max_, stretch='log', clip=False)
        xstart=k.header['crval1']
        xstop=xstart+k.header['naxis1']*k.header['cdelt1']
        ystart=k.header['crval2']
        ystop=ystart+k.header['naxis2']*k.header['cdelt2']
        #print(xstart,xstop,ystart,ystop)
    
        y = 10**np.linspace(start=ystart, stop=ystop, num=k.header['naxis2'])
        x = 10**np.linspace(start=xstart, stop=xstop, num=k.header['naxis1'])
        ax.set_xscale('log')
        ax.set_yscale('log')
        #if True:
        #    chihist = exposure.equalize_hist(k.data)
        #    ax.pcolormesh(x,y,chihist,cmap=cmap)
        if image:
            ax.pcolormesh(x,y,k.data,cmap=cmap,norm=normalizer)
        
        locmaj = ticker.LogLocator(base=10.0, subs=(1.0, ),numticks=10)
        ax.xaxis.set_major_locator(locmaj)
        locmin = ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*.1,numticks=10) 
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        #todo: allow unit conversion to cgs or Draine,  1 Draine = 1.69 G0 (Habing)
        # 1 Habing = 5.29x10−14 erg cm−3
        ylab = 'Log($G_0$) [Habing]'
        ss = 'Log($P_{th}/k$) [cm$^{-2}$]'
        xlab = 'Log(n) [cm$^{-3}$]'
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)

        if contours:
            if image==False: colors='black'
            else: colors='white'
            if levels is None:
                # Figure out some autolevels 
                # ------------------------------------------
                #  below resulted in poorly spaced levels
                # ------------------------------------------
                #if ( max_ - min_ ) < 101.0:
                #    steps = 'lin'
                #else:
                #    steps = 'log'
                # ------------------------------------------
                steps='log'
                contourset = ax.contour(x,y,k.data, levels=self._autolevels(k.data,steps),colors=colors)
            else:
                contourset = ax.contour(x,y,k.data, levels=levels, colors=colors)
                # todo: add contour level labelling
                # See https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contour_label_demo.html
        if measurements is not None:
            for m in measurements:
            #todo: use matplotlib contourf to shade the area between +/- error rather than use dashed lines
                lstyles = ['--','-','--']
                colors = [self._ratiocolor,self._ratiocolor,self._ratiocolor]
                for i in range(0,3):
                    #print(lstyles)
                    #print(len(x))
                    #print(len(y))
                    #print(len(k.data))
                    #print(colors)
                    cset = ax.contour(x,y,k.data,levels=m.levels,
                           linestyles=lstyles, colors=colors)
                

    def plotConfidenceIntervals(self,image=True, cmap='plasma',contours=True,levels=None,reduced=False):
        if reduced:  chi2_stat = stats.distributions.chi2.cdf(self._reducedChisq.data,self._dof)
        else:       chi2_stat = stats.distributions.chi2.cdf(self._chisq.data,self._dof)
        normalizer = simple_norm(chi2_stat, stretch='log', clip=False)
        if image: plt.imshow(chi2_stat*100,aspect='equal',cmap=cmap,norm=normalizer)
        print(100*chi2_stat.min(),100*chi2_stat.max())
        if contours:
            if levels==None:
                levels = [68.,80., 95., 99.]
            plt.contour(chi2_stat*100,colors='black',levels=sorted(levels))
    
    def plotRatiosOnModels(self):
        CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
        i =0 
        for x in self._modelfilesUsed:
            self._ratiocolor = CB_color_cycle[i]
            self._plot('models/'+self._modelfilesUsed[x],cmap='plasma',
                       measurements=[self._observedratios[x]],image=False,contours=False,levels=None)
            i = i+1
        # do some sort of legend
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in CB_color_cycle[0:i]]
        labels = list(self._modelfilesUsed.keys())
        plt.legend(lines, labels)
        plt.show()

    def makeRatioOverlays(self,cmap='plasma'):
        for x in self._modelfilesUsed:
            #self._ratiocolor='silver'
            self._ratiocolor='#4daf4a'
            self._plot('models/'+self._modelfilesUsed[x],cmap=cmap,
                       measurements=[self._observedratios[x]],image=True,contours=True,levels=None)
            plt.title(x)
            plt.show()
            plt.clf()
            
    def _autolevels(self,data,steps='log',numlevels=None):
        # tip of the hat to the WIP autolevels code lev.
        # http://admit.astro.umd.edu/wip ,  wip/src/plot/levels.c
        # CVS at http://www.astro.umd.edu/~teuben/miriad/install.html
        max_ =data.max()
        min_ = data.min()
        if numlevels is None:
            numlevels = int(0.5+3*(np.log(max_)-np.log(min_))/np.log(10))
        #print("levels start %d levels"%numlevels)
        # force number of levels to be between 5 and 15
        numlevels = max(numlevels,5)
        numlevels = min(numlevels,15)
    
        #print("Autolevels got %d levels"%numlevels)
        if steps[0:3] == 'lin':
            slope = (max_ - min_)/(numlevels-1)
            levels = np.array([min_+slope*j for j in range(0,numlevels)])
        elif steps[0:3] == 'log':
            # if data minimum is non-positive (shouldn't happen for models),
            # start log contours at lgo10(1) = 0
            if min_ <= 0: min_=1
            slope = np.log10(max_/min_)/(numlevels - 1)
            levels = np.array([min_ * np.power(10,slope*j) for j in range(0,numlevels)])
        else:
           raise Exception("steps must be 'lin' or 'log'")
        return levels
        
    
if __name__ == "__main__":
   print("foo")
