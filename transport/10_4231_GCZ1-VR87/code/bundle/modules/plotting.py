import kwant
import numpy as np
import matplotlib.pyplot as plt

def plot_lattice(system, lattice, show=True, hopp=True):
    """ plots the lattice of a system with a 4-site unit cell similar to the WTe2 model.
        The four sites are grouped into two sets WA,TeA and WB,TeB. W atoms are black,
        Te atoms are white.
        
        Parameters: system:  Builder
                      contains the system to be plotted
                    lattice:  Polyatomic
                      Bravais lattice of a system with a 4-site unit cell
                    show:  Boolean
                      Whether matplotlib.pyplot.show() is to be called, and the output is
                      to be shown immediately. Defaults to True.
        Returns: None
    """
    WA, TeA, WB, TeB = lattice.sublattices
    
    def family_colors(site):
        if site.family == WA:
            return 'blue'
        elif site.family == WB:
            return 'blue'
        elif site.family == TeA:
            return 'white'
        elif site.family == TeB:
            return 'white'
    
    def family_edgecolors(site):
        if site.family == WA:
            return 'blue'
        elif site.family == WB:
            return 'blue'
        elif site.family == TeA:
            return 'red'
        elif site.family == TeB:
            return 'red'
        
    if hopp==True:
        hop_lw = 0.1
    else:
        hop_lw = 0.0
                        
    kwant.plot(system, site_color=family_colors, site_edgecolor=family_edgecolors,
               site_lw=0.05, hop_lw=hop_lw, colorbar=False, fig_size=(10, 8), show=show)


def plot_wavefunction(system, lattice, vector, overlay=True, show=True, 
                           samples=5, cmap="jet", figsize=(15,30), ax=None):
    """ plots the density of a wavefunction associated with a finite system or a 
        system with leads in the WTe2 model
        
        Parameters: system: Builder
                        non-finalized system without leads
                    lattice:  Polyatomic
                        Bravais lattice of a system with a 4-site unit cell                       
                    vector: array
                        an eigenvector of the model
                    overlay: Boolean
                        whether to overlay the weights of the wavefunction with the
                        underlying lattice. Default is True
                    show: Boolean
                        Whether matplotlib.pyplot.show() is to be called, and the 
                        output is to be shown immediately. Default is True.
                        Is ignored if ax is not None.
                    samples: integer
                        number of oversampling points between lattice sites.
                        Default is 5.
                    cmap: matplotlib color map
                        The color map used for sites. Default is jet.
                    fig_size : tuple
                        Figure size (width, height) in inches. Default is (15,30).
                    ax : matplotlib.axes.Axes instance or None
                         If ax is not None, no new figure is created, but the plot 
                         is done within the existing Axes ax and show is ignored.

        Returns: None    
    """   
    rho = kwant.operator.Density(system.finalized())
    density = rho(vector)
    
    WA, TeA, WB, TeB = lattice.sublattices
    
    def family_colors(site):        
        if site.family == WA:
            return 1
        elif site.family == WB:
            return 1
        elif site.family == TeA:
            return 2
        elif site.family == TeB:
            return 2
    
    if show:
        plt.figure(figsize=figsize)
        ax1 = plt.subplot(111)
    else:
        ax1 = ax
        plt.gcf().set_size_inches(figsize[0], figsize[1])

    kwant.plotter.map(system.finalized(), density, 
                      colorbar=False, oversampling=samples, cmap=cmap, method='nearest',
                          show=False, ax=ax1, num_lead_cells=1)
    
    if overlay:
        kwant.plot(system, hop_lw=0, site_lw=0.1, site_color=family_colors, colorbar=False, 
                   show=False, ax=ax1)
    if show:
        plt.show()     
