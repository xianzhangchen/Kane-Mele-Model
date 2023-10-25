import kwant
import scipy
import numpy as np
import scipy.sparse.linalg as sla

from numpy import cos, sin, exp, pi

import modules.parameters as param
import modules.constants as const
from modules.pauli import s0,sx,sy,sz
import math

def WTe2_template(lamb=1.0, mu=0.0, peierls=False, SOC_var=0, SOC_spin_conserving=False):
    """ creates a bulk template of the 8-band tight-binding model of WTe2 with SOC
        which can be used to create various finite and ribbon geometries.
        
        Parameters: lamb: float
                        scaling parameter to tune SOC. In particular, the size of the gap can 
                        be tuned by lamb (lamb=0 gapless, lamb=1 physical WTe2)
                    mu: float
                        chemical potential with respect to the fixed chemical potential 
                        of the model 
                    peierls: Boolean
                        if an orbital magnetic field should be included
                   
        Returns:    bulk: Builder
                        unfinalized tight-binding system of WTe2
                    WTe2: Polyatomic
                        Bravais lattice of WTe2 model with 4 sites as a basis
                    bulk_sym: Symmetry 
                        contains the translational symmetries of the bulk
                        
        free parameters: Ez (Zeeman energy [in eV]), Bz (orbital field [in T])
    """
    
    WTe2 = kwant.lattice.general([(param.a,0),(0,param.b)],[param.rAd,param.rAp,
                                param.rBd,param.rBp],norbs=[2,2,2,2])
    WA, TeA, WB, TeB = WTe2.sublattices
    
    # Make a builder for a 2D system with a minimal unit cell.
    
    bulk_sym = kwant.TranslationalSymmetry(WTe2.vec((1, 0)), WTe2.vec((0, 1)))   # For 2D bulk trans inv
    bulk = kwant.Builder(bulk_sym)
    
    # define on-site functions  
       
    def onsite_WA(site, Ez, mu):
        return (param.mud - (param.mu + mu))*s0 + Ez*sz
    
    def onsite_TeA(site, Ez, mu):
        return (param.mup - (param.mu + mu))*s0 + Ez*sz
    
    def onsite_WB(site, Ez, mu):
        return (param.mud - (param.mu + mu))*s0 + Ez*sz
    
    def onsite_TeB(site, Ez, mu):
        return (param.mup - (param.mu + mu))*s0 + Ez*sz
    
    # Onsite terms
    bulk[[WA(0, 0)]] = lambda site, Ez: onsite_WA(site, Ez, mu)
    bulk[[TeA(0, 0)]] = lambda site, Ez: onsite_TeA(site, Ez, mu)
    bulk[[WB(0, 0)]] = lambda site, Ez: onsite_WB(site, Ez, mu)
    bulk[[TeB(0, 0)]] = lambda site, Ez: onsite_TeB(site, Ez, mu)
    
    # define hopping functions
    
    def hop(site1, site2, Ez, lamb, Bz, hop_elements, SOC_matrix):
        
        if SOC_spin_conserving: # TR-symmetric, spin-conserving disorder
            local_lamb = np.random.normal(loc=lamb, scale=SOC_var)
            hop_mat = hop_elements*s0 + SOC_matrix(local_lamb, SOC_var=0)
        else: # TR-symmetric, spin-nonconserving disorder
            hop_mat = hop_elements*s0 + SOC_matrix(lamb, SOC_var=SOC_var)
        
        if peierls: 
            return hop_mat*peierls_phase(site1, site2, Bz)
        else:
            return hop_mat

    def peierls_phase(site1, site2, Bz):
        # Bz in Tesla
        x1, y1 = site1.pos  # source of hop
        x2, y2 = site2.pos  # destination of hop
        # lattice site positions are measured in Å.
        lBinv = np.sqrt(const.e*Bz/const.hbar)*1.0e-10
        dx = x2 - x1
        dy = y2 - y1
        # periodic in x direction
        #phase = -(lBinv**2) * dy * (x1 + 0.5*dx)
        # periodic in y direction
        phase = -(lBinv**2) * dx * (y1 + 0.5*dy)
        return np.exp(1j*phase)
    
    # Hopping
    bulk[[kwant.builder.HoppingKind((0,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_0_0_WB_TeA, param.SOC_0_0_WB_TeA)  
    
    bulk[[kwant.builder.HoppingKind((-1,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_m1_0_WB_TeA, param.SOC_m1_0_WB_TeA)  
    
    bulk[[kwant.builder.HoppingKind((0,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_0_0_TeB_WA, param.SOC_0_0_TeB_WA)
    
    bulk[[kwant.builder.HoppingKind((-1,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_m1_0_TeB_WA, param.SOC_m1_0_TeB_WA)
    
    
    bulk[[kwant.builder.HoppingKind((-1,0),WA,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_m1_0_WA_WA, param.SOC_m1_0_WA_WA)
    
    bulk[[kwant.builder.HoppingKind((-1,0),WB,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_m1_0_WB_WB, param.SOC_m1_0_WB_WB)
    
    bulk[[kwant.builder.HoppingKind((-1,0),TeA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_m1_0_TeA_TeA, param.SOC_m1_0_TeA_TeA)
    
    bulk[[kwant.builder.HoppingKind((-1,0),TeB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_m1_0_TeB_TeB, param.SOC_m1_0_TeB_TeB)
    
    
    bulk[[kwant.builder.HoppingKind((0,-1),WA,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_0_m1_WA_WB, param.SOC_0_m1_WA_WB)
        
    bulk[[kwant.builder.HoppingKind((1,-1),WA,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_1_m1_WA_WB, param.SOC_1_m1_WA_WB)
      
    bulk[[kwant.builder.HoppingKind((0,0),TeA,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_0_0_TeA_TeB, param.SOC_0_0_TeA_TeB)
        
    bulk[[kwant.builder.HoppingKind((1,0),TeA,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_1_0_TeA_TeB, param.SOC_1_0_TeA_TeB)
    
    
    
    bulk[[kwant.builder.HoppingKind((0,-1),TeA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_0_m1_TeA_TeA, param.SOC_0_m1_TeA_TeA)
        
    bulk[[kwant.builder.HoppingKind((0,-1),TeB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_0_m1_TeB_TeB, param.SOC_0_m1_TeB_TeB)
    
    # new terms:
    
    bulk[[kwant.builder.HoppingKind((1,0),TeA,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_1_0_TeA_WA, param.SOC_1_0_TeA_WA)
    
    bulk[[kwant.builder.HoppingKind((-1,0),TeA,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_m1_0_TeA_WA, param.SOC_m1_0_TeA_WA)
    
    bulk[[kwant.builder.HoppingKind((1,0),TeB,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_1_0_TeB_WB, param.SOC_1_0_TeB_WB)
    
    bulk[[kwant.builder.HoppingKind((-1,0),TeB,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_m1_0_TeB_WB, param.SOC_m1_0_TeB_WB)
    
    
    bulk[[kwant.builder.HoppingKind((1,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_1_0_WB_TeA, param.SOC_1_0_WB_TeA)
    
    bulk[[kwant.builder.HoppingKind((-2,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_m2_0_WB_TeA, param.SOC_m2_0_WB_TeA)
    
    bulk[[kwant.builder.HoppingKind((1,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_1_0_TeB_WA, param.SOC_1_0_TeB_WA)
    
    bulk[[kwant.builder.HoppingKind((-2,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_m2_0_TeB_WA, param.SOC_m2_0_TeB_WA)
    
    
    bulk[[kwant.builder.HoppingKind((0,0),WA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_0_0_WA_TeA, param.SOC_0_0_WA_TeA)
    
    bulk[[kwant.builder.HoppingKind((0,0),WB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_0_0_WB_TeB, param.SOC_0_0_WB_TeB)
    
    bulk[[kwant.builder.HoppingKind((0,-1),WA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_0_m1_WA_TeA, param.SOC_0_m1_WA_TeA)
    
    bulk[[kwant.builder.HoppingKind((0,1),WB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, lamb, Bz, 
                                                param.hop_0_1_WB_TeB, param.SOC_0_1_WB_TeB)
    
    return bulk, WTe2, bulk_sym


def build_WTe2_to_lead_contact(x1=0.0, x2=10.0, mux1=0.0, mux2=0.0,
                               lamb1=1.0, lamb2=1.0, Ly=11, peierls=False):
    """ creates a rectangular region of WTe2 in which SOC and chemical potential are
        interpolated between the left and the right.
        
        Parameters: x1,x2: float
                        real space positions of the left and right edge of the rectangle
                    mux1, mux2: float
                        chemical potential on the left and on the right
                    lamb1, lamb2: float
                        SOC scaling parameter on the left and on the right
                    Ly: integer
                        height of the rectangle
                    peierls: Boolean
                        if an orbital magnetic field should be included
                                      
        Returns:    syst: Builder
                        unfinalized tight-binding system
                        
        free parameters: Ez (Zeeman energy [in eV]), Bz (orbital field [in T])
    """
   
    WTe2 = kwant.lattice.general([(param.a,0),(0,param.b)],[param.rAd,param.rAp,
                                param.rBd,param.rBp],norbs=[2,2,2,2])
    WA, TeA, WB, TeB = WTe2.sublattices

    syst = kwant.Builder()
    
    dW = x2 - x1
    dmu = mux2 - mux1
    dlamb = lamb2 - lamb1
    
    def rectangle(pos):
        (x, y) = pos
        return (x1 <= x < x2) and (-Ly*param.b/2 <= y < Ly*param.b/2)
    
    # define on-site functions  
    
    def onsite_WA(site, Ez):
        x,y = site.pos
        dx = x-x1
        return ((param.mud - (param.mu + mux1 + dx/dW*dmu))*s0 + Ez*sz)
    
    def onsite_TeA(site, Ez):
        x,y = site.pos
        dx = x-x1
        return ((param.mup - (param.mu + mux1 + dx/dW*dmu))*s0 + Ez*sz)
    
    def onsite_WB(site, Ez):
        x,y = site.pos
        dx = x-x1
        return ((param.mud - (param.mu + mux1 + dx/dW*dmu))*s0 + Ez*sz)
    
    def onsite_TeB(site, Ez):
        x,y = site.pos
        dx = x-x1
        return ((param.mup - (param.mu + mux1 + dx/dW*dmu))*s0 + Ez*sz)
    
    # Onsite terms
    syst[[WA.shape(rectangle, (x1+dW/2, 0))]] = onsite_WA
    syst[[TeA.shape(rectangle, (x1+dW/2, 0))]] = onsite_TeA
    syst[[WB.shape(rectangle, (x1+dW/2, 0))]] = onsite_WB
    syst[[TeB.shape(rectangle, (x1+dW/2, 0))]] = onsite_TeB
    
    # define hopping functions
    
    def hop(site1, site2, Ez, Bz, hop_elements, SOC_matrix):
        xs, ys = site1.pos  # source of hop
        xd, yd = site2.pos  # destination of hop
        
        dx = (xs + (xd-xs)/2) - x1
        lamb = lamb1 + dx/dW*dlamb
        
        hop_mat = hop_elements*s0 + SOC_matrix(lamb)
        if peierls: 
            return hop_mat*peierls_phase(site1, site2, Bz)
        else:
            return hop_mat

    def peierls_phase(site1, site2, Bz):
        # Bz in Tesla
        x1, y1 = site1.pos  # source of hop
        x2, y2 = site2.pos  # destination of hop
        # lattice site positions are measured in Å.
        lBinv = np.sqrt(const.e*Bz/const.hbar)*1.0e-10
        dx = x2 - x1
        dy = y2 - y1
        phase = -(lBinv**2) * dy * (x1 + 0.5*dx)
        return np.exp(1j*phase)
    
    # Hopping
    syst[[kwant.builder.HoppingKind((0,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_0_WB_TeA, param.SOC_0_0_WB_TeA)  
    
    syst[[kwant.builder.HoppingKind((-1,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_WB_TeA, param.SOC_m1_0_WB_TeA)  
    
    syst[[kwant.builder.HoppingKind((0,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_0_TeB_WA, param.SOC_0_0_TeB_WA)
    
    syst[[kwant.builder.HoppingKind((-1,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_TeB_WA, param.SOC_m1_0_TeB_WA)
    
    
    syst[[kwant.builder.HoppingKind((-1,0),WA,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_WA_WA, param.SOC_m1_0_WA_WA)
    
    syst[[kwant.builder.HoppingKind((-1,0),WB,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_WB_WB, param.SOC_m1_0_WB_WB)
    
    syst[[kwant.builder.HoppingKind((-1,0),TeA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_TeA_TeA, param.SOC_m1_0_TeA_TeA)
    
    syst[[kwant.builder.HoppingKind((-1,0),TeB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_TeB_TeB, param.SOC_m1_0_TeB_TeB)
    
    
    syst[[kwant.builder.HoppingKind((0,-1),WA,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_m1_WA_WB, param.SOC_0_m1_WA_WB)
        
    syst[[kwant.builder.HoppingKind((1,-1),WA,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_m1_WA_WB, param.SOC_1_m1_WA_WB)
      
    syst[[kwant.builder.HoppingKind((0,0),TeA,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_0_TeA_TeB, param.SOC_0_0_TeA_TeB)
        
    syst[[kwant.builder.HoppingKind((1,0),TeA,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_0_TeA_TeB, param.SOC_1_0_TeA_TeB)
    
    
    
    syst[[kwant.builder.HoppingKind((0,-1),TeA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_m1_TeA_TeA, param.SOC_0_m1_TeA_TeA)
        
    syst[[kwant.builder.HoppingKind((0,-1),TeB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_m1_TeB_TeB, param.SOC_0_m1_TeB_TeB)
    
    # new terms:
    
    syst[[kwant.builder.HoppingKind((1,0),TeA,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_0_TeA_WA, param.SOC_1_0_TeA_WA)
    
    syst[[kwant.builder.HoppingKind((-1,0),TeA,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_TeA_WA, param.SOC_m1_0_TeA_WA)
    
    syst[[kwant.builder.HoppingKind((1,0),TeB,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_0_TeB_WB, param.SOC_1_0_TeB_WB)
    
    syst[[kwant.builder.HoppingKind((-1,0),TeB,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_TeB_WB, param.SOC_m1_0_TeB_WB)
    
    
    syst[[kwant.builder.HoppingKind((1,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_0_WB_TeA, param.SOC_1_0_WB_TeA)
    
    syst[[kwant.builder.HoppingKind((-2,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m2_0_WB_TeA, param.SOC_m2_0_WB_TeA)
    
    syst[[kwant.builder.HoppingKind((1,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_0_TeB_WA, param.SOC_1_0_TeB_WA)
    
    syst[[kwant.builder.HoppingKind((-2,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m2_0_TeB_WA, param.SOC_m2_0_TeB_WA)
    
    
    syst[[kwant.builder.HoppingKind((0,0),WA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_0_WA_TeA, param.SOC_0_0_WA_TeA)
    
    syst[[kwant.builder.HoppingKind((0,0),WB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_0_WB_TeB, param.SOC_0_0_WB_TeB)
    
    syst[[kwant.builder.HoppingKind((0,-1),WA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_m1_WA_TeA, param.SOC_0_m1_WA_TeA)
    
    syst[[kwant.builder.HoppingKind((0,1),WB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_1_WB_TeB, param.SOC_0_1_WB_TeB)
    
    return syst

def build_WTe2_to_lead_contact_y(y1=0.0, y2=10.0, muy1=0.0, muy2=0.0,
                               lamb1=1.0, lamb2=1.0, Lx=11, peierls=False):
    """ creates a rectangular region of WTe2 in which SOC and chemical potential are
        interpolated between the top and the bottom. This function is for contacting
        leads in the y direction.
        
        Parameters: y1,y2: float
                        real space positions of the bottom and top edge of the rectangle
                    muy1, muy2: float
                        chemical potential at the bottom and at the top
                    lamb1, lamb2: float
                        SOC scaling parameter at the bottom and at the top
                    Lx: integer
                        width of the rectangle
                    peierls: Boolean
                        if an orbital magnetic field should be included
                                      
        Returns:    syst: Builder
                        unfinalized tight-binding system
                        
        free parameters: Ez (Zeeman energy [in eV]), Bz (orbital field [in T])
    """
   
    WTe2 = kwant.lattice.general([(param.a,0),(0,param.b)],[param.rAd,param.rAp,
                                param.rBd,param.rBp],norbs=[2,2,2,2])
    WA, TeA, WB, TeB = WTe2.sublattices

    syst = kwant.Builder()
    
    dh = y2 - y1
    dmu = muy2 - muy1
    dlamb = lamb2 - lamb1
    
    LWA = Lx-1
    LTeA = Lx-1
    LWB = Lx
    LTeB = Lx-2
    
    def rectangleWA(pos):
        (x, y) = pos
        L = Lx-1
        return (y1 <= y < y2) and (-L*param.a/2 <= x < L*param.a/2)
    
    def rectangleTeA(pos):
        (x, y) = pos
        L = Lx-1
        return (y1 <= y < y2) and (-L*param.a/2 <= x < L*param.a/2)
    
    def rectangleWB(pos):
        (x, y) = pos
        L = Lx
        return (y1 <= y < y2) and (-L*param.a/2 <= x < L*param.a/2)
    
    def rectangleTeB(pos):
        (x, y) = pos
        L = Lx-2
        return (y1 <= y < y2) and (-L*param.a/2 <= x < L*param.a/2)
    
    # define on-site functions  
    
    def onsite_WA(site, Ez):
        x,y = site.pos
        dy = y-y1
        return ((param.mud - (param.mu + muy1 + dy/dh*dmu))*s0 + Ez*sz)
    
    def onsite_TeA(site, Ez):
        x,y = site.pos
        dy = y-y1
        return ((param.mup - (param.mu + muy1 + dy/dh*dmu))*s0 + Ez*sz)
    
    def onsite_WB(site, Ez):
        x,y = site.pos
        dy = y-y1
        return ((param.mud - (param.mu + muy1 + dy/dh*dmu))*s0 + Ez*sz)
    
    def onsite_TeB(site, Ez):
        x,y = site.pos
        dy = y-y1
        return ((param.mup - (param.mu + muy1 + dy/dh*dmu))*s0 + Ez*sz)
    
    # Onsite terms
    syst[[WA.shape(rectangleWA, (0, y1+dh/2))]] = onsite_WA
    syst[[TeA.shape(rectangleTeA, (0, y1+dh/2))]] = onsite_TeA
    syst[[WB.shape(rectangleWB, (0, y1+dh/2))]] = onsite_WB
    syst[[TeB.shape(rectangleTeB, (0, y1+dh/2))]] = onsite_TeB
    
    # define hopping functions
    
    def hop(site1, site2, Ez, Bz, hop_elements, SOC_matrix):
        xs, ys = site1.pos  # source of hop
        xd, yd = site2.pos  # destination of hop
        
        dy = (ys + (yd-ys)/2) - y1
        lamb = lamb1 + dy/dh*dlamb
        
        hop_mat = hop_elements*s0 + SOC_matrix(lamb)
        if peierls: 
            return hop_mat*peierls_phase(site1, site2, Bz)
        else:
            return hop_mat

    def peierls_phase(site1, site2, Bz):
        # Bz in Tesla
        x1, y1 = site1.pos  # source of hop
        x2, y2 = site2.pos  # destination of hop
        # lattice site positions are measured in Å.
        lBinv = np.sqrt(const.e*Bz/const.hbar)*1.0e-10
        dx = x2 - x1
        dy = y2 - y1
        phase = -(lBinv**2) * dy * (x1 + 0.5*dx)
        return np.exp(1j*phase)
    
    # Hopping
    syst[[kwant.builder.HoppingKind((0,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_0_WB_TeA, param.SOC_0_0_WB_TeA)  
    
    syst[[kwant.builder.HoppingKind((-1,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_WB_TeA, param.SOC_m1_0_WB_TeA)  
    
    syst[[kwant.builder.HoppingKind((0,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_0_TeB_WA, param.SOC_0_0_TeB_WA)
    
    syst[[kwant.builder.HoppingKind((-1,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_TeB_WA, param.SOC_m1_0_TeB_WA)
    
    
    syst[[kwant.builder.HoppingKind((-1,0),WA,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_WA_WA, param.SOC_m1_0_WA_WA)
    
    syst[[kwant.builder.HoppingKind((-1,0),WB,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_WB_WB, param.SOC_m1_0_WB_WB)
    
    syst[[kwant.builder.HoppingKind((-1,0),TeA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_TeA_TeA, param.SOC_m1_0_TeA_TeA)
    
    syst[[kwant.builder.HoppingKind((-1,0),TeB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_TeB_TeB, param.SOC_m1_0_TeB_TeB)
    
    
    syst[[kwant.builder.HoppingKind((0,-1),WA,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_m1_WA_WB, param.SOC_0_m1_WA_WB)
        
    syst[[kwant.builder.HoppingKind((1,-1),WA,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_m1_WA_WB, param.SOC_1_m1_WA_WB)
      
    syst[[kwant.builder.HoppingKind((0,0),TeA,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_0_TeA_TeB, param.SOC_0_0_TeA_TeB)
        
    syst[[kwant.builder.HoppingKind((1,0),TeA,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_0_TeA_TeB, param.SOC_1_0_TeA_TeB)
    
    
    
    syst[[kwant.builder.HoppingKind((0,-1),TeA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_m1_TeA_TeA, param.SOC_0_m1_TeA_TeA)
        
    syst[[kwant.builder.HoppingKind((0,-1),TeB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_m1_TeB_TeB, param.SOC_0_m1_TeB_TeB)
    
    # new terms:
    
    syst[[kwant.builder.HoppingKind((1,0),TeA,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_0_TeA_WA, param.SOC_1_0_TeA_WA)
    
    syst[[kwant.builder.HoppingKind((-1,0),TeA,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_TeA_WA, param.SOC_m1_0_TeA_WA)
    
    syst[[kwant.builder.HoppingKind((1,0),TeB,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_0_TeB_WB, param.SOC_1_0_TeB_WB)
    
    syst[[kwant.builder.HoppingKind((-1,0),TeB,WB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m1_0_TeB_WB, param.SOC_m1_0_TeB_WB)
    
    
    syst[[kwant.builder.HoppingKind((1,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_0_WB_TeA, param.SOC_1_0_WB_TeA)
    
    syst[[kwant.builder.HoppingKind((-2,0),WB,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m2_0_WB_TeA, param.SOC_m2_0_WB_TeA)
    
    syst[[kwant.builder.HoppingKind((1,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_1_0_TeB_WA, param.SOC_1_0_TeB_WA)
    
    syst[[kwant.builder.HoppingKind((-2,0),TeB,WA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_m2_0_TeB_WA, param.SOC_m2_0_TeB_WA)
    
    
    syst[[kwant.builder.HoppingKind((0,0),WA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_0_WA_TeA, param.SOC_0_0_WA_TeA)
    
    syst[[kwant.builder.HoppingKind((0,0),WB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_0_WB_TeB, param.SOC_0_0_WB_TeB)
    
    syst[[kwant.builder.HoppingKind((0,-1),WA,TeA)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_m1_WA_TeA, param.SOC_0_m1_WA_TeA)
    
    syst[[kwant.builder.HoppingKind((0,1),WB,TeB)]] = \
    lambda site1, site2, Ez, Bz: hop(site1, site2, Ez, Bz, 
                                                param.hop_0_1_WB_TeB, param.SOC_0_1_WB_TeB)
    return syst

def simple_leads_template(mu=0.0, t=1.0):
    """ creates a simple bulk template for the leads that has the same lattice as WTe2.
        
        Parameters: t: float
                     hopping parameter for leads
                    mu: float
                     chemical potential of the lead
                    
        Returns:    bulk:     Builder
                        unfinalized tight-binding system of the leads
                    WTe2:     Polyatomic
                        Bravais lattice of WTe2 model with 4 sites as a basis
                    bulk_sym: Symmetry
                        contains the translational symmetries of the bulk
    """
    WTe2 = kwant.lattice.general([(param.a,0),(0,param.b)],[param.rAd,param.rAp,
                                param.rBd,param.rBp],norbs=[2,2,2,2])
    WA, TeA, WB, TeB = WTe2.sublattices
    
    # Make a builder for a 2D system with a minimal unit cell.
    bulk_sym = kwant.TranslationalSymmetry(WTe2.vec((1, 0)), WTe2.vec((0, 1)))   # For 2D bulk trans inv
    bulk = kwant.Builder(bulk_sym)
    
    # create lattice sites
    bulk[[WA(0, 0)]] = mu*s0
    bulk[[TeA(0, 0)]] = mu*s0
    bulk[[WB(0, 0)]] = mu*s0
    bulk[[TeB(0, 0)]] = mu*s0
    
    # Hopping
    bulk[[kwant.builder.HoppingKind((0,0),WB,TeA)]] = -t*s0   
    bulk[[kwant.builder.HoppingKind((-1,0),WB,TeA)]] = t*s0    
    bulk[[kwant.builder.HoppingKind((0,-1),WA,WB)]] = t*s0
    bulk[[kwant.builder.HoppingKind((1,-1),WA,WB)]] = t*s0    
    bulk[[kwant.builder.HoppingKind((0,0),TeB,WA)]] = t*s0
    bulk[[kwant.builder.HoppingKind((-1,0),TeB,WA)]] = -t*s0    
    bulk[[kwant.builder.HoppingKind((0,0),TeA,TeB)]] = t*s0
    bulk[[kwant.builder.HoppingKind((1,0),TeA,TeB)]] = t*s0
    
    return bulk, WTe2, bulk_sym


def Bloch_Ham(kx, ky, mu_ex=0.0, lamb=1.0, Ez=0.0):
    """ computes the Bloch Hamiltonian of the 8-band model for WTe2
        
        Parameters: kx,ky: float
                        crystal momentum at which the Bloch Hamiltonian is to be calculated
                    mu_ex: float
                        external chemical potential
                    lamb: float
                        SOC scaling parameter to tune the gap
                    Ez: float
                        Zeeman energy
                        
        Returns: H_Bloch: array
                        8x8 Bloch Hamiltonian matrix of the system
    """
    
    from modules.parameters import a,b,rBd,rAd,rBp,rAp
    from modules.parameters import mud,mup,mu
    from modules.parameters import tdx,tpx,tpy,tdAB,tpAB,t0AB
    from modules.parameters import tpy,t0ABx,t0x
    from modules.parameters import lambdz,lambdy,lambpz,lambpy
    from modules.parameters import lamb0ABz,lamb0ABy,lamb0ABx
    from modules.parameters import lamb0y,lamb0z,lamb0IIy, lamb0IIz
    
    H0 = np.zeros((4, 4), dtype=complex)
    k = np.array([kx, ky])
    
    # diagonal elements
    H0[0,0] += mud + 2.*tdx*cos(a*kx)
    H0[1,1] += mup + 2.*tpx*cos(a*kx)
    H0[2,2] += mud + 2.*tdx*cos(a*kx)
    H0[3,3] += mup + 2.*tpx*cos(a*kx)
    
    # additional NNN diagonal elements
    H0[1,1] += 2.*tpy*cos(b*ky)
    H0[3,3] += 2.*tpy*cos(b*ky)
    
    # off-diagonal elements
    H0[0,2] += tdAB*exp(1.j*k.dot(rAd-rBd))* (exp(-1.j*ky*b) + exp(1.j*(kx*a-ky*b)))
    H0[0,3] += t0AB*exp(1.j*k.dot(rAd-rBp))* (1. - exp(1.j*kx*a))
    H0[1,2] += t0AB*exp(1.j*k.dot(rAp-rBd))* (exp(1.j*kx*a) - 1.)
    H0[1,3] += tpAB*exp(1.j*k.dot(rAp-rBp))* (1. + exp(1.j*kx*a))
    
    # longer-range hopping 
    H0[0,3] += t0ABx*exp(1.j*k.dot(rAd-rBp))* (exp(-1.j*kx*a) - exp(2.j*kx*a))                                               
    H0[1,2] += t0ABx*exp(1.j*k.dot(rAp-rBd))* (exp(2.j*kx*a) - exp(-1.j*kx*a))
    
    H0[0,1] += t0x*exp(1.j*k.dot(rAd-rAp))* (exp(-1.j*kx*a) - exp(1.j*kx*a))
    H0[2,3] += t0x*exp(1.j*k.dot(rBd-rBp))* (exp(-1.j*kx*a) - exp(1.j*kx*a))
    
    # make the system Hermitian
    H0[2,0] = H0[0,2].conj()
    H0[3,0] = H0[0,3].conj()
    H0[2,1] = H0[1,2].conj()
    H0[3,1] = H0[1,3].conj()
    H0[1,0] = H0[0,1].conj()
    H0[3,2] = H0[2,3].conj()
    
    # shift of Fermi energy
    H0 -= (mu+mu_ex)*np.eye(4) 
    
    # add SOC
    HSOC = np.zeros((8,8), dtype=complex)
    mat_temp = np.zeros((4,4), dtype=complex)
    
    f = 2.*lambdz(lamb)* sin(a*kx)
    mat_temp[0,0] = f
    mat_temp[2,2] = -f
    HSOC += np.kron(sz,mat_temp)
    
    mat_temp *= 0.0
    f = 2.*lambdy(lamb)* sin(a*kx)
    mat_temp[0,0] = f
    mat_temp[2,2] = -f
    HSOC += np.kron(sy,mat_temp)
    
    mat_temp *= 0.0
    f = 2.*lambpz(lamb)* sin(a*kx)
    mat_temp[1,1] = f
    mat_temp[3,3] = -f
    HSOC += np.kron(sz,mat_temp)
    
    mat_temp *= 0.0
    f = 2.*lambpy(lamb)* sin(a*kx)
    mat_temp[1,1] = f
    mat_temp[3,3] = -f
    HSOC += np.kron(sy,mat_temp)
    
    mat_temp *= 0.0
    f = -1.j*lamb0ABz(lamb)* exp(1.j*k.dot(rAd-rBp))* (1. + exp(1.j*kx*a))
    mat_temp[0,3] = f
    mat_temp[1,2] = f
    mat_temp[2,1] = f.conj()
    mat_temp[3,0] = f.conj()
    HSOC += np.kron(sz,mat_temp)
    
    mat_temp *= 0.0
    f = -lamb0ABy(lamb)*1.j* exp(1.j*k.dot(rAd-rBp))* (1. + exp(1.j*kx*a))
    mat_temp[0,3] = f
    mat_temp[1,2] = f
    mat_temp[2,1] = f.conj()
    mat_temp[3,0] = f.conj()
    HSOC += np.kron(sy,mat_temp)
    
    mat_temp *= 0.0
    f = -1.j*lamb0ABx(lamb)* exp(1.j*k.dot(rAd-rBp))* (1. - exp(1.j*kx*a))
    mat_temp[0,3] = f
    mat_temp[1,2] = f
    mat_temp[2,1] = f.conj()
    mat_temp[3,0] = f.conj()
    HSOC += np.kron(sx,mat_temp)  
    
    # new NN SOC terms:    
    
    mat_temp *= 0.0
    f1 = -1.j* lamb0z(lamb) *exp(1.j*k.dot(rAd-rAp))
    f2 = 1.j* lamb0z(lamb) *exp(-1.j*k.dot(rAd-rAp))
    mat_temp[0,1] = f1
    mat_temp[2,3] = f2
    mat_temp[1,0] = f1.conj()
    mat_temp[3,2] = f2.conj()
    HSOC += np.kron(sz,mat_temp)
    
    mat_temp *= 0.0
    f1 = -1.j* lamb0y(lamb) *exp(1.j*k.dot(rAd-rAp))
    f2 = 1.j* lamb0y(lamb) *exp(-1.j*k.dot(rAd-rAp))
    mat_temp[0,1] = f1
    mat_temp[2,3] = f2
    mat_temp[1,0] = f1.conj()
    mat_temp[3,2] = f2.conj()
    HSOC += np.kron(sy,mat_temp)
    
    mat_temp *= 0.0
    f1 = -1.j* lamb0IIz(lamb) *exp(1.j*k.dot(rAd-rAp))* exp(-1.j*ky*b)
    f2 = 1.j* lamb0IIz(lamb) *exp(-1.j*k.dot(rAd-rAp))* exp(1.j*ky*b)
    mat_temp[0,1] = f1
    mat_temp[2,3] = f2
    mat_temp[1,0] = f1.conj()
    mat_temp[3,2] = f2.conj()
    HSOC += np.kron(sz,mat_temp)
    
    mat_temp *= 0.0
    f1 = -1.j* lamb0IIy(lamb) *exp(1.j*k.dot(rAd-rAp))* exp(-1.j*ky*b)
    f2 = 1.j* lamb0IIy(lamb) *exp(-1.j*k.dot(rAd-rAp))* exp(1.j*ky*b)
    mat_temp[0,1] = f1
    mat_temp[2,3] = f2
    mat_temp[1,0] = f1.conj()
    mat_temp[3,2] = f2.conj()
    HSOC += np.kron(sy,mat_temp)
    
    id4 = np.kron(s0,s0)
    H_Bloch = HSOC + np.kron(s0,H0) + Ez*np.kron(sz,id4)
   
    return H_Bloch
    
    
def compute_conductance(system, emin, emax, N=50):
    """ computes the conductance of a system with leads in units of e^2/h 
        using S-matrix formalism. 
        
        Parameters: system: Builder
                        non-finalized system with leads
                    emin, emax: float
                        the conductance is computed in the interval [emin,emax)
                    N: integer
                        number of energies for which the conductance is calculated
        Returns: energies: array
                     contains the energies for which the conductance is calculated
                 data: array
                     contains the computed conductance corresponding to energies                
    """
    energies = []
    data = []

    for ie in range(N):
        energy = emin + ie*(emax-emin)/N

        # compute the scattering matrix at a given energy
        smatrix = kwant.smatrix(system.finalized(), energy)

        # compute the transmission probability from lead 0 to lead 1
        energies.append(energy)
        data.append(smatrix.transmission(1, 0))
        print(energy, end='\r')
    
    return energies, data


def compute_eigenstates_finite(system, N=10, E=0.0):
    """ computes a few eigenstates and corresponding energies of a finite system
        close to a specified energy
        
        Parameters: system: Builder
                        non-finalized system without leads
                    N: integer
                        number of states to be found
                    E: float
                        energy close to which eigenstates are to be found
        Returns: evecs: array
                     contains eigenstates of the system. evecs[:,i] is the i-th 
                     eigenstate in the array
                 evals: array
                     contains the energies corresponding to the computed eigenstates              
    """    
    ham = system.finalized().hamiltonian_submatrix(sparse=True)
    evals, evecs = sla.eigsh(ham, k=N, sigma=E, return_eigenvectors=True)
        
    return evecs, evals


def compute_scattering_states(system, from_lead=0, E=0.0):
    """ computes the wave functions within a scattering region due to each 
        incoming mode of the given lead at a certain energy
        
        Parameters: system: Builder
                        non-finalized system with leads
                    from_lead: integer
                        the number of the lead that provides the incoming modes
                    E: float
                        energy of the incoming modes
        Returns: wf: array
                    the scattering states of the system. wf[:,i] is scattering
                    state corresponding to the i-th incoming mode
                 momenta: array
                    momenta corresponding to the incoming modes with negative velocities 
                 velocities: array
                    velocities corresponding to the incoming modes  
    """    
    lead = system.leads[from_lead]
    modes = lead.finalized().modes(energy=E)
    
    momenta = modes[0].momenta
    velocities = modes[0].velocities

    # only incoming modes with negative velocities
    mask = (velocities<=0)
    momenta = np.extract(mask,momenta)
    velocities = np.extract(mask,velocities)
    

    wfs = kwant.wave_function(system.finalized(), energy=E)
    wf = wfs(from_lead)

    return wf, momenta, velocities
        
def piecewise(x,template):
    """ Creates a piecewise defined function f(x) using an array of points 
        (xi,yi) as a template.
        
        The function is defined as follows:
        f(x)=y[0]  for x<x[0]
        f(x)=y[i]  for x[i]<=x<x[i+1], i<len(xs)
        f(x)=y[-1] for x>=x[-1]
        
        Parameters: x: array
                     arguments of the function f(x)
                    template: array
                     template for the piecewise function. Needs to be of
                     the shape (N,2) such that template[i] is a pair (xi,yi)
                     of points
                     
        Returns: f: array
                    values of the piecewise function. Has the same shape as x.
    """
    Nx = np.shape(template)[0]
    f = np.zeros(len(x))
    
    for i,xi in enumerate(x):
        if xi<template[0][0]:
            f[i] = template[0][1] 
            
        elif xi>=template[-1][0]:
            f[i] = template[-1][1]
            
        else:
            for j in range(Nx-1):
                if (template[j][0] <= xi < template[j+1][0]):
                    f[i] = template[j][1]
                    break
                    
    return f

def piecewise_periodic(x,template):
    """ Creates a periodic piecewise defined function f(x) using an array of points 
        (xi,yi) for which y[0]=y[-1].
        
        The function is defined as follows
        f(x)=y[i]  for x[i]<= xrel <x[i+1], i<len(xi)
            with xrel= floor((x - x[0])/period)*period
            and period = x[-1] - x[0]
             
        Parameters: x: array
                     arguments of the function f(x)
                    template: array
                     template for the piecewise function. Needs to be of
                     the shape (N,2) such that template[i] is a pair (xi,yi)
                     of points. Requires y[0]==y[-1] and the xi must be ordered.
                     
        Returns: f: array
                    values of the piecewise function. Has the same shape as x.
    """
    if template[0][1] == template[-1][1]:
        period = abs(template[-1][0]-template[0][0])
    else:
        raise Exception('First and last y value of the template must be the same.')
    
    x0 = template[0][0]
    Nx = np.shape(template)[0]
    f = np.zeros(len(x))
    check = False
    
    for i,xi in enumerate(x):
        xrel = xi - math.floor((xi - x0)/period)*period
        for j in range(Nx-1):
            if (template[j][0] <= xrel < template[j+1][0]):
                f[i] = template[j][1]
                check = True
                break 
        if check==False:
            raise Exception('There is something wrong with the intervals.')               
    return f


def biased_random_walk(p=1.0,N=100,t0=0.0,dt=1.0,dx=1.0,periodic=False, suppress_warning=False):
    """ Creates an array of points (t,x) where t is a time parameter
        and x(t) is a biased random walk.
        
        The biased random walk is defined as follows:
        Take a step with probability p, do nothing with probability (1-p).
        A step is taken with equal probability in the negative or positive
        x direction.
        
        Parameters: p: float (0<=p<=1)
                     proability of taking a step
                    N: integer
                     number of time steps
                    t0: float 
                     initial time
                    dt: float
                     length of a time step
                    dx: float
                     length of a step in the random walk
                    periodic: Boolean
                     whether the random walk is continued after N steps until
                     it returns to the initial point. In this case, the walk
                     has a random length L>N.
                    suppress_warning: Boolean
                     whether a warning is printed in case the random walk does not
                     return to its starting point
                                        
        Returns: walk: array
                    contains the points [t,x(t)], i.e., the shape of the
                    array is (N,2).
    """
    
    def make_step(t0,x0,p,dt,dx):
        # decide whether a step should be taken 
        if (0.0<p<1.0):
            act = np.random.sample()
            if (act<p):
                step_t = np.random.randint(0,2)
            else:
                step_t = 2       
        elif p==0.0:
            step_t = 2
        elif p==1.0:
            step_t = np.random.randint(0,2)
        else:
            raise Exception('p must be in the interval [0,1].')
        
        # take (or not take) the step and return the new point
        if step_t == 1:
            return t0+dt, x0+dx
        elif step_t == 0:
            return t0+dt, x0-dx
        else:
            return t0+dt, x0
       
    x0 = 0.0
    walk = [[t0,x0]]
    returned = False
    
    if periodic==False:
        for j in range(N-1):          
            t1,x1 = make_step(walk[j][0],walk[j][1],p,dt,dx)
            walk.append([t1,x1])
    else:
        N_exit = 100*N # exit threshold
        dx_exit = dx/100.      
        for j in range(N_exit-1):
            t1,x1 = make_step(walk[j][0],walk[j][1],p,dt,dx)
            walk.append([t1,x1])
            if (j+1 >= N-1) and (x0-dx_exit < x1 < x0+dx_exit):
                walk[j+1]=[t1,x0]
                returned = True
                break
    
    if periodic and not(returned) and not(suppress_warning):
        print("WARNING: the random walk did not return to its starting point.")
    
    return np.array(walk)


def potential_random_walk(p=1, N=100, t0=0, dt=1, dx=1, potential=0, periodic=False):
    """ Creates an array of points (t,x) where t is a time parameter
        and x(t) is a biased random walk.
        
        The random walk is defined as follows:
        Take a step with probability p, do nothing with probability (1-p).
        A step is taken with in the positive or negative x direction with
        probability 1/2 - potential*x and 1/2 + potential*x respectively,
        mimicking the effect of a quadratic potential pulling back to x=0.
        
        Parameters: p: float (0<=p<=1)
                     proability of taking a step
                    N: integer
                     number of time steps
                    t0: float 
                     initial time
                    dt: float
                     length of a time step
                    dx: float
                     length of a step in the random walk
                    periodic: Boolean
                     whether the random walk is continued after N steps until
                     it returns to the initial point. In this case, the walk
                     has a random length L>N.
                    
        Returns: walk: array
                    contains the points [t,x(t)], i.e., the shape of the
                    array is (N,2).
    """
    if not 0<=p<=1:
        raise Exception('p must be in the interval [0,1].')

    def make_step(t0, x0):
        # decide whether a step should be taken 
        act = np.random.sample()
        if act < p:
            direction = np.random.sample()
            step_t = (1 if direction < 1/2 - x0*potential else -1)
        else:
            step_t = 0

        # take (or not take) the step and return the new point
        return t0 + dt, x0 + step_t * dx

    x0 = 0.0
    walk = [[t0,x0]]
    returned = False

    if periodic==False:
        for j in range(N-1):          
            t1,x1 = make_step(walk[j][0], walk[j][1])
            walk.append([t1,x1])
    else:
        N_exit = 100*N # exit threshold
        dx_exit = dx/100.      
        for j in range(N_exit-1):
            t1,x1 = make_step(walk[j][0], walk[j][1])
            walk.append([t1,x1])
            if (j+1 >= N-1) and (x0-dx_exit < x1 < x0+dx_exit):
                walk[j+1]=[t1,x0]
                returned = True
                break
    
    if periodic and not(returned):
        print("WARNING: the random walk did not return to its starting point.")
    
    return np.array(walk)


def periodic_random_walk(p=1.0, N=100, t0=0.0, dt=1.0, dx=1.0, fix_start=True):
    """ Creates an array of points (t,x) where t is a time parameter
        and x(t) is a biased random walk that is N long and returns
        to the starting point.

        The biased random walk is defined as follows:
        Take a step with probability p, do nothing with probability (1-p).
        A step is taken with equal probability in the negative or positive
        x direction.

        Parameters: p: float (0<=p<=1)
                     proability of taking a step
                    N: integer
                     number of time steps
                    t0: float
                     initial time
                    dt: float
                     length of a time step
                    dx: float
                     length of a step in the random walk
                    fix_start:
                     whether to fix both the start and end to be at x = 0.
                     If True, the first and last points are fixed and only
                     N-1 steps are taken.
                     If False, the first point can already be at x = +-1.

        Returns: walk: array
                    contains the points [t,x(t)], i.e., the shape of the
                    array is (N,2).
    """
    fix_start = int(fix_start)
    # Pick where to make steps
    n_steps = -1
    if (N-fix_start) % 2 == 1 and np.isclose((1-p) * N, 0, atol=1e-3) and p != 1:
        print('If p is very close to 1 and the number of steps is odd the algorithm may take very long.')

    if 0. <= p < 1.:
        # Make sure it is an even number
        while n_steps % 2 == 1:
            step_mask = np.random.random((N-fix_start, )) < p
            n_steps = np.sum(step_mask)
        step_inds,  = step_mask.nonzero()
        # Pick exactly half of them for right steps
        if len(step_inds) != 0:
            right_inds = np.random.choice(step_inds, n_steps//2, replace=False)
        else:
            right_inds = []
    elif p == 1:
        step_mask = np.full(N-fix_start, True)
        if (N-fix_start) % 2 == 1: step_mask[-1] = False
        # ensure that there is an even number of steps
        n_steps = np.sum(step_mask)
        step_inds,  = step_mask.nonzero()
        # Pick exactly half of them for right steps
        right_inds = np.random.choice(step_inds, n_steps//2, replace=False)
    else:
        raise Exception('p must be in the interval [0,1].')
        
    # Generate the step sequence
    step_list = np.zeros((N, ), dtype=int)
    if len(step_inds) != 0: 
        step_list[step_inds + fix_start] = -1
        step_list[right_inds + fix_start] = 1
        assert np.sum(step_list) == 0

    ts = np.linspace(t0, t0 + N * dt, N, endpoint=False)
    walk = np.array([ts, dx * np.cumsum(step_list)]).T

    return walk
