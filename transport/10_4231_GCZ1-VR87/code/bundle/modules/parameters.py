# parameters for the 4-orbital tight-binding model of monolayer WTe2 
# up to NNN hopping and with spin-orbit coupling
#
# The spinless hopping parameters were determined from a DFT fit (Ray)
#
# The leading symmetry-allowed SOC terms were determined using the Python package Qsymm.
#
# The unknown SOC parameters were fit to existing experimental data from ARPES measurements.

import numpy as np
from modules.pauli import s0,sx,sy,sz

# lattice constants in Angstrom
a = 3.477
b = 6.249

# tight-binding parameters in eV
mud = 0.24
mup = -2.25
tdAB = 0.51
tpAB = 0.40
t0AB = 0.39
tdx = -0.41 # td in the fit
tpx = 1.13 # dp in the fit
t0ABx = 0.29 # t0ABx2 in the fit

tpy = 0.15-0.02
t0x = 0.12+0.02  # t02 in the fit

# overall chemical potential
# shifted towards the center of the gap
mu = -0.47-0.03

# atomic positions (Wannier centers)
rBd = np.array([0.25*a, -0.32*b])  # W
rAd = np.array([-0.25*a, 0.32*b])  # W
rAp = np.array([-0.25*a, -0.07*b]) # Te
rBp = np.array([0.25*a, 0.07*b])   # Te

# hopping elements
# syntax: hop_dx_dy_subl2_subl1, 
# dx,dy = m1,0,1 (in terms of primitive lattice vectors)
# subl1, subl2 = WA,WB,TeA,TeB

hop_0_0_WB_TeA = -t0AB
hop_m1_0_WB_TeA = t0AB    
hop_0_0_TeB_WA = t0AB    
hop_m1_0_TeB_WA = -t0AB

hop_m1_0_WB_WB = tdx   
hop_m1_0_WA_WA = tdx
hop_m1_0_TeB_TeB = tpx   
hop_m1_0_TeA_TeA = tpx

hop_0_m1_WA_WB = tdAB  
hop_1_m1_WA_WB = tdAB 
hop_0_0_TeA_TeB = tpAB
hop_1_0_TeA_TeB = tpAB    

# additional terms
hop_0_m1_TeA_TeA = tpy
hop_0_m1_TeB_TeB = tpy 

hop_1_0_TeA_WA = t0x
hop_m1_0_TeA_WA = -t0x
hop_1_0_TeB_WB = t0x
hop_m1_0_TeB_WB = -t0x

hop_1_0_WB_TeA = -t0ABx
hop_m2_0_WB_TeA = t0ABx
hop_1_0_TeB_WA = t0ABx
hop_m2_0_TeB_WA = -t0ABx

# not allowed by symmetry
hop_0_0_WA_TeA = 0.
hop_0_0_WB_TeB = 0.
hop_0_m1_WA_TeA = 0.
hop_0_1_WB_TeB = 0.

# SOC parameters in eV
# obtained from fit to ARPES data around Gamma
# (parametrized by a scaling parameter lamb)
#
# signs of lambpy and lampdy are flipped with respect
# to fit results due to a redefinition of these parameters
# in the final Hamiltonian

# new fit
def lambdz(lamb): return lamb* (-0.008)
def lambdy(lamb): return lamb* (-0.031)
def lambpz(lamb): return lamb* (-0.010)
def lambpy(lamb): return lamb* (-0.040)

def lamb0ABz(lamb): return lamb* 0.0
def lamb0ABy(lamb): return lamb* 0.011
def lamb0ABx(lamb): return lamb* 0.0

def lamb0y(lamb): return lamb* 0.051
def lamb0z(lamb): return lamb* 0.012
def lamb0IIy(lamb): return lamb* 0.050
def lamb0IIz(lamb): return lamb* 0.012

# SOC matrix functions 
#(functions of lamb to tune the gap opening)

def SOC_m1_0_WA_WA(lamb, SOC_var=0):
    return 1.j*lambdz(lamb)*sz + 1.j*lambdy(lamb)*sy
    
def SOC_m1_0_WB_WB(lamb, SOC_var=0):
    return -1.j*lambdz(lamb)*sz - 1.j*lambdy(lamb)*sy

def SOC_m1_0_TeA_TeA(lamb, SOC_var=0):
    return 1.j*lambpz(lamb)*sz + 1.j*lambpy(lamb)*sy

def SOC_m1_0_TeB_TeB(lamb, SOC_var=0):
    return -1.j*lambpz(lamb)*sz - 1.j*lambpy(lamb)*sy


def SOC_0_0_WB_TeA(lamb, SOC_var=0): 
    return 1.j*lamb0ABz(lamb)*sz + 1.j*lamb0ABy(lamb)*sy + 1.j*lamb0ABx(lamb)*sx

def SOC_m1_0_WB_TeA(lamb, SOC_var=0):
    return 1.j*lamb0ABz(lamb)*sz + 1.j*lamb0ABy(lamb)*sy - 1.j*lamb0ABx(lamb)*sx

def SOC_0_0_TeB_WA(lamb, SOC_var=0):
    return 1.j*lamb0ABz(lamb)*sz + 1.j*lamb0ABy(lamb)*sy + 1.j*lamb0ABx(lamb)*sx
    
def SOC_m1_0_TeB_WA(lamb, SOC_var=0):
    return 1.j*lamb0ABz(lamb)*sz + 1.j*lamb0ABy(lamb)*sy - 1.j*lamb0ABx(lamb)*sx

  
def SOC_0_m1_WA_WB(lamb, SOC_var=0):
    return 0*s0

def SOC_1_m1_WA_WB(lamb, SOC_var=0):
    return 0*s0

def SOC_0_0_TeA_TeB(lamb, SOC_var=0):
    return 0*s0

def SOC_1_0_TeA_TeB(lamb, SOC_var=0):
    return 0*s0

def SOC_0_m1_TeA_TeA(lamb, SOC_var=0):
    return 0*s0

def SOC_0_m1_TeB_TeB(lamb, SOC_var=0):
    return 0*s0



def SOC_0_0_WA_TeA(lamb, SOC_var=0):
    return -1.j*lamb0z(lamb)*sz - 1.j*lamb0y(lamb)*sy

def SOC_0_0_WB_TeB(lamb, SOC_var=0):
    return 1.j*lamb0z(lamb)*sz + 1.j*lamb0y(lamb)*sy

def SOC_0_m1_WA_TeA(lamb, SOC_var=0):
    SOC_disorder = np.random.normal(loc=0, scale=SOC_var)
    return -1.j*lamb0IIz(lamb)*sz - 1.j*lamb0IIy(lamb)*sy + 1.j*SOC_disorder*sx

def SOC_0_1_WB_TeB(lamb, SOC_var=0):
    SOC_disorder = np.random.normal(loc=0, scale=SOC_var)
    return 1.j*lamb0IIz(lamb)*sz + 1.j*lamb0IIy(lamb)*sy + 1.j*SOC_disorder*sx



def SOC_1_0_TeA_WA(lamb, SOC_var=0):
    return 0*s0
    
def SOC_m1_0_TeA_WA(lamb, SOC_var=0):
    return 0*s0

def SOC_1_0_TeB_WB(lamb, SOC_var=0):
    return 0*s0

def SOC_m1_0_TeB_WB(lamb, SOC_var=0):
    return 0*s0

def SOC_1_0_WB_TeA(lamb, SOC_var=0):
    return 0*s0

def SOC_m2_0_WB_TeA(lamb, SOC_var=0):
    return 0*s0

def SOC_1_0_TeB_WA(lamb, SOC_var=0):
    return 0*s0

def SOC_m2_0_TeB_WA(lamb, SOC_var=0):
    return 0*s0