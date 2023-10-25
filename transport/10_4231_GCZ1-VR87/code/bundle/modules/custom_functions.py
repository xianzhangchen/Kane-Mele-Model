
import kwant, sys, math
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import tinyarray as tiny
import ipyparallel as ipp

import modules.parameters as param
import modules.functions as func
import modules.shapes as shapes

from modules.pauli import s0,sx,sy,sz
from numpy import cos, sin, exp, pi

# Define quantum point contact (QPC) shape
def QPC_shape(site, W=100, L=200, L_gap=10, L_base=50, L_flat=30, x0=0, y0=0, vertical=True):
    x, y = site.pos
    x, y = x - x0 + 1, y - y0
    
    if not vertical:
        temp = x
        x = y
        y = temp
        
    if -W/2 >= x or x >= W/2:
        return False
    
    if -L_base/2 <= y <= L_base/2:
        
        if -L_flat/2 <= y <= L_flat/2:
            return not (-W/2 <= x <= -L_gap/2 or L_gap/2 <= x <= W/2)
        else:
            width = (L_base - L_flat)/2
            height = (W - L_gap)/2
            
            if -L_base/2 <= y <= -L_flat/2:
                gaussian = height*np.exp(-8*(y + L_flat/2)**2/width**2)
                return not (-W/2 <= x <= -W/2 + gaussian or W/2 - gaussian <= x <= W/2)
            else:
                gaussian = height*np.exp(-8*(y - L_flat/2)**2/width**2)
                return not (-W/2 <= x <= -W/2 + gaussian or W/2 - gaussian <= x <= W/2)
            
    elif -L/2 <= y <= L/2:
        return True
    
    else:
        return False
    
# Create quantum point-contact (QPC) system 
def QPC_system(L=300, W=200, Wlead=200, mu=0.0, mu_lead=-0.1, lamb=0.06, lamb_lead = 0.0,
               zigzag=False, sym_vec=s0, L_gap=50, L_base=200, L_flat=100):

    bulk, WTe2, bulk_sym = func.WTe2_template(lamb=lamb, mu=mu)
    bulk_lead, WTe2, bulk_sym = func.WTe2_template(lamb=lamb_lead, mu=mu_lead) # metal

    syst = kwant.Builder()

    syst.fill(bulk, lambda pos: QPC_shape(pos, W=W, L=L, L_gap=L_gap, L_base=L_base, L_flat=L_flat),
              (0, 0), max_sites=float('inf'))

    '''
    sym_lead = bulk_sym.subgroup((0, 1))
    lead1 = kwant.Builder(sym_lead, conservation_law=sym_vec)
    lead1.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead/param.a), (0, 0), max_sites=float('inf'))
    syst.attach_lead(lead1)

    sym_lead = bulk_sym.subgroup((0, -1))
    lead2 = kwant.Builder(sym_lead, conservation_law=sym_vec)
    lead2.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead/param.a), (0, 0), max_sites=float('inf'))
    syst.attach_lead(lead2)
    '''

    dx = (W-Wlead)/2
    
    #attach top left lead
    sym_lead = bulk_sym.subgroup((0,1))
    lead_tl = kwant.Builder(sym_lead, conservation_law=sym_vec)
    lead_tl.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead/param.a, x0=-dx/param.a), (-dx,0), max_sites=float('inf'))
    syst.attach_lead(lead_tl)
                     
    #attach top right lead
    sym_lead = bulk_sym.subgroup((0,1))
    lead_tr = kwant.Builder(sym_lead, conservation_law=sym_vec)
    lead_tr.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead/param.a, x0=dx/param.a), (dx,0), max_sites=float('inf'))
    syst.attach_lead(lead_tr)
    
    #attach bottom left lead
    sym_lead = bulk_sym.subgroup((0,-1))
    lead_bl = kwant.Builder(sym_lead, conservation_law=sym_vec)
    lead_bl.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead/param.a, x0=-dx/param.a), (-dx,0), max_sites=float('inf'))
    syst.attach_lead(lead_bl)
                     
    #attach bottom right lead
    sym_lead = bulk_sym.subgroup((0,-1))
    lead_br = kwant.Builder(sym_lead, conservation_law=sym_vec)
    lead_br.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead/param.a, x0=dx/param.a), (dx,0), max_sites=float('inf'))
    syst.attach_lead(lead_br)
    
    return syst, WTe2


# Calculate the spin-resolved, charge, and spin currents in an n-terminal system with
# ferromagnetic leads using Landauer-like formula
#
# syst: n-terminal system
# voltages: nx2 array of voltages; first index is for terminal, second index is for spin up/down
# E: Fermi energy of the sample
# Returns nx2 array of spin-resolved currents, nx1 array of charge currents, and nx1 array of spin currents
def currents_Landauer_nTerm(syst, voltages, E=0, params=dict(Ez=0, Bz=0)):
    
    #calculate scattering matrix
    smatrix = kwant.smatrix(syst, E, params=params)
    leads = smatrix.lead_info
    
    num_leads = len(leads)
    num_modes = np.array([leads[i].block_nmodes for i in range(num_leads)])
    
    #calculate spin-resolved currents
    resolved_currents = np.zeros((num_leads, 2)) #same indexing as voltages
    for term_index in range(num_leads):
        for spin_index in range(2):
            
            current = -num_modes[term_index, spin_index]*voltages[term_index][spin_index]
            
            for term_index2 in range(num_leads):
                for spin_index2 in range(2):
                    
                    T = smatrix.transmission((term_index, spin_index), (term_index2, spin_index2))
                    V = voltages[term_index2][spin_index2]
                    current += T*V
            
            resolved_currents[term_index, spin_index] = current
    
    #calculate charge and spin currents in each terminal
    charge_currents = resolved_currents[:, 0] + resolved_currents[:, 1]
    spin_currents = resolved_currents[:, 0] - resolved_currents[:, 1]
    
    return resolved_currents, charge_currents, spin_currents


# Calculate the spin-resolved, charge, and spin conductances in an n-terminal system with
# ferromagnetic leads using Landauer-like formula
#
# syst: n-terminal system
# V_test: nonzero test voltage to calculate conductance values
# E: Fermi energy of the sample
# Returns 2nx2n spin-resolved conductance matrix, nx2n charge conductance matrix, and nx2n spin conductance matrix
def conductances_Landauer_nTerm(syst, V_test=.05, E=0, params=dict(Ez=0, Bz=0)):
    
    num_leads = len(syst.leads)
    Gr = np.zeros((2*num_leads, 2*num_leads))  #spin-resolved conductance matrix
    Gc = np.zeros((num_leads, 2*num_leads))    #charge conductance matrix
    Gs = np.zeros((num_leads, 2*num_leads))    #spin conductance matrix
    
    for lead_index in range(num_leads):
        for spin_index in range(2):
            voltages = np.zeros((num_leads, 2))
            voltages[lead_index, spin_index] = V_test
            
            resolved_currents, charge_currents, spin_currents = currents_Landauer_nTerm(syst, voltages, E=E, params=params)
            Gr[:, 2*lead_index + spin_index] = resolved_currents.flatten()/V_test
            Gc[:, 2*lead_index + spin_index] = charge_currents/V_test
            Gs[:, 2*lead_index + spin_index] = spin_currents/V_test
            
    return Gr, Gc, Gs

# Calculate spin transmission conductance of four-terminal system
def transmission_conductance(syst, V_test=.05, E=0, params=dict(Ez=0, Bz=0)):

    voltages = np.array([[V_test/2, V_test/2], [-V_test/2, -V_test/2], [0, 0], [0, 0]])
    resolved_currents, charge_currents, spin_currents = currents_Landauer_nTerm(syst, voltages, E=E, params=params)
    Gs_T = (spin_currents[2] + spin_currents[3])/V_test
    
    return Gs_T


# Add Anderson disorder to the system
# mu: mean perturbation strength
# sigma: stand. dev. of peturbation strength
# alligned: whether the spins of the perturbations are alligned or not; false for random spin directions
# spin: direction of spin allignment; unused if alligned == False
# shape: optional function to specify a region for the disorder
def add_Anderson_disorder(syst, lattice, mu, sigma, alligned=False, spin=s0, shape=None, origin=(0,0), Ez=0, delete=False):
    
    if shape == None:
        sites = syst.sites()
    else:
        sites = list(lattice.shape(shape, origin)(syst))
        
    if delete: #delete sites for testing purposes
        
        for site in sites:
            try:
                del syst[site]
            except: pass
    else:
        if alligned: #perturbations alligned in spin direction

            for site in sites:
                try:
                    syst[site] = syst[site](site, Ez) + np.random.normal(loc=mu, scale=sigma)*spin
                except: pass

        else:  #perturbations are a random mix of Sx, Sy, and Sz

            for site in sites:

                #generate random point on 3-sphere using three independent Gaussians
                unit_vec = np.random.normal(loc=0, scale=1, size=3)
                unit_vec = unit_vec/np.linalg.norm(unit_vec)

                #add perturbation of random magnitude in that direction
                mag = np.random.normal(loc=mu, scale=sigma)
                try:
                    syst[site] = syst[site](site, Ez) + mag*(unit_vec[0]*sx + unit_vec[1]*sy + unit_vec[2]*sz)
                except: pass

    return syst


# Add onsite barrier to the system
# onsite: onsite potential term
# shape: function which accepts pos=(x,y) and returns whether pos is in the barrier
def add_barrier(syst, lattice, shape, origin=(0,0), onsite=.1*s0, Ez=0):
    
    sites = list(lattice.shape(shape, origin)(syst))
    for site in sites:
        syst[site] = syst[site](site, Ez) + onsite
        #del syst[site]
    
    return syst
        
        
# Rectangle shape function for add_barrier
def rectangle(pos, x1=0, x2=1, y1=0, y2=1):
    x, y = pos
    return x1 <= x <= x2 and y1 <= y <= y2

# Cross shape function for add_barrier
def cross(pos, horz_x=10, horz_y=5, vert_x=5, vert_y=10):
    x, y = pos
    return (-horz_x/2 <= x <= horz_x/2 and -horz_y/2 <= y <= horz_y/2) or (-vert_x/2 <= x <= vert_x/2 and -vert_y/2 <= y <= vert_y/2)


def build_x_ribbon(L=40, L_contact=10, W=41, Wlead=31, mu=0.0, mu_lead=-0.1, 
                   lamb=1.0, lamb_lead = 1.0, edge1=lambda x: 0.0, edge2=lambda x: 0.0, sym_vec=s0):

        # odd width W for W termination
        # even width W for Te termination 

        bulk, WTe2, bulk_sym = func.WTe2_template(lamb=lamb, mu=mu)
        bulk_lead, WTe2, bulk_sym = func.WTe2_template(lamb=lamb_lead, mu=mu_lead) # metal

        syst = kwant.Builder()
        syst.fill(bulk, shapes.x_ribbon_finite_irregular(Lx=L, Ly=W, edge1=edge1, edge2=edge2),
                                (0, 0), max_sites=float('inf'))  
        
        
        
        #### Sample to lead contact ####
        '''
        x2 = -L*param.a/2 + param.a + param.a
        x1 = x2 - L_contact*param.a
        mux1 = mu_lead
        mux2 = mu
        lamb1 = lamb_lead
        lamb2 = lamb
        left_contact = func.build_WTe2_to_lead_contact(x1=x1, x2=x2, mux1=mux1, mux2=mux2,
                                                       lamb1=lamb1, lamb2=lamb2, Ly=W)

        syst.update(left_contact)

        x1 = L*param.a/2 - param.a - param.a
        x2 = x1 + L_contact*param.a
        mux1 = mu
        mux2 = mu_lead
        lamb1 = lamb
        lamb2 = lamb_lead
        right_contact = func.build_WTe2_to_lead_contact(x1=x1, x2=x2, mux1=mux1, mux2=mux2,
                                                       lamb1=lamb1, lamb2=lamb2, Ly=W)

        syst.update(right_contact)
        '''
        ################################
        
        sym_lead = bulk_sym.subgroup((2, 0))
        lead1 = kwant.Builder(sym_lead, conservation_law=sym_vec)
        lead1.fill(bulk_lead, shapes.x_ribbon(Ly=Wlead), (0, 0), max_sites=float('inf'))
        syst.attach_lead(lead1)

        sym_lead = bulk_sym.subgroup((-2, 0))
        lead2 = kwant.Builder(sym_lead, conservation_law=sym_vec)
        lead2.fill(bulk_lead, shapes.x_ribbon(Ly=Wlead), (0, 0), max_sites=float('inf'))
        syst.attach_lead(lead2)

        return syst, WTe2
    
    
def build_y_ribbon(L=40, L_contact=10, W=41, Wlead=31, mu=0.0, mu_lead=-0.1, 
                   lamb=0.06, lamb_lead = 0.0, zigzag=False, sym_vec=s0, sym_vec2=-s0):
        
        if sym_vec2 == -s0:
            sym_vec2 = sym_vec
        
        bulk, WTe2, bulk_sym = func.WTe2_template(lamb=lamb, mu=mu)
        bulk_lead, WTe2, bulk_sym = func.WTe2_template(lamb=lamb_lead, mu=mu_lead) # metal

        syst = kwant.Builder()
        
        if zigzag:
            syst.fill(bulk, shapes.rectangle_zigzag_y(lat=WTe2, Lx=W, Ly=L),
                                (0, 0), max_sites=float('inf'))
        else:
            syst.fill(bulk, shapes.rectangle_straight_y_(lat=WTe2, Lx=W, Ly=L),
                                (0, 0), max_sites=float('inf'))  

        #### Sample to lead contact ####
        '''
        y2 = -L*param.b/2 + param.b
        y1 = y2 - L_contact*param.b
        muy1 = mu_lead
        muy2 = mu
        lamb1 = lamb_lead
        lamb2 = lamb
        left_contact = func.build_WTe2_to_lead_contact_y(y1=y1, y2=y2, muy1=muy1, muy2=muy2,
                                                       lamb1=lamb1, lamb2=lamb2, Lx=W)

        syst.update(left_contact)

        y1 = L*param.b/2 - param.b
        y2 = y1 + L_contact*param.b
        muy1 = mu
        muy2 = mu_lead
        lamb1 = lamb
        lamb2 = lamb_lead
        right_contact = func.build_WTe2_to_lead_contact_y(y1=y1, y2=y2, muy1=muy1, muy2=muy2,
                                                       lamb1=lamb1, lamb2=lamb2, Lx=W)

        syst.update(right_contact)
        '''
        ################################
            
        sym_lead = bulk_sym.subgroup((0, 1))
        lead1 = kwant.Builder(sym_lead, conservation_law=sym_vec)
        if zigzag:
            lead1.fill(bulk_lead, shapes.y_ribbon_zigzag(lat=WTe2, Lx=Wlead), (0, 0), max_sites=float('inf'))
        else:
            lead1.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead), (0, 0), max_sites=float('inf'))
        syst.attach_lead(lead1)

        sym_lead = bulk_sym.subgroup((0, -1))
        lead2 = kwant.Builder(sym_lead, conservation_law=sym_vec2)
        if zigzag:
            lead2.fill(bulk_lead, shapes.y_ribbon_zigzag(lat=WTe2, Lx=Wlead), (0, 0), max_sites=float('inf'))
        else:
            lead2.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead), (0, 0), max_sites=float('inf'))
        syst.attach_lead(lead2)

        return syst, WTe2


def build_Hjunction(Lx=300, Ly=400, Wlead=100, Llead=40, mu=0, mu_lead=-.4, zigzag=False, sym_vec=s0, SOC_var=0, SOC_spin_conserving=False):
    
    bulk, WTe2, bulk_sym = func.WTe2_template(lamb=1, mu=mu, SOC_var=SOC_var, SOC_spin_conserving=SOC_spin_conserving)
    bulk_lead, WTe2, bulk_sym = func.WTe2_template(lamb=0, mu=mu_lead)
    syst = kwant.Builder()
    
    #build sample
    dx = (Lx-Wlead)/2
    if zigzag:
        rect = shapes.rectangle_zigzag_y(lat=WTe2, Lx=Lx/param.a, Ly=Ly/param.b)
        term_l = shapes.rectangle_zigzag_y(lat=WTe2, Lx=Wlead/param.a, Ly=(Ly+2*Llead)/param.b, x0=-dx/param.a)
        term_r = shapes.rectangle_zigzag_y(lat=WTe2, Lx=Wlead/param.a, Ly=(Ly+2*Llead)/param.b, x0=dx/param.a)
        
        bulk_shape = lambda site: rect(site) or term_l(site) or term_r(site)
        syst.fill(bulk, bulk_shape, (-dx,0), max_sites=float('inf'))
    else:
        rect = shapes.rectangle_straight_y_(lat=WTe2, Lx=Lx/param.a, Ly=Ly/param.b)
        term_l = shapes.rectangle_straight_y_(lat=WTe2, Lx=Wlead/param.a, Ly=(Ly+2*Llead)/param.b, x0=-dx/param.a)
        term_r = shapes.rectangle_straight_y_(lat=WTe2, Lx=Wlead/param.a, Ly=(Ly+2*Llead)/param.b, x0=dx/param.a)
        
        bulk_shape = lambda site: rect(site) or term_l(site) or term_r(site)
        syst.fill(bulk, bulk_shape, (-dx,0), max_sites=float('inf'))
        
    #attach top left lead
    sym_lead = bulk_sym.subgroup((0,1))
    lead_tl = kwant.Builder(sym_lead, conservation_law=sym_vec)
    dx = -(Lx-Wlead)/2
    if zigzag:
        lead_tl.fill(bulk_lead, shapes.y_ribbon_zigzag(lat=WTe2, Lx=Wlead/param.a, x0=dx/param.a), (dx,0), max_sites=float('inf'))
    else:
        lead_tl.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead/param.a, x0=dx/param.a), (dx,0), max_sites=float('inf'))
    syst.attach_lead(lead_tl)
                     
    #attach top right lead
    sym_lead = bulk_sym.subgroup((0,1))
    lead_tr = kwant.Builder(sym_lead, conservation_law=sym_vec)
    dx = (Lx-Wlead)/2
    if zigzag:
        lead_tr.fill(bulk_lead, shapes.y_ribbon_zigzag(lat=WTe2, Lx=Wlead/param.a, x0=dx/param.a), (dx,0), max_sites=float('inf'))
    else:
        lead_tr.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead/param.a, x0=dx/param.a), (dx,0), max_sites=float('inf'))
    syst.attach_lead(lead_tr)
    
    #attach bottom left lead
    sym_lead = bulk_sym.subgroup((0,-1))
    lead_bl = kwant.Builder(sym_lead, conservation_law=sym_vec)
    dx = -(Lx-Wlead)/2
    if zigzag:
        lead_bl.fill(bulk_lead, shapes.y_ribbon_zigzag(lat=WTe2, Lx=Wlead/param.a, x0=dx/param.a), (dx,0), max_sites=float('inf'))
    else:
        lead_bl.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead/param.a, x0=dx/param.a), (dx,0), max_sites=float('inf'))
    syst.attach_lead(lead_bl)
                     
    #attach bottom right lead
    sym_lead = bulk_sym.subgroup((0,-1))
    lead_br = kwant.Builder(sym_lead, conservation_law=sym_vec)
    dx = (Lx-Wlead)/2
    if zigzag:
        lead_br.fill(bulk_lead, shapes.y_ribbon_zigzag(lat=WTe2, Lx=Wlead/param.a, x0=dx/param.a), (dx,0), max_sites=float('inf'))
    else:
        lead_br.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead/param.a, x0=dx/param.a), (dx,0), max_sites=float('inf'))
    syst.attach_lead(lead_br)
    
    return syst, WTe2

def build_Tjunction(Lx_horz=70, Ly_horz=20, Lx_vert=36, Ly_vert=38, Wlead_horz=20, Wlead_vert=36,
                   mu=0, mu_lead=-.4, lamb=1, lamb_lead=0, zigzag_vert=False, sym_vec=s0):
    
    # odd Ly_horz/Wlead_horz for W x-termination
    # even Ly_horz/Wlead_horz for Te x-termination
    # zigzag False for straight y-termination
    # zigzag Treu for zigzag y-termination
    
    bulk, WTe2, bulk_sym = func.WTe2_template(lamb=lamb, mu=mu) #sample bulk
    bulk_lead, WTe2, bulk_sym = func.WTe2_template(lamb=lamb_lead, mu=mu_lead) #lead bulk
    syst = kwant.Builder()
    
    #fill scattering region
    x_ribbon_shape = shapes.x_ribbon_finite_irregular(Lx=2*Lx_horz, Ly=Ly_horz)
    x_half_shape = lambda site: x_ribbon_shape(site) if site.pos[0] <= 0 else False
    
    if zigzag_vert: 
        y_ribbon_shape = shapes.rectangle_zigzag_y(lat=WTe2, Lx=Lx_vert, Ly=Ly_vert)
    else:
        y_ribbon_shape = shapes.rectangle_straight_y_(lat=WTe2, Lx=Lx_vert, Ly=Ly_vert)

    Tjunction_shape = lambda site: x_half_shape(site) or y_ribbon_shape(site)
    syst.fill(bulk, Tjunction_shape, (0,0), max_sites=float('inf'))
    
    #attach horizontal lead
    sym_lead = bulk_sym.subgroup((-2,0))
    lead2 = kwant.Builder(sym_lead, conservation_law=sym_vec)
    lead2.fill(bulk_lead, shapes.x_ribbon(Ly=Wlead_horz), (0,0), max_sites=float('inf'))
    syst.attach_lead(lead2)
    
    #attach vertical leads
    sym_lead = bulk_sym.subgroup((0,1))
    lead1 = kwant.Builder(sym_lead, conservation_law=sym_vec)
    if zigzag_vert:
        lead1.fill(bulk_lead, shapes.y_ribbon_zigzag(lat=WTe2, Lx=Wlead_vert), (0,0), max_sites=float('inf'))
    else:
        lead1.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead_vert), (0,0), max_sites=float('inf'))
    syst.attach_lead(lead1)

    sym_lead = bulk_sym.subgroup((0,-1))
    lead2 = kwant.Builder(sym_lead, conservation_law=sym_vec)
    if zigzag_vert:
        lead2.fill(bulk_lead, shapes.y_ribbon_zigzag(lat=WTe2, Lx=Wlead_vert), (0,0), max_sites=float('inf'))
    else:
        lead2.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead_vert), (0,0), max_sites=float('inf'))
    syst.attach_lead(lead2)
    
    return syst, WTe2


def build_fourTerm_syst(Lx_horz=70, Ly_horz=20, Lx_vert=36, Ly_vert=38, Wlead_horz=20, Wlead_vert=36,
                        mu=0, mu_lead=-.4, lamb=1, lamb_lead=0, zigzag_vert=False, walk_prob=0, sym_vec=s0):
    
    # odd Ly_horz/Wlead_horz for W x-termination
    # even Ly_horz/Wlead_horz for Te x-termination
    # zigzag False for straight y-termination
    # zigzag Treu for zigzag y-termination
    
    
    bulk, WTe2, bulk_sym = func.WTe2_template(lamb=lamb, mu=mu) #sample bulk
    bulk_lead, WTe2, bulk_sym = func.WTe2_template(lamb=lamb_lead, mu=mu_lead) #lead bulk
    syst = kwant.Builder()
    
    if walk_prob <= 0 or walk_prob > 1: #clean cut
        
        #fill scattering region
        x_ribbon_shape = shapes.x_ribbon_finite_irregular(Lx=Lx_horz, Ly=Ly_horz)
        if zigzag_vert:
            y_ribbon_shape = shapes.rectangle_zigzag_y(lat=WTe2, Lx=Lx_vert, Ly=Ly_vert)
        else:
            y_ribbon_shape = shapes.rectangle_straight_y_(lat=WTe2, Lx=Lx_vert, Ly=Ly_vert)

        fourTerm_shape = lambda site: x_ribbon_shape(site) or y_ribbon_shape(site)
        syst.fill(bulk, fourTerm_shape, (0,0), max_sites=float('inf'))
        
    else: #disordered edges
        
        p = walk_prob
        
        delta_x = param.a/2
        delta_y = param.b/2
        
        horz_walk_length = Lx_horz - Lx_vert
        vert_walk_length = Ly_vert - Ly_horz
        
        #create edges by generating 8 random walks
        left_top_walk = func.periodic_random_walk(p=p, N=horz_walk_length, t0=-Lx_horz*param.a/2, dt=delta_x, dx=delta_y, fix_start=True)
        left_bot_walk = func.periodic_random_walk(p=p, N=horz_walk_length, t0=-Lx_horz*param.a/2, dt=delta_x, dx=delta_y, fix_start=True)
        right_top_walk = func.periodic_random_walk(p=p, N=horz_walk_length, t0=Lx_vert*param.a/2, dt=delta_x, dx=delta_y, fix_start=True)
        right_bot_walk = func.periodic_random_walk(p=p, N=horz_walk_length, t0=Lx_vert*param.a/2, dt=delta_x, dx=delta_y, fix_start=True)

        top_left_walk = func.periodic_random_walk(p=p, N=vert_walk_length, t0=Ly_horz*param.b/2, dt=delta_y, dx=delta_x, fix_start=True)
        top_right_walk = func.periodic_random_walk(p=p, N=vert_walk_length, t0=Ly_horz*param.b/2, dt=delta_y, dx=delta_x, fix_start=True)
        bot_left_walk = func.periodic_random_walk(p=p, N=vert_walk_length, t0=-Ly_vert*param.b/2, dt=delta_y, dx=delta_x, fix_start=True)
        bot_right_walk = func.periodic_random_walk(p=p, N=vert_walk_length, t0=-Ly_vert*param.b/2, dt=delta_y, dx=delta_x, fix_start=True)

        left_top_edge = np.array(left_top_walk) + np.array([0, Ly_horz*param.b/2])
        left_bot_edge = np.array(left_bot_walk) + np.array([0, -Ly_horz*param.b/2])
        right_top_edge = np.array(right_top_walk) + np.array([0, Ly_horz*param.b/2])
        right_bot_edge = np.array(right_bot_walk) + np.array([0, -Ly_horz*param.b/2])

        top_left_edge = np.array(top_left_walk) + np.array([0, -Lx_vert*param.a/2])
        top_right_edge = np.array(top_right_walk) + np.array([0, Lx_vert*param.a/2])
        bot_left_edge = np.array(bot_left_walk) + np.array([0, -Lx_vert*param.a/2])
        bot_right_edge = np.array(bot_right_walk) + np.array([0, Lx_vert*param.a/2])
        
        #use edges to define jagged cross shape
        
        def jagged_cross(site):
            pos = site.pos
            x, y = pos
            
            #automatically return true if in central box
            if -Lx_vert*param.a/2 <= x <= Lx_vert*param.a/2 and -Ly_horz*param.b/2 <= y <= Ly_horz*param.b/2:
                return True
                
            #automatically return false if outside exterior box
            if -Lx_horz*param.a/2 > x or Lx_horz*param.a/2 < x or -Ly_vert*param.b/2 > y or Ly_vert*param.b/2 < y:
                return False
                
            else:
                if x < -Lx_vert*param.a/2: #left terminal
                    
                    x_vals = left_top_edge[:, 0]
                    y_vals_max = left_top_edge[:, 1]
                    y_vals_min = left_bot_edge[:, 1]
                    
                    maxY = np.interp(x, x_vals, y_vals_max)
                    minY = np.interp(x, x_vals, y_vals_min)
                    
                    if minY - .01 <= y <= maxY + .01:
                        return True
                    
                if x > Lx_vert*param.a/2: #right terminal
                    
                    x_vals = right_top_edge[:, 0]
                    y_vals_max = right_top_edge[:, 1]
                    y_vals_min = right_bot_edge[:, 1]
                    
                    maxY = np.interp(x, x_vals, y_vals_max)
                    minY = np.interp(x, x_vals, y_vals_min)
                    
                    if minY - .01 <= y <= maxY + .01:
                        return True
                
                if y > Ly_horz*param.b/2: #top terminal
                
                    y_vals = top_left_edge[:, 0]
                    x_vals_max = top_right_edge[:, 1]
                    x_vals_min = top_left_edge[:, 1]
                    
                    maxX = np.interp(y, y_vals, x_vals_max)
                    minX = np.interp(y, y_vals, x_vals_min)
                    
                    if minX - .01 <= x <= maxX + .01:
                        return True
                    
                if y < -Ly_horz*param.b/2: #bottom terminal
                    
                    y_vals = bot_left_edge[:, 0]
                    x_vals_max = bot_right_edge[:, 1]
                    x_vals_min = bot_left_edge[:, 1]
                    
                    maxX = np.interp(y, y_vals, x_vals_max)
                    minX = np.interp(y, y_vals, x_vals_min)
                    
                    if minX - .01 <= x <= maxX + .01:
                        return True
                
                return False
            
        syst.fill(bulk, jagged_cross, (0,0), max_sites=float('inf'))
        
    #attach horizontal leads
    sym_lead = bulk_sym.subgroup((2,0))
    lead1 = kwant.Builder(sym_lead, conservation_law=sym_vec)
    lead1.fill(bulk_lead, shapes.x_ribbon(Ly=Wlead_horz), (0,0), max_sites=float('inf'))
    syst.attach_lead(lead1)

    sym_lead = bulk_sym.subgroup((-2,0))
    lead2 = kwant.Builder(sym_lead, conservation_law=sym_vec)
    lead2.fill(bulk_lead, shapes.x_ribbon(Ly=Wlead_horz), (0,0), max_sites=float('inf'))
    syst.attach_lead(lead2)
    
    #attach vertical leads
    sym_lead = bulk_sym.subgroup((0,1))
    lead1 = kwant.Builder(sym_lead, conservation_law=sym_vec)
    if zigzag_vert:
        lead1.fill(bulk_lead, shapes.y_ribbon_zigzag(lat=WTe2, Lx=Wlead_vert), (0,0), max_sites=float('inf'))
    else:
        lead1.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead_vert), (0,0), max_sites=float('inf'))
    syst.attach_lead(lead1)

    sym_lead = bulk_sym.subgroup((0,-1))
    lead2 = kwant.Builder(sym_lead, conservation_law=sym_vec)
    if zigzag_vert:
        lead2.fill(bulk_lead, shapes.y_ribbon_zigzag(lat=WTe2, Lx=Wlead_vert), (0,0), max_sites=float('inf'))
    else:
        lead2.fill(bulk_lead, shapes.y_ribbon(lat=WTe2, Lx=Wlead_vert), (0,0), max_sites=float('inf'))
    syst.attach_lead(lead2)
    
    return syst, WTe2
