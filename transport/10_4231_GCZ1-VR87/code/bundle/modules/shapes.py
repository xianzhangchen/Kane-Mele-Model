import numpy as np
import modules.parameters as param

        
def x_ribbon(Ly=10, y0=0):
    """ creates a ribbon along the x axis of height Ly centered at (0,y0)
    
        Parameters: Ly,y0: integer
                      height and center of the ribbon in units of the lattice constant b
        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """  
    a = param.a # lattice constant along x
    b = param.b # lattice constant along y

    def shape_function(site):
        x, y = site.pos
        return (y0*b - Ly*b/2. <= y < y0*b + Ly*b/2.)
    return shape_function

def y_ribbon(lat, Lx=10, x0=0):
    """ creates a ribbon along the y axis of width Lx with straight edges centered centered at (x0,0)
    
        Parameters: lat: lattice
                      4-atomic lattice of WTe2
                    Lx,x0: integer
                      width and center of the ribbon in units of the lattice constant a
        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """
    a = param.a # lattice constant along x
    b = param.b # lattice constant along y
    
    LWA = Lx-1
    LTeA = Lx-1
    LWB = Lx
    LTeB = Lx
    
    WA, TeA, WB, TeB = lat.sublattices
    
    def shape_function(site):
        x, y = site.pos
        sublat = site.family
        if sublat==WA:
            return (x0*a - LWA*a/2. <= x < x0*a + LWA*a/2.)
        elif sublat==TeA:
            return (x0*a - LTeA*a/2. <= x < x0*a + LTeA*a/2.)
        elif sublat==WB:
            return (x0*a - LWB*a/2. <= x < x0*a + LWB*a/2.)
        elif sublat==TeB:
            return (x0*a - LTeB*a/2. <= x < x0*a + LTeB*a/2.)
        
    return shape_function

def y_ribbon_zigzag(lat, Lx=10, x0=0):
    """ creates a ribbon along the y axis of width Lx with zigzag edges centered centered at (x0,0)
    
        Parameters: lat: lattice
                      4-atomic lattice of WTe2
                    Lx,x0: integer
                      width and center of the ribbon in units of the lattice constant a
        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """
    a = param.a # lattice constant along x
    b = param.b # lattice constant along y
    
    LWA = Lx-1
    LTeA = Lx-1
    LWB = Lx
    LTeB = Lx-2
    
    WA, TeA, WB, TeB = lat.sublattices
    
    def shape_function(site):
        x, y = site.pos
        sublat = site.family
        if sublat==WA:
            return (x0*a - LWA*a/2. <= x < x0*a + LWA*a/2.)
        elif sublat==TeA:
            return (x0*a - LTeA*a/2. <= x < x0*a + LTeA*a/2.)
        elif sublat==WB:
            return (x0*a - LWB*a/2. <= x < x0*a + LWB*a/2.)
        elif sublat==TeB:
            return (x0*a - LTeB*a/2. <= x < x0*a + LTeB*a/2.)
        
    return shape_function

def ribbon(Ly=10, v1=-2, v2=1, y0=0, x0=0):
    """ creates a general ribbon along the direction (v1*a,v2*b) of width Ly measured
        along the the y axis, and centered at (x0,y0)
    
        Parameters: Ly,y0,x0: integer
                      width and center of the ribbon in units of the lattice constant a for x0
                      and b for Ly and y0
                    v1,v2: integer
                      components of the vector (v1,v2) that defines the alignment direction of 
                      the ribbon. v1 is measured in units if the lattice constant a, v2 in units
                      of the lattice constant b
        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """
    a = param.a # lattice constant along x
    b = param.b # lattice constant along y
    
    def shape_function(site):
        x, y = site.pos
        return (b*v2/(a*v1)*(x-a*x0) - Ly*b/2.  <= (y-b*y0) < b*v2/(a*v1)*(x-a*x0) + Ly*b/2.)
    return shape_function

def rectangle(Lx=10, Ly=10, x0=0, y0=0):
    """ creates a rectangle of width Lx and height Ly with center at (x0,y0)
    
        Parameters: Lx,Ly: integer
                      width and height of the system in units of the lattice constants 
                      a and b, respectively.
                    x0, y0: float
                      center of the rectangle in real space units
        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """
    a = param.a # lattice constant along x
    b = param.b # lattice constant along y
    
    def shape_function(site):
        x, y = site.pos
        return (-Lx*a/2. <= x-x0 < Lx*a/2.) and (-Ly*b/2. <= y-y0 < Ly*b/2.)
    return shape_function

def rectangle_straight_y_(lat, Lx=10, Ly=10, x0=0, y0=0):
    """ creates a rectangle of width Lx and height Ly with center at (x0,y0) which has a straight
        edge along the y direction and whose edges are symmetric.
    
        Parameters: lat: lattice
                      4-atomic lattice of WTe2
                    Lx,Ly: integer
                      width and height of the system in units of the lattice constants 
                      a and b, respectively.
                    x0, y0: float
                      center of the rectangle in real space units
        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """
    a = param.a # lattice constant along x
    b = param.b # lattice constant along y
    
    LWA = Lx-1
    LTeA = Lx-1
    LWB = Lx
    LTeB = Lx
    
    WA, TeA, WB, TeB = lat.sublattices
    
    def shape_function(site):
        x, y = site.pos
        sublat = site.family
        
        if sublat==WA:
            return (x0*a - LWA*a/2. <= x < x0*a + LWA*a/2.) and (-Ly*b/2. <= y-y0 < Ly*b/2.)
        elif sublat==TeA:
            return (x0*a - LTeA*a/2. <= x < x0*a + LTeA*a/2.) and (-Ly*b/2. <= y-y0 < Ly*b/2.)
        elif sublat==WB:
            return (x0*a - LWB*a/2. <= x < x0*a + LWB*a/2.) and (-Ly*b/2. <= y-y0 < Ly*b/2.)
        elif sublat==TeB:
            return (x0*a - LTeB*a/2. <= x < x0*a + LTeB*a/2.) and (-Ly*b/2. <= y-y0 < Ly*b/2.)
        
    return shape_function

def rectangle_zigzag_y(lat, Lx=10, Ly=10, x0=0, y0=0):
    """ creates a rectangle of width Lx and height Ly with center at (x0,y0) which has a zigzag
        edge along the y direction.
    
        Parameters: lat: lattice
                      4-atomic lattice of WTe2
                    Lx,Ly: integer
                      width and height of the system in units of the lattice constants 
                      a and b, respectively.
                    x0, y0: float
                      center of the rectangle in real space units
        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """
    a = param.a # lattice constant along x
    b = param.b # lattice constant along y
    
    LWA = Lx-1
    LTeA = Lx-1
    LWB = Lx
    LTeB = Lx-2
    
    WA, TeA, WB, TeB = lat.sublattices
    
    def shape_function(site):
        x, y = site.pos
        sublat = site.family
        
        if sublat==WA:
            return (x0*a - LWA*a/2. <= x < x0*a + LWA*a/2.) and (-Ly*b/2. <= y-y0 < Ly*b/2.)
        elif sublat==TeA:
            return (x0*a - LTeA*a/2. <= x < x0*a + LTeA*a/2.) and (-Ly*b/2. <= y-y0 < Ly*b/2.)
        elif sublat==WB:
            return (x0*a - LWB*a/2. <= x < x0*a + LWB*a/2.) and (-Ly*b/2. <= y-y0 < Ly*b/2.)
        elif sublat==TeB:
            return (x0*a - LTeB*a/2. <= x < x0*a + LTeB*a/2.) and (-Ly*b/2. <= y-y0 < Ly*b/2.)
        
    return shape_function

def rhombus(Lx=10, Ly=10, v1=-2, v2=1):
    """ creates a rhombus standing on an axis parallel to the y axis. Its height is Lx 
        and its width is Ly measured along the y axis. Its skew sides point along the 
        direction (v1*a,v2*b)
    
        Parameters: Lx,Ly: integer
                      height and width of the system in units of the lattice constants 
                      a and b, respectively.
                    v1,v2: integer
                      components of the vector (v1,v2) that defines the direction of the
                      skew sides of the rhombus. v1 is measured in units if the lattice
                      constant a, v2 in units of the lattice constant b
        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """     
    a = param.a # lattice constant along x
    b = param.b # lattice constant along y
    
    def shape_function(site):
        x, y = site.pos
        return ((b*v2/(a*v1)*x - Ly*b/2. <= y < b*v2/(a*v1)*x + Ly*b/2.) 
                                  and (-Lx*a/2. <= x < Lx*a/2.))
    return shape_function

def disk(r=10.0):
    """ creates a disk of radius r

        Parameters: r:  float
                      radius of the disk
        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """
    a = param.a # lattice constant along x
    b = param.b # lattice constant along y
    
    def shape_function(site):
        x, y = site.pos
        rsq = x**2 + y**2
        return (rsq <= r**2)
    return shape_function

def x_ribbon_finite_irregular(Lx=100, Ly=10, edge1=lambda x: 0.0, edge2=lambda x: 0.0):
    """ creates a finite ribbon with irregular edges along the x axis

        Parameters: Lx,Ly: integer
                        width and height of the ribbon in units of the 
                        lattice constants a and b
                    edge1, edge2: function
                        functions that describe the deviations of the x edges
                        from a straight edge

        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """  
    a = param.a
    b = param.b

    def shape_function(site):
        x, y = site.pos
        x0 = -Lx*a/2
        return ((-Ly*b/2. + edge2(np.array([x-x0])) <= y <= Ly*b/2. + edge1(np.array([x-x0])))
                and (x0 <= x < x0 + Lx*a))

    return shape_function

def y_zigzag_ribbon_finite_irregular(lat, Lx=10, Ly=100, edge1=lambda y: 0.0, edge2=lambda y: 0.0):
    """ creates a finite ribbon with irregular edges along the y axis based on a zigzag termination

        Parameters: lat: lattice
                        4-atomic lattice of WTe2
                    Lx,Ly: integer
                        width and height of the ribbon in units of the 
                        lattice constants a and b
                    edge1, edge2: function
                        functions that describe the deviations of the y edges
                        from a zigzag edge

        Returns:    shape_function
                      A boolean function of site returning whether the site should be added
    """  
    a = param.a
    b = param.b
    
    LWA = Lx-1
    LTeA = Lx-1
    LWB = Lx
    LTeB = Lx-2
    
    WA, TeA, WB, TeB = lat.sublattices

    def shape_function(site):
        x, y = site.pos
        sublat = site.family
        y0 = -Ly*b/2
        if sublat==WA:
            return ((-LWA*a/2. + edge2(np.array([y-y0])) <= x <= LWA*a/2. + edge1(np.array([y-y0])))
                and (y0 <= y < y0 + Ly*b))
        elif sublat==TeA:
            return ((-LTeA*a/2. + edge2(np.array([y-y0])) <= x <= LTeA*a/2. + edge1(np.array([y-y0])))
                and (y0 <= y < y0 + Ly*b))
        elif sublat==WB:
            return ((-LWB*a/2. + edge2(np.array([y-y0])) <= x <= LWB*a/2. + edge1(np.array([y-y0])))
                and (y0 <= y < y0 + Ly*b))
        elif sublat==TeB:
            return ((-LTeB*a/2. + edge2(np.array([y-y0])) <= x <= LTeB*a/2. + edge1(np.array([y-y0])))
                and (y0 <= y < y0 + Ly*b))
        
    return shape_function
