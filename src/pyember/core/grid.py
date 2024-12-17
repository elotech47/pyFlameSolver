import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
from scipy.interpolate import CubicSpline

class BoundaryCondition(Enum):
    """Boundary conditions matching C++ implementation"""
    FixedValue = "fixed_value"
    ZeroGradient = "zero_gradient"
    WallFlux = "wall_flux"
    ControlVolume = "control_volume"
    Floating = "floating"

@dataclass
class GridConfig:
    """Configuration for grid adaptation"""
    # Grid control
    vtol: float = 0.12  # Value tolerance (gradient)
    dvtol: float = 0.2  # Derivative tolerance (curvature)
    rmTol: float = 0.6  # Point removal tolerance
    absvtol: float = 1e-8  # Absolute value tolerance
    
    # Grid bounds
    gridMin: float = 5e-7  # Minimum spacing
    gridMax: float = 2e-4  # Maximum spacing
    dampConst: float = 7.0  # Damping constant
    uniformityTol: float = 2.5  # Grid uniformity tolerance
    
    # Boundary control
    boundaryTol: float = 5e-5  # Boundary extension tolerance
    boundaryTolRm: float = 1e-5  # Boundary point removal tolerance
    addPointCount: int = 3  # Points to add at boundaries
    centerGridMin: float = 1e-4  # Minimum center spacing
    unstrainedDownstreamWidth: float = 5.0  # Unstrained flame width factor
    
    # Geometry
    fixedBurnedVal: bool = True  # Fix burned boundary values
    unburnedLeft: bool = True  # Unburned mixture on left
    fixedLeftLoc: bool = False  # Fix leftmost point
    twinFlame: bool = False  # Twin flame configuration
    cylindricalFlame: bool = False  # Cylindrical coordinates
    discFlame: bool = False  # Disc flame configuration

class OneDimGrid:
    """
    One-dimensional adaptive grid implementation matching Ember C++ code.
    """
    def __init__(self):
        # State flags
        self.updated = True
        self.leftBC = BoundaryCondition.FixedValue
        self.rightBC = BoundaryCondition.FixedValue
        
        # Grid points
        self.x = None  # Grid point locations
        self.nPoints = 0
        self.jj = 0  # nPoints - 1
        
        # Grid metrics
        self.hh = None  # Grid spacing
        self.cfm = None  # Left coefficients
        self.cf = None  # Center coefficients
        self.cfp = None  # Right coefficients
        self.dlj = None  # Local grid spacing
        self.rphalf = None  # r^alpha at half points
        self.r = None  # r^alpha at grid points
        self.dampVal = None  # Damping values
        
        # Adaptation parameters
        self.vtol_in = 0.12
        self.dvtol_in = 0.2
        self.vtol = None
        self.dvtol = None
        self.nVars = 0
        self.nAdapt = 0
        
        # Physical indices
        self.ju = 0  # Unburned point index
        self.jb = 0  # Burned point index
        
        # Configuration
        self.alpha = 0  # Coordinate system (0=planar, 1=cylindrical)
        self.beta = 1.0  # Strain metric (1=planar/cylindrical, 2=disc)
        
        # Constants for adaptation
        self.kMomentum = 1  # Index for momentum equation
        self.kEnergy = 0  # Index for energy equation
        
        # Configuration flags
        self.fixedBurnedVal = True
        self.unburnedLeft = True
        self.fixedLeftLoc = False
        self.twinFlame = False
        self.cylindricalFlame = False
        self.discFlame = False
        
    def setOptions(self, options: GridConfig):
        """Set grid options from configuration"""
        # Adaptation parameters
        self.vtol_in = options.vtol
        self.dvtol_in = options.dvtol
        self.absvtol = options.absvtol
        self.rmTol = options.rmTol
        self.uniformityTol = options.uniformityTol
        self.gridMin = options.gridMin
        self.gridMax = options.gridMax
        self.dampConst = options.dampConst
        self.centerGridMin = options.centerGridMin
        
        # Configuration flags
        self.fixedBurnedVal = options.fixedBurnedVal
        self.unburnedLeft = options.unburnedLeft
        self.fixedLeftLoc = options.fixedLeftLoc
        self.twinFlame = options.twinFlame
        self.cylindricalFlame = options.cylindricalFlame
        self.discFlame = options.discFlame
        
        # Boundary parameters
        self.boundaryTol = options.boundaryTol
        self.boundaryTolRm = options.boundaryTolRm
        self.addPointCount = options.addPointCount
        self.unstrainedDownstreamWidth = options.unstrainedDownstreamWidth
        
        # Coordinate system
        self.alpha = 1 if options.cylindricalFlame else 0
        self.beta = 2 if options.discFlame else 1

    def setSize(self, new_nPoints: int):
        """Set grid size"""
        self.nPoints = new_nPoints
        self.jj = new_nPoints - 1

    def updateValues(self):
        """Update grid metrics"""
        # Resize arrays
        self.hh = np.zeros(self.jj)
        self.cfm = np.zeros(self.jj + 1)
        self.cf = np.zeros(self.jj + 1)
        self.cfp = np.zeros(self.jj + 1)
        self.dlj = np.zeros(self.jj + 1)
        self.rphalf = np.zeros(self.jj)
        self.r = np.zeros(self.jj + 1)
        
        # Calculate spacing and r values
        for j in range(self.jj):
            self.hh[j] = self.x[j+1] - self.x[j]
            self.rphalf[j] = np.power(0.5*(self.x[j] + self.x[j+1]), self.alpha)
            
        self.r = np.power(self.x, self.alpha)
        
        # Calculate finite difference coefficients
        for j in range(1, self.jj):
            self.cfp[j] = self.hh[j-1] / (self.hh[j] * (self.hh[j] + self.hh[j-1]))
            self.cf[j] = (self.hh[j] - self.hh[j-1]) / (self.hh[j] * self.hh[j-1])
            self.cfm[j] = -self.hh[j] / (self.hh[j-1] * (self.hh[j] + self.hh[j-1]))
            self.dlj[j] = 0.5 * (self.x[j+1] - self.x[j-1])

    def adapt(self, y: List[np.ndarray]):
        """Adapt grid to solution"""
        self.nVars = len(y)
        assert self.nAdapt <= self.nVars, "Adaptation variables exceed solution size"
        assert np.all(self.dampVal >= 0), "Damping values must be positive"
        
        self.setSize(len(y[0]))
        
        self.nVars = len(y)

        # Initialize tolerance arrays - this was missing!
        self.vtol = np.full(self.nAdapt, self.vtol_in)
        self.dvtol = np.full(self.nAdapt, self.dvtol_in)
        insertion_indices = []
        removal_indices = []
        
        # Point insertion
        j = 0
        while j < self.jj:
            self.updateValues()
            dv = np.zeros(self.jj + 1)
            insert = False
            
            # Check each variable
            for k in range(self.nAdapt):
                v = y[k]
                
                # Calculate derivatives
                for i in range(1, self.jj):
                    dv[i] = self.cfp[i]*v[i+1] + self.cf[i]*v[i] + self.cfm[i]*v[i-1]
                
                v_range = np.max(v) - np.min(v) # Value range
                dv_range = np.max(dv[1:self.jj]) - np.min(dv[1:self.jj])
                
                if v_range < self.absvtol:
                    continue
                
                # Value resolution
                if abs(v[j+1] - v[j]) > self.vtol[k] * v_range:
                    insert = True
                    print(f"Adapt: v resolution wants grid point at {j}, k={k}, | v(j+1) - v(j) | = {abs(v[j+1] - v[j])}, v_range = {v_range}")
                    
                # Derivative resolution
                if (j != 0 and j != self.jj-1 and 
                    abs(dv[j+1] - dv[j]) > self.dvtol[k] * dv_range):
                    print(f"Adapt: dv resolution wants grid point at {j}, k={k}, | dv(j+1) - dv(j) | = {abs(dv[j+1] - dv[j])}, dv_range = {dv_range}")
                    insert = True
                    
            # Damping criterion
            if self.hh[j] > self.dampConst * self.dampVal[j]:
                print(f"Adapt: damping wants grid point at {j}, hh = {self.hh[j]}, dampVal = {self.dampVal[j]}")
                insert = True
                
            # Maximum grid size
            if self.hh[j] > self.gridMax:
                print(f"Adapt: max grid size wants grid point at {j}, hh = {self.hh[j]}")
                insert = True
                
            # Grid uniformity
            if j != 0 and self.hh[j]/self.hh[j-1] > self.uniformityTol:
                print(f"Adapt: uniformity wants grid point at {j}, hh = {self.hh[j]}, hh[j-1] = {self.hh[j-1]}")
                insert = True
                
            if j != self.jj-1 and self.hh[j]/self.hh[j+1] > self.uniformityTol:
                print(f"Adapt: uniformity wants grid point at {j}, hh = {self.hh[j]}, hh[j+1] = {self.hh[j+1]}")
                insert = True
                
            # Special handling for center region
            if (j == 0 and (self.leftBC == BoundaryCondition.ControlVolume or 
                           self.leftBC == BoundaryCondition.WallFlux)):
                x_left_min = min(self.centerGridMin, 0.02*self.x[self.jj])
                if self.hh[j] < 2*x_left_min:
                    insert = False
                    print(f"Adapt: grid point addition canceled at {j} by minimum center grid size j = {j}, hh = {self.hh[j]}")
                    
            # Minimum grid size
            if insert and self.hh[j] < 2*self.gridMin:
                print(f"Adapt: grid point addition canceled at {j} by minimum grid size j = {j}, hh = {self.hh[j]}")
                insert = False
                
            if insert:
                insertion_indices.append(j)
                self.addPoint(j, y)
                self.updated = True
                self.setSize(self.nPoints + 1)
                j += 2
            else:
                j += 1
                
        # Point removal
        j = 1
        while j < self.jj:
            self.updateValues()
            remove = True
            
            for k in range(self.nAdapt):
                v = y[k]
                for i in range(1, self.jj):
                    dv[i] = self.cfp[i]*v[i+1] + self.cf[i]*v[i] + self.cfm[i]*v[i-1]
                    
                v_range = np.max(v) - np.min(v)
                dv_range = np.max(dv[1:self.jj]) - np.min(dv[1:self.jj])
                
                if v_range < self.absvtol:
                    print(f"Adapt: No removal at {j} due to v range")
                    continue
                    
                # Value resolution
                if abs(v[j+1] - v[j-1]) > self.rmTol * self.vtol[k] * v_range:
                    print(f"Adapt: No removal at {j} due to v resolution")
                    remove = False
                    
                # Derivative resolution
                if (j != 2 and j != self.jj-1 and 
                    abs(dv[j+1] - dv[j-1]) > self.rmTol * self.dvtol[k] * dv_range):
                    print(f"Adapt: No removal at {j} due to dv resolution")
                    remove = False
                    
            # Damping criterion
            if self.hh[j] + self.hh[j-1] >= self.rmTol * self.dampConst * self.dampVal[j]:
                print(f"Adapt: No removal at {j} due to damping")
                remove = False
                
            # Maximum grid size
            if self.hh[j] + self.hh[j-1] > self.gridMax:
                print(f"Adapt: No removal at {j} due to max grid size")
                remove = False
                
            # Grid uniformity
            if j >= 2 and self.hh[j] + self.hh[j-1] > self.uniformityTol * self.hh[j-2]:
                print(f"Adapt: No removal at {j} due to uniformity")
                remove = False
                
            if j <= self.jj-2 and self.hh[j] + self.hh[j-1] > self.uniformityTol * self.hh[j+1]:
                print(f"Adapt: No removal at {j} due to uniformity")
                remove = False
                
            # Special handling for center
            if (j == 1 and (self.leftBC == BoundaryCondition.ControlVolume or 
                           self.leftBC == BoundaryCondition.WallFlux)):
                remove = False
                
            if remove:
                removal_indices.append(j)
                self.removePoint(j, y)
                self.setSize(self.nPoints - 1)
                self.updated = True
            else:
                j += 1
                
        if self.updated:
            self.updateValues()
            self.updateBoundaryIndices()
        
        print(f"Adapt: Inserted points at {insertion_indices}")
        print(f"Adapt: Removed points at {removal_indices}")
        return self.updated

    def addPoint(self, j_insert: int, y: List[np.ndarray]):
        """Add a grid point"""
        assert len(self.x) == len(self.dampVal)
        N = len(self.x)
        
        # New point location
        x_insert = 0.5 * (self.x[j_insert+1] + self.x[j_insert])
        
        # Insert damping value
        val = self._spline_interpolate(self.x, self.dampVal, x_insert)
        self.dampVal = np.concatenate([
            self.dampVal[:j_insert + 1],
            [val],
            self.dampVal[j_insert + 1:]
        ])
        
        # Insert in solution vectors
        for i in range(len(y)):
            y_new = self._spline_interpolate(self.x, y[i], x_insert)
            y[i] = np.insert(y[i], j_insert + 1, y_new)
            
        # Update grid
        self.x = np.insert(self.x, j_insert + 1, x_insert)

    def removePoint(self, j_remove: int, y: List[np.ndarray]):
        """Remove grid point with proper numpy handling"""
        # Update grid points
        self.x = np.delete(self.x, j_remove)
        self.dampVal = np.delete(self.dampVal, j_remove)
        
        # Update solution arrays
        for k in range(len(y)):
            y[k] = np.delete(y[k], j_remove)
    
    def removeRight(self, y: List[np.ndarray]) -> bool:
        """
        Remove points from right boundary matching C++ implementation exactly
        """
        # Don't remove if too few points
        if self.jj < 3:
            return False
            
        point_removed = True  # Assume we can remove
        
        # Check each variable
        for k in range(self.nAdapt):
            y_max = np.max(abs(y[k]))
            
            # Skip minor variables
            if y_max < self.absvtol:
                #print(f"Warning: Variable {k} has small range {y_max}")
                continue
                
            # Check gradient exactly like C++
            local_diff = abs(y[k][self.jj] - y[k][self.jj-1])
            nonlocal_diff = abs(y[k][self.jj] - y[k][self.jj-2])
            
            # Scale by maximum value
            local_grad = local_diff / y_max
            nonlocal_grad = nonlocal_diff / y_max
            
            # if (nonlocal_grad > self.boundaryTolRm):
            #     #print(f"Warning: Failed removal criteria for variable {k} - local: {local_grad}, nonlocal: {nonlocal_grad}")
            #     point_removed = False
            #     break
                
            # Check additional C++ criteria
            if self.jj < 3 or local_grad > self.boundaryTolRm:
                point_removed = False
                #print(f"Warning: Failed removal criteria for variable {k} - local: {local_grad}")
                break
                
            # # Check spacing against damping
            # if (self.hh[self.jj-1] > self.gridMax or 
            # (self.jj > 1 and self.hh[self.jj-1]/self.hh[self.jj-2] > self.uniformityTol)):
            #     print(f"Warning: Failed removal criteria for variable {k} - spacing")
            #     point_removed = False
            #     break
        
        if point_removed:
            self.removePoint(self.jj, y)
            self.setSize(self.nPoints - 1)
            self.updateBoundaryIndices()
            
        return point_removed


    def regrid(self, y: List[np.ndarray]):
        """Complete regridding implementation matching C++"""
        self.nVars = len(y)
        assert self.nAdapt <= self.nVars
        assert np.all(self.dampVal > 0)
        
        self.setSize(len(y[0]))
        
        # Try adding points first
        right_addition = self.addRight(y)
        left_addition = self.addLeft(y)
        
        right_removal = False
        left_removal = False
        right_removal_count = 0
        left_removal_count = 0
        
        # Then try removing points if no additions were made
        if not right_addition:
            while True:
                removed = self.removeRight(y)
                if not removed:
                    break
                right_removal = True
                right_removal_count += 1
                
        if not left_addition:
            while True:
                removed = self.removeLeft(y)
                if not removed:
                    break
                left_removal = True
                left_removal_count += 1
                
        self.updated = self.updated or left_addition or right_addition or left_removal or right_removal
        
        if self.updated:
            self.updateValues()
            self.updateBoundaryIndices()

    # 2. Fix addRight to maintain boundary:
    def addRight(self, y: List[np.ndarray]) -> bool:
        """Add points to right boundary while maintaining bounds"""
        # Comparison points depend on boundary condition
        dj_mom = 1
        dj_other = 2 if (self.jb == self.jj and not self.fixedBurnedVal) else 1
        
        point_added = False
        
        # Check solution flatness at boundary
        for k in range(self.nAdapt):
            dj = dj_mom if k == self.kMomentum else dj_other
            y_max = np.max(abs(y[k]))
            if y_max > self.absvtol:
                if abs(y[k][self.jj] - y[k][self.jj-dj])/y_max > self.boundaryTol:
                    point_added = True
                    break
                    
        if point_added:
            print("Regrid: Adding points to right boundary")    
            x_right = self.x[-1]  # Save right boundary
            # Add requested number of points
            for i in range(self.addPointCount):
                # New point spacing based on uniformity
                new_dx = np.power(self.uniformityTol, 1.0/(1+self.addPointCount)) * (self.x[self.jj] - self.x[self.jj-1])
                new_x = min(self.x[self.jj] + new_dx, x_right)  # Don't exceed original boundary
                
                # Extend arrays
                self.x = np.append(self.x, new_x)
                self.dampVal = np.append(self.dampVal, self.dampVal[self.jj])
                
                # Extend solution vectors
                for k in range(len(y)):
                    y[k] = np.append(y[k], y[k][self.jj])
                    
                self.setSize(self.nPoints + 1)
                print(f"Regrid: Added point at {new_x}")
                
            self.updateBoundaryIndices()
        else:
            print("Regrid: No points added to right boundary")
        return point_added

    def addLeft(self, y: List[np.ndarray]) -> bool:
        """Add points to left boundary"""
        dj_other = 2 if (self.jb == 1 and not self.fixedBurnedVal) else 1
        dj_mom = 1
        
        point_added = False
        
        if not self.fixedLeftLoc:
            # Check solution flatness
            for k in range(self.nAdapt):
                y_max = np.max(y[k])
                dj = dj_mom if k == self.kMomentum else dj_other
                if (abs(y[k][dj] - y[k][0])/y_max > self.boundaryTol and 
                    y_max > self.absvtol):
                    point_added = True
                    break
                    
        # Force left boundary to x=0 if needed
        if (self.fixedLeftLoc and 
            self.leftBC not in [BoundaryCondition.ControlVolume, BoundaryCondition.WallFlux] and 
            self.x[0] > 0.0):
            point_added = True
            
        if point_added:
            for i in range(self.addPointCount):
                # New point location
                x_left = self.x[0] - np.sqrt(self.uniformityTol) * (self.x[1] - self.x[0])
                
                # Special handling for twin/cylindrical flames
                if self.twinFlame or self.cylindricalFlame:
                    if self.x[0] == 0:
                        break
                        
                    x_left_min = min(self.centerGridMin, 0.005 * self.x[self.jj])
                    if x_left < 0.0:
                        x_left = 0.0
                    elif x_left < x_left_min:
                        x_left = x_left_min
                        
                # Insert new point
                self.x = np.insert(self.x, 0, x_left)
                self.dampVal = np.insert(self.dampVal, 0, self.dampVal[0])
                
                for k in range(self.nVars):
                    y[k].insert(0, y[k][0])
                    
                self.setSize(self.nPoints + 1)
                
            self.updateBoundaryIndices()
            
        return point_added


    def removeLeft(self, y: List[np.ndarray]) -> bool:
        """Remove points from left boundary"""
        dj_mom = 2
        dj_other = 3 if (self.jb == 1 and not self.fixedBurnedVal) else 2
        
        point_removed = True
        
        # Don't remove if left location is fixed
        if self.fixedLeftLoc:
            point_removed = False
            
        # Check each variable
        for k in range(self.nAdapt):
            dj = dj_mom if k == self.kMomentum else dj_other
            y_max = np.max(y[k])
            if (abs(y[k][dj] - y[k][0])/y_max > self.boundaryTolRm and 
                y_max > self.absvtol):
                point_removed = False
                break
                
        # Don't remove too many points
        if self.jj < 3:
            point_removed = False
            
        if point_removed:
            self.removePoint(0, y)
            self.setSize(self.nPoints - 1)
            self.updateBoundaryIndices()
            
        return point_removed

    def updateBoundaryIndices(self):
        """Update indices for burned/unburned regions"""
        if self.unburnedLeft:
            self.ju = 0
            self.jb = self.jj
        else:
            self.jb = 0
            self.ju = self.jj
            
    def _spline_interpolate(self, x: np.ndarray, y: np.ndarray, x_new: float) -> float:
        """Helper for cubic spline interpolation"""
        cs = CubicSpline(x, y)
        return float(cs(x_new))
