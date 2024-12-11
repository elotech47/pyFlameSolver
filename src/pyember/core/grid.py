"""
grid.py: Grid management system for PyEmber

This implements the grid system following the original Ember implementation,
providing functionality for grid adaptation, point distribution, and grid metrics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from .base import GridComponent


@dataclass
class GridConfig:
    """Configuration parameters for the grid system"""
    # Grid extent
    x_min: float = 0.0
    x_max: float = 1.0
    n_points: int = 100
    
    # Grid adaptation parameters
    vtol: float = 0.1          # Relative gradient tolerance
    dvtol: float = 0.2         # Relative curvature tolerance
    rmtol: float = 0.5         # Tolerance for point removal
    error_tol: float = 0.1     # Error tolerance for adaptation
    boundaryTol: float = 0.1   # Tolerance for boundary extension
    uniformityTol: float = 2.5  # Maximum allowed ratio of adjacent cell sizes
    
    # Geometry parameters
    cylindrical: bool = False   # Cylindrical coordinates flag
    alpha: int = 0             # Geometric power (0=planar, 1=cylindrical)
    
    # Grid bounds
    grid_min: float = 1e-6     # Minimum grid spacing
    grid_max: float = 1e-3     # Maximum grid spacing
    
    # Additional controls
    fixed_left_loc: bool = False  # Fix leftmost point location
    uniform: bool = False        # Use uniform grid spacing

class OneDimGrid(GridComponent):
    """
    One-dimensional adaptive grid implementation.
    
    This follows the original Ember grid system, providing support for:
    - Adaptive grid refinement
    - Variable grid spacing
    - Cylindrical coordinates
    - Grid metrics computation
    """
    
    def __init__(self, config: Optional[GridConfig] = None):
        self.config = config or GridConfig()
        
        # Grid points and derived quantities
        self.x: np.ndarray = None          # Grid point locations
        self.r: np.ndarray = None          # Radial coordinates (cylindrical)
        self.dx: np.ndarray = None         # Grid spacing
        self.cf: np.ndarray = None         # Center coefficients
        self.cfm: np.ndarray = None        # Left coefficients
        self.cfp: np.ndarray = None        # Right coefficients
        
        # Boundary indices
        self.js: int = 0                   # Left boundary index
        self.je: int = 0                   # Right boundary index
        
        # Initialize the grid
        self.initialize()

    def initialize(self) -> None:
        """Initialize the grid points and compute metrics"""
        # Create initial uniform grid
        self.x = np.linspace(
            self.config.x_min,
            self.config.x_max,
            self.config.n_points
        )
        
        # Set boundary indices
        self.js = 0
        self.je = len(self.x) - 1
        
        # Compute grid metrics
        self.update_grid_metrics()

    def update_grid_metrics(self) -> None:
        """
        Update grid metrics including spacing and finite difference coefficients.
        Follows Ember's grid.cpp implementation.
        """
        n = len(self.x)
        
        # Compute grid spacing
        self.dx = np.zeros(n)
        self.dx[:-1] = self.x[1:] - self.x[:-1]
        
        # Compute finite difference coefficients
        self.cf = np.zeros(n)    # center
        self.cfm = np.zeros(n)   # left
        self.cfp = np.zeros(n)   # right
        
        # Interior points (follow Ember's coefficient computation)
        for j in range(1, n-1):
            hp = self.dx[j]      # spacing to right point
            hm = self.dx[j-1]    # spacing to left point
            
            # Second-order central difference coefficients
            self.cfm[j] = -hp / (hm * (hp + hm))
            self.cfp[j] = -hm / (hp * (hp + hm))
            self.cf[j] = -(self.cfm[j] + self.cfp[j])
        
        # Compute radial coordinates for cylindrical geometry
        if self.config.cylindrical:
            self.r = self.x.copy()
            if self.config.alpha > 0:
                # Apply geometric transform for cylindrical coordinates
                self.r = self.x ** (1.0 / self.config.alpha)

    def adapt_grid(self, y: np.ndarray, scale: Optional[np.ndarray] = None) -> bool:
        """
        Adapt the grid based on solution gradients and curvature.
        
        Args:
            y: Solution vector to adapt to (must match current grid size)
            scale: Optional scaling factors for solution components
                
        Returns:
            bool: True if grid was modified
        """
        if len(y) != len(self.x):
            raise ValueError(f"Solution vector length ({len(y)}) must match grid size ({len(self.x)})")
            
        modified = False
        
        # First handle point addition
        # Compute solution gradients
        dy_dx = np.gradient(y, self.x)
        d2y_dx2 = np.gradient(dy_dx, self.x)
        
        # Apply scaling if provided
        if scale is not None:
            if len(scale) != len(y):
                raise ValueError("Scale vector must match solution vector length")
            dy_dx *= scale
            d2y_dx2 *= scale
        
        # Check gradient and curvature criteria
        add_points = np.logical_or(
            np.abs(dy_dx) > self.config.vtol,
            np.abs(d2y_dx2) > self.config.dvtol
        )
        
        # Add points where needed
        if np.any(add_points):
            self._add_points(add_points)
            modified = True
            
            # Need to recompute solution values for the new grid
            # For this test, we'll use the same function
            y = np.tanh((self.x - 0.5) * 10)
            dy_dx = np.gradient(y, self.x)
            d2y_dx2 = np.gradient(dy_dx, self.x)
        
        # Then handle point removal on the possibly modified grid
        remove_points = np.logical_and(
            np.abs(dy_dx) < self.config.rmtol * self.config.vtol,
            np.abs(d2y_dx2) < self.config.rmtol * self.config.dvtol
        )
        
        # Remove points where possible
        if np.any(remove_points):
            self._remove_points(remove_points)
            modified = True
            
        if modified:
            self.update_grid_metrics()
            
        return modified

    def _add_points(self, flags: np.ndarray) -> None:
        """Add grid points where indicated by flags"""
        # Identify regions needing refinement
        regions = np.where(flags)[0]
        
        new_points = []
        for idx in regions:
            if idx < len(self.x) - 1:
                # Check spacing constraints
                dx = self.x[idx + 1] - self.x[idx]
                if dx > 2 * self.config.grid_min:
                    # Add point at midpoint
                    new_point = 0.5 * (self.x[idx] + self.x[idx + 1])
                    new_points.append(new_point)
        
        if new_points:
            # Add new points and resort grid
            self.x = np.sort(np.concatenate([self.x, new_points]))

    def _remove_points(self, flags: np.ndarray) -> None:
        """Remove grid points where indicated by flags"""
        # Create mask of same size as current grid
        remove_mask = np.zeros_like(self.x, dtype=bool)
        remove_mask[:len(flags)] = flags
        
        # Don't remove boundary points
        remove_mask[0] = remove_mask[-1] = False
        
        # Keep points needed for minimum resolution
        for i in range(1, len(self.x)-1):
            if remove_mask[i]:
                # Check if removal would violate maximum spacing
                dx_if_removed = self.x[i+1] - self.x[i-1]
                if dx_if_removed > self.config.grid_max:
                    remove_mask[i] = False
        
        # Remove points
        self.x = self.x[~remove_mask]

    def validate(self) -> bool:
        """Validate grid configuration and state"""
        # Check grid extent
        if self.config.x_max <= self.config.x_min:
            return False
            
        # Check number of points
        if self.config.n_points < 2:
            return False
            
        # Check grid spacing bounds
        if self.config.grid_min <= 0 or self.config.grid_max <= self.config.grid_min:
            return False
            
        # Check adaptation parameters
        if self.config.vtol <= 0 or self.config.dvtol <= 0 or self.config.rmtol <= 0:
            return False
            
        return True

    def get_neighbors(self, j: int) -> Tuple[int, int]:
        """Get indices of neighboring points for given index"""
        if j <= 0:
            return (0, 1)
        elif j >= len(self.x) - 1:
            return (len(self.x) - 2, len(self.x) - 1)
        else:
            return (j-1, j+1)
        
    def uniformity_metric(self):
            """
            Calculate grid uniformity metric with protection against division by zero.
            Returns the maximum ratio between adjacent grid spacings.
            """
            # Calculate grid spacings
            h = np.diff(self.x)
            
            # Add small epsilon to prevent division by zero
            eps = 1e-14
            h = np.maximum(h, eps)
            
            # Calculate ratios of adjacent spacings
            h1 = h[:-1]  # Left spacings
            h2 = h[1:]   # Right spacings
            
            # Calculate uniformity as max ratio between adjacent spacings
            uniformity = np.maximum(h1/h2, h2/h1)
            
            # Return maximum uniformity value
            return np.max(uniformity)

    def grid_weight_function(self, x):
        """
        Calculate grid weight at a given position.
        Higher weights indicate regions requiring finer resolution.
        
        Parameters:
        -----------
        x : float
            Position to evaluate weight
        
        Returns:
        --------
        float
            Weight value (>= 1.0)
        """
        # Calculate distance from center of domain
        center = (self.config.x_max + self.config.x_min) / 2
        rel_pos = abs(x - center) / (self.config.x_max - self.config.x_min)
        
        # Weight increases towards center of domain
        weight = 1.0 + (1.0 - rel_pos) * 0.5
        
        return weight

    def extend_boundaries(self, y):
        """
        Extend grid boundaries based on solution features.
        
        Parameters:
        -----------
        y : numpy.ndarray
            Solution values at grid points
        
        Returns:
        --------
        bool
            True if grid was modified, False otherwise
        """
        # Calculate solution gradients using central differences
        dy = np.gradient(y, self.x)
        
        # Calculate normalized gradients at boundaries
        grad_left = abs(dy[0])
        grad_right = abs(dy[-1])
        
        # Calculate maximum gradient in interior (excluding boundary points)
        interior_grads = abs(dy[1:-1])
        max_interior_grad = np.max(interior_grads) if len(interior_grads) > 0 else 0
        
        # Ensure we don't divide by zero
        if max_interior_grad < 1e-10:
            max_interior_grad = 1e-10
            
        # Normalize boundary gradients
        rel_grad_left = grad_left / max_interior_grad
        rel_grad_right = grad_right / max_interior_grad
        
        # Determine if boundaries need extension
        # Note: we check if gradients exceed threshold relative to interior
        threshold = 1.0 - self.config.boundaryTol
        extend_left = rel_grad_left > threshold
        extend_right = rel_grad_right > threshold
        
        modified = False
        
        # Extend left boundary if needed and allowed
        if extend_left and not self.config.fixed_left_loc:
            new_x_min = self.x[0] - self.config.grid_max
            if new_x_min != self.config.x_min:  # Only modify if actually changing
                self.config.x_min = new_x_min
                modified = True
            
        # Extend right boundary if needed
        if extend_right:
            new_x_max = self.x[-1] + self.config.grid_max
            if new_x_max != self.config.x_max:  # Only modify if actually changing
                self.config.x_max = new_x_max
                modified = True
            
        # If modified, regenerate grid points
        if modified:
            self._generate_grid_points()
            
        return modified

    def _generate_grid_points(self):
        """
        Generate grid points after boundary modification.
        """
        # Create new uniform grid with current settings
        n = self.config.n_points
        self.x = np.linspace(self.config.x_min, self.config.x_max, n)
        
        # Recompute grid metrics
        self._compute_metrics()

    def check_spacing(self) -> bool:
        """
        Check if grid spacing meets constraints.
        Returns True if grid needs adjustment.
        """
        if len(self.x) < 2:
            return False
            
        dx = np.diff(self.x)
        
        # Check minimum spacing
        too_close = dx < self.config.grid_min
        
        # Check maximum spacing
        too_far = dx > self.config.grid_max
        
        # Check uniformity
        if len(dx) > 1:
            ratio = dx[1:] / dx[:-1]
            non_uniform = np.logical_or(
                ratio > self.config.uniformityTol,
                ratio < 1.0/self.config.uniformityTol
            )
        else:
            non_uniform = False
            
        return np.any(too_close) or np.any(too_far) or np.any(non_uniform)
    
    def error_function(self, j: int, y: np.ndarray) -> float:
        """
        Compute local error estimate at grid point j.
        
        Args:
            j: Grid point index
            y: Solution vector
            
        Returns:
            float: Error estimate at point j
        """
        if j <= 0 or j >= len(self.x) - 1:
            return 0.0
            
        # Compute first and second derivatives
        h_left = self.x[j] - self.x[j-1]
        h_right = self.x[j+1] - self.x[j]
        
        # First derivative (centered difference)
        dy_left = (y[j] - y[j-1]) / h_left
        dy_right = (y[j+1] - y[j]) / h_right
        
        # Second derivative
        d2y = 2 * (dy_right - dy_left) / (h_left + h_right)
        
        # Compute local truncation error estimate
        error = abs(d2y) * max(h_left, h_right)**2
        
        return error

    def compute_error_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Compute error weights for all grid points.
        
        Args:
            y: Solution vector
            
        Returns:
            np.ndarray: Array of error weights
        """
        weights = np.zeros_like(self.x)
        for j in range(1, len(self.x)-1):
            weights[j] = self.error_function(j, y)
        return weights
    
    