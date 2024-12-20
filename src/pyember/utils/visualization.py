"""
Visualization tools for PyEmber flame solutions
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Tuple

class FlameVisualizer:
    """
    Visualization tools for flame solutions
    """
    def __init__(self, flame):
        self.flame = flame
        self.fig = None
        self._animation = None
        self.history = {
            't': [],
            'T': [],
            'Y': [],
            'V': [],
            'U': []
        }

    def save_state(self):
        """Save current state for animation"""
        self.history['t'].append(self.flame.t)
        self.history['T'].append(self.flame.T.copy())
        self.history['Y'].append(self.flame.Y.copy())
        self.history['V'].append(self.flame.V.copy())
        self.history['U'].append(self.flame.U.copy())

    def plot_current_state(self, species_names: Optional[List[str]] = None):
        """
        Plot current flame state
        
        Args:
            species_names: List of species to plot (if None, plots major species)
        """
        x = self.flame.grid.x * 1000  # Convert to mm
        
        # Create figure with 3 subplots
        if self.fig is None:
            self.fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            self.fig.suptitle('Flame Structure')
        else:
            ax1, ax2, ax3 = self.fig.axes
            for ax in (ax1, ax2, ax3):
                ax.clear()

        # Temperature profile
        ax1.plot(x, self.flame.T, 'r-', label='Temperature')
        ax1.set_ylabel('Temperature [K]')
        ax1.legend()
        ax1.grid(True)

        # Species profiles
        if species_names is None:
            # Plot major species (Y > 0.01 anywhere)
            mask = np.max(self.flame.Y, axis=1) > 0.01
            species_indices = np.where(mask)[0]
            species_names = [self.flame.gas.species_name(k) for k in species_indices]
        else:
            species_indices = [self.flame.gas.species_index(name) 
                             for name in species_names]

        for k, name in zip(species_indices, species_names):
            ax2.plot(x, self.flame.Y[k], label=name)
        ax2.set_ylabel('Mass Fraction')
        ax2.legend()
        ax2.grid(True)

        # Velocity profile
        ax3.plot(x, self.flame.U, 'b-', label='Mass Flux')
        ax3.set_xlabel('Position [mm]')
        ax3.set_ylabel('Mass Flux [kg/m²/s]')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

    def create_animation(self, species_names: Optional[List[str]] = None,
                        interval: int = 50) -> FuncAnimation:
        """
        Create animation of flame evolution
        
        Args:
            species_names: List of species to animate
            interval: Time between frames in milliseconds
            
        Returns:
            matplotlib.animation.FuncAnimation object
        """
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Flame Evolution')

        x = self.flame.grid.x * 1000  # mm
        
        # Set up species to plot
        if species_names is None:
            mask = np.max(self.flame.Y, axis=1) > 0.01
            species_indices = np.where(mask)[0]
            species_names = [self.flame.gas.species_name(k) for k in species_indices]
        else:
            species_indices = [self.flame.gas.species_index(name) 
                             for name in species_names]
            print(species_indices)

        # Temperature limits
        T_min = min(np.min(T) for T in self.history['T'])
        T_max = max(np.max(T) for T in self.history['T'])
        
        # Species limits
        Y_max = {k: max(np.max(Y[k]) for Y in self.history['Y'])
                 for k in species_indices}
        
        # Velocity limits
        V_min = min(np.min(V) for V in self.history['V'])
        V_max = max(np.max(V) for V in self.history['V'])

        def animate(frame):
            # Clear axes
            for ax in (ax1, ax2, ax3):
                ax.clear()
                ax.grid(True)

            # Temperature
            ax1.plot(x, self.history['T'][frame], 'r-')
            ax1.set_ylabel('Temperature [K]')
            ax1.set_ylim(T_min, T_max * 1.1)

            # Species
            for k, name in zip(species_indices, species_names):
                ax2.plot(x, self.history['Y'][frame][k], label=name)
            ax2.set_ylabel('Mass Fraction')
            ax2.legend()
            #ax2.set_ylim(0, max(Y_max.values()) * 1.1)

            # Velocity
            ax3.plot(x, self.history['V'][frame], 'b-')
            ax3.set_xlabel('Position [mm]')
            ax3.set_ylabel('Mass Flux [kg/m²/s]')
            ax3.set_ylim(V_min * 1.1, V_max * 1.1)

            # Time stamp
            fig.suptitle(f'Flame Evolution (t = {self.history["t"][frame]:.3f} s)')

        self._animation = FuncAnimation(
            fig, animate, frames=len(self.history['t']),
            interval=interval, blit=False
        )
        
        return self._animation

    def save_animation(self, filename: str, fps: int = 20):
        """
        Save animation to file
        
        Args:
            filename: Output filename (.mp4 or .gif)
            fps: Frames per second
        """
        if self._animation is None:
            self.create_animation()
            
        self._animation.save(filename, fps=fps)