import numpy as np
from pyember.solvers.diffusion_flame import DiffusionFlame, FlameConfig

# Create test configuration 
config = FlameConfig(
    mechanism='gri30.yaml',  # Or your preferred mechanism
    fuel='CH4:1.0',
    oxidizer='O2:0.21, N2:0.79',
    pressure=101325.0,  # 1 atm
    T_fuel=300.0,      # K
    T_oxidizer=300.0,  # K
    strain_rate=100.0, # 1/s
    
    # Grid parameters
    grid_points=101,
    x_min=-0.02,  # m
    x_max=0.02,   # m
    
    # Adaptation parameters 
    regrid_time_interval=1e-4,
    regrid_step_interval=10,
    
    # Initial profile parameters
    center_width=0.001,
    slope_width=0.0005,
    smooth_count=4
)

# Initialize solver
flame = DiffusionFlame(config)

# Time stepping
dt = 1e-5  # Initial time step
t_end = 0.01  # Total simulation time

# Arrays for recording solution evolution
times = [0.0]
T_history = [flame.T.copy()]
Y_history = [flame.Y.copy()]

# Main time stepping loop
while flame.t < t_end:
    # Take time step
    flame.step(dt)
    
    # Record solution every few steps
    if len(times) % 10 == 0:
        times.append(flame.t)
        T_history.append(flame.T.copy())
        Y_history.append(flame.Y.copy())
        
    print(f"t = {flame.t:.6f}, dt = {dt:.6e}, n_points = {len(flame.grid.x)}")

# Plot results
import matplotlib.pyplot as plt

# Temperature evolution
plt.figure(figsize=(10,6))
for i in range(0, len(times), 2):
    plt.plot(flame.grid.x, T_history[i], label=f't={times[i]:.4f}')
plt.xlabel('Position [m]')
plt.ylabel('Temperature [K]')
plt.title('Temperature profile Evolution')
plt.legend()
plt.grid(True)
plt.show()

# Species evolution
plt.figure(figsize=(10,6))
plt.plot(flame.grid.x, Y_history[-1][0], 'b-', label='final CH4')
plt.plot(flame.grid.x, Y_history[0][0], 'b--', label='initial CH4')
plt.plot(flame.grid.x, Y_history[-1][1], 'r-', label='final O2') 
plt.plot(flame.grid.x, Y_history[0][1], 'r--', label='initial O2')
plt.plot(flame.grid.x, Y_history[-1][2], 'g-', label='final N2')
plt.plot(flame.grid.x, Y_history[0][2], 'g--', label='initial N2')
plt.xlabel('Position [m]')
plt.ylabel('Mass Fraction')
plt.title('CH4, N2 and O2 profile Evolution')
plt.legend()
plt.grid(True)
plt.show()