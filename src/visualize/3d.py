import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import sys

# Check if folder argument is provided
if len(sys.argv) != 2:
    print("Usage: python 3d.py <folder>")
    sys.exit(1)

folder = sys.argv[1]  # Get folder name from command line argument
base_path = f"out/{folder}/3d"  # Create base path

# Load energy and magnetization data
energy = np.loadtxt(f"{base_path}/energy_3d.dat")[:, 1]
magnetization = np.loadtxt(f"{base_path}/magnetization_3d.dat")[:, 1]

# # Read the lattice size from the first line of the file
# with open(f"{base_path}/lattice_3d.dat", 'r') as f:
#     first_line = f.readline()  # Skip "Slice 0:" line
#     first_data_line = f.readline()
#     L = len(first_data_line.strip().split())

# # Read the lattice data properly
# lattice = np.zeros((L, L, L))
# current_slice = 0
# current_row = 0
# with open(f"{base_path}/lattice_3d.dat", 'r') as f:
#     for line in f:
#         if line.startswith("Slice"):
#             current_slice = int(line.split()[1][:-1])  # Remove the colon
#             current_row = 0  # Reset row counter for new slice
#         elif line.strip():  # If line is not empty
#             lattice[current_slice][current_row] = [int(x) for x in line.strip().split()]
#             current_row += 1

# Plot energy over time
plt.figure()
plt.plot(energy, label="Energy")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Energy")
plt.title("Energy vs Monte Carlo Steps (3D)")
plt.legend()
plt.grid()
plt.savefig(f"{base_path}/energy_vs_steps.png")

# Plot magnetization over time
plt.figure()
plt.plot(magnetization, label="Magnetization", color="orange")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Magnetization")
plt.title("Magnetization vs Monte Carlo Steps (3D)")
plt.legend()
plt.grid()
plt.savefig(f"{base_path}/magnetization_vs_steps.png")




# # Extract coordinates and spins
# x, y, z = np.meshgrid(range(L), range(L), range(L), indexing="ij")
# spins = lattice.flatten()
# x, y, z = x.flatten(), y.flatten(), z.flatten()

# # Filter spins for color coding
# spin_up = spins == 1
# spin_down = spins == -1

# # Create the figure
# fig = go.Figure()

# # Add scatter points for spins +1
# fig.add_trace(go.Scatter3d(
#     x=x[spin_up], y=y[spin_up], z=z[spin_up],
#     mode='markers',
#     marker=dict(size=5, color='red'),
#     name='Spin +1'
# ))

# # Add scatter points for spins -1
# fig.add_trace(go.Scatter3d(
#     x=x[spin_down], y=y[spin_down], z=z[spin_down],
#     mode='markers',
#     marker=dict(size=5, color='blue'),
#     name='Spin -1'
# ))

# # Layout
# fig.update_layout(
#     title="Interactive 3D Lattice Visualization",
#     scene=dict(
#         xaxis_title='X',
#         yaxis_title='Y',
#         zaxis_title='Z',
#     )
# )

# # Save the figure as an HTML file
# fig.write_html(f"{base_path}/lattice_3d.html")

