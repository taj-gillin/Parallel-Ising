import matplotlib.pyplot as plt
import numpy as np

# Load lattice configuration
lattice = np.loadtxt("out/cpu/2d/lattice_2d.dat")

# Load energy and magnetization data
energy = np.loadtxt("out/cpu/2d/energy_2d.dat")[:, 1]
magnetization = np.loadtxt("out/cpu/2d/magnetization_2d.dat")[:, 1]

# Plot final lattice configuration
plt.figure(figsize=(6, 6))
plt.imshow(lattice, cmap="coolwarm", interpolation="nearest")
plt.title("Final Lattice Configuration")
plt.colorbar(label="Spin")
plt.savefig("out/cpu/2d/lattice_2d.png")

# Plot energy over time
plt.figure()
plt.plot(energy, label="Energy")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Energy")
plt.title("Energy vs Monte Carlo Steps (2D)")
plt.legend()
plt.grid()
plt.savefig("out/cpu/2d/energy_vs_steps.png")

# Plot magnetization over time
plt.figure()
plt.plot(magnetization, label="Magnetization", color="orange")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Magnetization")
plt.title("Magnetization vs Monte Carlo Steps (2D)")
plt.legend()
plt.grid()
plt.savefig("out/cpu/2d/magnetization_vs_steps.png")
