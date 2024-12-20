import matplotlib.pyplot as plt
import numpy as np
import os

# Print current working directory
print(os.getcwd())

# Load data
energy = np.loadtxt("out/cpu/1d/energy_with_field.dat")[:, 1]
magnetization = np.loadtxt("out/cpu/1d/magnetization_with_field.dat")[:, 1]

# Plot energy over time
plt.figure()
plt.plot(energy, label="Energy")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Energy")
plt.title("Energy vs Monte Carlo Steps (with Magnetic Field)")
plt.legend()
plt.grid()
plt.savefig("out/cpu/1d/energy_vs_steps.png")

# Plot magnetization over time
plt.figure()
plt.plot(magnetization, label="Magnetization", color="orange")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Magnetization")
plt.title("Magnetization vs Monte Carlo Steps (with Magnetic Field)")
plt.legend()
plt.grid()
plt.savefig("out/cpu/1d/magnetization_vs_steps.png")
