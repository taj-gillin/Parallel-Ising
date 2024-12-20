#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

// Parameters for the simulation
const int L = 10000;             // Lattice size
const int num_steps = 100000;  // Number of Monte Carlo steps
const double J = 1.0;          // Interaction strength
const double T = 2.0;          // Temperature (in units of k_B)
const double h = 0.5;          // External magnetic field

// Random number generation
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

// Initialize lattice with random spins (+1 or -1)
void initialize_lattice(std::vector<int>& spins) {
    std::uniform_int_distribution<> spin_dis(0, 1);
    for (int i = 0; i < L; i++) {
        spins[i] = spin_dis(gen) == 0 ? 1 : -1;
    }
}

// Calculate the total energy of the system
double calculate_energy(const std::vector<int>& spins) {
    double energy = 0.0;
    for (int i = 0; i < L; i++) {
        int right = (i + 1) % L; // Periodic boundary condition
        energy -= J * spins[i] * spins[right];
        energy -= h * spins[i]; // External magnetic field contribution
    }
    return energy;
}

// Perform one Monte Carlo step
void monte_carlo_step(std::vector<int>& spins, double& energy, double& magnetization) {
    // Perform L individual spin flips
    for (int flip = 0; flip < L; flip++) {
        // Randomly select a site
        int i = std::uniform_int_distribution<>(0, L-1)(gen);
        
        int left = (i - 1 + L) % L;  // Periodic boundary condition
        int right = (i + 1) % L;

        // Calculate energy change for flipping this spin
        double delta_energy = 2.0 * J * spins[i] * (spins[left] + spins[right]) + 2.0 * h * spins[i];

        // Metropolis acceptance criterion
        if (delta_energy < 0.0 || dis(gen) < exp(-delta_energy / T)) {
            spins[i] = -spins[i];     // Flip the spin
            energy += delta_energy;    // Update energy
        }
    }

    // Update magnetization
    magnetization = 0.0;
    for (int i = 0; i < L; i++) {
        magnetization += spins[i];
    }
    magnetization /= L;
}

// Write data to file for visualization
void write_to_file(const std::string& filename, const std::vector<double>& data) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < data.size(); i++) {
            file << i << " " << data[i] << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// Main simulation
int main() {
    std::vector<int> spins(L);
    initialize_lattice(spins);

    double energy = calculate_energy(spins);
    double magnetization = 0.0;

    std::vector<double> energy_history;
    std::vector<double> magnetization_history;

    for (int step = 0; step < num_steps; step++) {
        monte_carlo_step(spins, energy, magnetization);
        energy_history.push_back(energy);
        magnetization_history.push_back(magnetization);

        // Print progress
        if (step % 1000 == 0) {
            std::cout << "Step: " << step
                      << " Energy: " << energy
                      << " Magnetization: " << magnetization << std::endl;
        }
    }

    // Write data to files
    write_to_file("out/cpu/1d/energy_with_field.dat", energy_history);
    write_to_file("out/cpu/1d/magnetization_with_field.dat", magnetization_history);

    return 0;
}
