#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>

// Parameters for the simulation
const int L = 50;              // Lattice size (L x L grid)
const int num_steps = 10000;  // Number of Monte Carlo steps
const double J = 1.0;          // Interaction strength
const double T = 2.0;          // Temperature (in units of k_B)
const double h = 0.0;          // External magnetic field

// Random number generation
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

// Initialize lattice with random spins (+1 or -1)
void initialize_lattice(std::vector<std::vector<int>>& spins) {
    std::uniform_int_distribution<> spin_dis(0, 1);
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            spins[i][j] = spin_dis(gen) == 0 ? 1 : -1;
        }
    }
}

// Calculate the total energy of the system
double calculate_energy(const std::vector<std::vector<int>>& spins) {
    double energy = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            int up = (i - 1 + L) % L;    // Periodic boundary for top neighbor
            int down = (i + 1) % L;     // Periodic boundary for bottom neighbor
            int left = (j - 1 + L) % L; // Periodic boundary for left neighbor
            int right = (j + 1) % L;    // Periodic boundary for right neighbor

            // Interaction with neighbors
            energy -= J * spins[i][j] * (spins[up][j] + spins[down][j] + spins[i][left] + spins[i][right]);
            // Interaction with external magnetic field
            energy -= h * spins[i][j];
        }
    }
    return energy / 2.0; // Each pair is counted twice
}

// Perform one Monte Carlo step
void monte_carlo_step(std::vector<std::vector<int>>& spins, double& energy, double& magnetization) {
    // Perform L*L individual spin flips
    for (int flip = 0; flip < L * L; flip++) {
        // Randomly select a site
        int i = std::uniform_int_distribution<>(0, L-1)(gen);
        int j = std::uniform_int_distribution<>(0, L-1)(gen);

        int up = (i - 1 + L) % L;
        int down = (i + 1) % L;
        int left = (j - 1 + L) % L;
        int right = (j + 1) % L;

        // Calculate energy change for flipping this spin
        double delta_energy = 2.0 * J * spins[i][j] * 
            (spins[up][j] + spins[down][j] + spins[i][left] + spins[i][right]) + 
            2.0 * h * spins[i][j];

        // Metropolis acceptance criterion
        if (delta_energy < 0.0 || dis(gen) < exp(-delta_energy / T)) {
            spins[i][j] = -spins[i][j];  // Flip the spin
            energy += delta_energy;       // Update energy
        }
    }

    // Update magnetization
    magnetization = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            magnetization += spins[i][j];
        }
    }
    magnetization /= (L * L);
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

// Write the final lattice configuration to a file
void write_lattice_to_file(const std::string& filename, const std::vector<std::vector<int>>& spins) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < L; i++) {
            for (int j = 0; j < L; j++) {
                file << spins[i][j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

// Main simulation
int main() {
    std::vector<std::vector<int>> spins(L, std::vector<int>(L));
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
    write_to_file("out/cpu/2d/energy_2d.dat", energy_history);
    write_to_file("out/cpu/2d/magnetization_2d.dat", magnetization_history);
    write_lattice_to_file("out/cpu/2d/lattice_2d.dat", spins);

    return 0;
}
