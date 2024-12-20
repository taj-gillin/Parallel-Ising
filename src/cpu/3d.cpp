#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <filesystem>
#include <chrono>

using namespace std::chrono;

// Parameters for the simulation
const int L = 64;              // Lattice size (L x L x L grid)
const int num_steps = 1000;    // Number of Monte Carlo steps
const double J = 1.0;          // Interaction strength
const double T = 2.5;          // Temperature (in units of k_B)
const double h = 0.0;            // External magnetic field
const int save_interval = 100;   // Interval to save the lattice

// Timer utility
class Timer {
public:
    void start() {
        start_time = high_resolution_clock::now();
    }

    void stop() {
        end_time = high_resolution_clock::now();
        elapsed += duration_cast<duration<double>>(end_time - start_time).count();
    }

    double get_elapsed() const {
        return elapsed;
    }

    void reset() {
        elapsed = 0.0;
    }

private:
    high_resolution_clock::time_point start_time, end_time;
    double elapsed = 0.0;
};

// Random number generation
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

// Initialize lattice with random spins (+1 or -1)
void initialize_lattice(std::vector<std::vector<std::vector<int>>>& spins) {
    std::uniform_int_distribution<> spin_dis(0, 1);
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                spins[i][j][k] = (spin_dis(gen) == 0) ? 1 : -1;
            }
        }
    }
}

// Calculate the total energy of the system
double calculate_energy(const std::vector<std::vector<std::vector<int>>>& spins) {
    double energy = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                int left = (i - 1 + L) % L;
                int right = (i + 1) % L;
                int up = (j - 1 + L) % L;
                int down = (j + 1) % L;
                int front = (k - 1 + L) % L;
                int back = (k + 1) % L;

                energy -= J * spins[i][j][k] *
                          (spins[left][j][k] + spins[right][j][k] +
                           spins[i][up][k] + spins[i][down][k] +
                           spins[i][j][front] + spins[i][j][back]);
                energy -= h * spins[i][j][k];
            }
        }
    }
    return energy / 2.0; // Each pair is counted twice
}

void calculate_magnetization(const std::vector<std::vector<std::vector<int>>>& spins, double &magnetization) {
    magnetization = 0.0;
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                magnetization += spins[i][j][k];
            }
        }
    }
    magnetization /= (L * L * L);
}

// Perform one Monte Carlo sweep
// Instead of picking random spins, we attempt to flip every spin once.
void monte_carlo_sweep(const std::vector<std::vector<std::vector<int>>>& spins_old,
                       std::vector<std::vector<std::vector<int>>>& spins_new,
                       double &energy, double &magnetization) {
    // Start with a copy of the old configuration in the new array
    // so we only update spins_new based on spins_old
    spins_new = spins_old;

    // Attempt to flip each spin once
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                int left = (i - 1 + L) % L;
                int right = (i + 1) % L;
                int up = (j - 1 + L) % L;
                int down = (j + 1) % L;
                int front = (k - 1 + L) % L;
                int back = (k + 1) % L;

                double current_spin = spins_old[i][j][k];
                double sum_neighbors = spins_old[left][j][k] + spins_old[right][j][k] +
                                       spins_old[i][up][k] + spins_old[i][down][k] +
                                       spins_old[i][j][front] + spins_old[i][j][back];

                double delta_energy = 2.0 * J * current_spin * sum_neighbors + 2.0 * h * current_spin;

                // Metropolis acceptance
                if (delta_energy < 0.0 || dis(gen) < exp(-delta_energy / T)) {
                    // Flip spin
                    spins_new[i][j][k] = -spins_old[i][j][k];
                    // Update energy incrementally
                    energy += delta_energy;
                } else {
                    // No flip
                    spins_new[i][j][k] = spins_old[i][j][k];
                }
            }
        }
    }

    // After the sweep, calculate magnetization of the new configuration
    calculate_magnetization(spins_new, magnetization);
}

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

void write_lattice_to_file(const std::string& filename, const std::vector<std::vector<std::vector<int>>>& spins) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < L; i++) {
            file << "Slice " << i << ":\n";
            for (int j = 0; j < L; j++) {
                for (int k = 0; k < L; k++) {
                    file << spins[i][j][k] << " ";
                }
                file << "\n";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

int main() {
    std::filesystem::create_directories("out/cpu/3d/lattice");

    Timer init_timer, mc_timer, energy_timer, magnet_timer, io_timer, total_timer;
    total_timer.start();

    init_timer.start();
    std::vector<std::vector<std::vector<int>>> spins(L, std::vector<std::vector<int>>(
                                                        L, std::vector<int>(L)));
    std::vector<std::vector<std::vector<int>>> new_spins(L, std::vector<std::vector<int>>(
                                                             L, std::vector<int>(L)));
    initialize_lattice(spins);

    double energy = calculate_energy(spins);
    double magnetization = 0.0;
    calculate_magnetization(spins, magnetization);

    init_timer.stop();

    std::vector<double> energy_history;
    std::vector<double> magnetization_history;

    mc_timer.start();
    for (int step = 0; step < num_steps; step++) {
        // Perform a full sweep update
        monte_carlo_sweep(spins, new_spins, energy, magnetization);

        // Now new_spins holds the updated configuration.
        // Swap spins and new_spins so that "spins" always points
        // to the current configuration going into the next step
        std::swap(spins, new_spins);

        energy_history.push_back(energy);
        magnetization_history.push_back(magnetization);

        if (step % save_interval == 0) {
            io_timer.start();
            write_lattice_to_file("out/cpu/3d/lattice/step_" + std::to_string(step) + ".dat", spins);
            io_timer.stop();

            std::cout << "Step: " << step
                      << " Energy: " << energy
                      << " Magnetization: " << magnetization << std::endl;
        }
    }
    mc_timer.stop();

    io_timer.start();
    write_to_file("out/cpu/3d/energy_3d.dat", energy_history);
    write_to_file("out/cpu/3d/magnetization_3d.dat", magnetization_history);
    io_timer.stop();

    total_timer.stop();

    std::cout << "Timing Summary:\n";
    std::cout << "Initialization Time: " << init_timer.get_elapsed() << " seconds\n";
    std::cout << "Monte Carlo Steps Time: " << mc_timer.get_elapsed() << " seconds\n";
    std::cout << "File I/O Time: " << io_timer.get_elapsed() << " seconds\n";
    std::cout << "Total Execution Time: " << total_timer.get_elapsed() << " seconds\n";

    return 0;
}
