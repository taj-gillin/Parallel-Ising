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
const int num_steps = 1000;     // Number of Monte Carlo steps
const double J = 1.0;          // Interaction strength
const double T = 2.5;          // Temperature (in units of k_B)
const double h = 0;            // External magnetic field
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

// Add these after the Timer class definition
struct PerformanceCounters {
    long long total_flops = 0;
    long long total_bytes = 0;
    
    void add_site_update() {
        // Count operations for one site update:
        // - 6 neighbor lookups and additions
        // - 1 multiplication for each neighbor (J * spin)
        // - 5 additions for summing neighbors
        // - 1 multiplication for external field
        // - 1 comparison
        // - 1 random number generation
        // - 1 exponential calculation (from lookup table)
        total_flops += 15;
        
        // Count memory operations:
        // - Read current spin (4 bytes)
        // - Read 6 neighbors (24 bytes)
        // - Write updated spin (4 bytes)
        total_bytes += 32;
    }
    
    void print_stats(double elapsed_time) {
        double gflops = static_cast<double>(total_flops) / (elapsed_time * 1e9);
        double bandwidth = static_cast<double>(total_bytes) / (elapsed_time * 1e9);
        double arithmetic_intensity = static_cast<double>(total_flops) / total_bytes;
        
        std::cout << "\nPerformance Counters:" << std::endl;
        std::cout << "Total FLOPS: " << total_flops << std::endl;
        std::cout << "Total Bytes: " << total_bytes << std::endl;
        std::cout << "GFLOPS: " << gflops << std::endl;
        std::cout << "Bandwidth (GB/s): " << bandwidth << std::endl;
        std::cout << "Arithmetic Intensity (FLOPS/byte): " << arithmetic_intensity << std::endl;
    }
};

// Perform a red-black Monte Carlo sweep
// parity: 0 = update spins where (i+j+k)%2 == 0
//         1 = update spins where (i+j+k)%2 == 1
void monte_carlo_sweep(const std::vector<std::vector<std::vector<int>>>& spins_old,
                       std::vector<std::vector<std::vector<int>>>& spins_new,
                       double &energy, double &magnetization, int parity,
                       PerformanceCounters& perf) {
    // Start with old configuration
    spins_new = spins_old;

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < L; j++) {
            for (int k = 0; k < L; k++) {
                // Update only the sub-lattice of chosen parity
                if (((i + j + k) % 2) == parity) {
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
                    
                    // Add performance counter update
                    perf.add_site_update();
                }
            }
        }
    }

    // After the half-sweep, calculate magnetization of the new configuration
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
    std::filesystem::create_directories("out/redblack/3d/lattice");

    Timer init_timer, mc_timer, io_timer, total_timer;
    total_timer.start();

    init_timer.start();
    std::vector<std::vector<std::vector<int>>> spins(L, std::vector<std::vector<int>>(L, std::vector<int>(L)));
    std::vector<std::vector<std::vector<int>>> new_spins(L, std::vector<std::vector<int>>(L, std::vector<int>(L)));
    initialize_lattice(spins);

    double energy = calculate_energy(spins);
    double magnetization = 0.0;
    calculate_magnetization(spins, magnetization);

    init_timer.stop();

    std::vector<double> energy_history;
    std::vector<double> magnetization_history;

    PerformanceCounters perf;
    mc_timer.start();
    // Each full "step" now consists of two half-sweeps: one for parity=0, one for parity=1
    // If you define num_steps as full sweeps, each sweep is two calls to monte_carlo_sweep.
    for (int step = 0; step < num_steps; step++) {
        // Update parity=0 sublattice
        monte_carlo_sweep(spins, new_spins, energy, magnetization, 0, perf);
        std::swap(spins, new_spins);

        // Update parity=1 sublattice
        monte_carlo_sweep(spins, new_spins, energy, magnetization, 1, perf);
        std::swap(spins, new_spins);

        energy_history.push_back(energy);
        magnetization_history.push_back(magnetization);

        if (step % save_interval == 0) {
            io_timer.start();
            write_lattice_to_file("out/redblack/3d/lattice/step_" + std::to_string(step) + ".dat", spins);
            io_timer.stop();

            std::cout << "Step: " << step
                      << " Energy: " << energy
                      << " Magnetization: " << magnetization << std::endl;
        }
    }
    mc_timer.stop();

    // Print performance statistics
    perf.print_stats(mc_timer.get_elapsed());

    io_timer.start();
    write_to_file("out/redblack/3d/energy_3d.dat", energy_history);
    write_to_file("out/redblack/3d/magnetization_3d.dat", magnetization_history);
    io_timer.stop();

    total_timer.stop();

    std::cout << "Timing Summary:\n";
    std::cout << "Initialization Time: " << init_timer.get_elapsed() << " seconds\n";
    std::cout << "Monte Carlo Steps Time: " << mc_timer.get_elapsed() << " seconds\n";
    std::cout << "File I/O Time: " << io_timer.get_elapsed() << " seconds\n";
    std::cout << "Total Execution Time: " << total_timer.get_elapsed() << " seconds\n";

    return 0;
}
