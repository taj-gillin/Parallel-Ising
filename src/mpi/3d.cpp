#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <mpi.h>
#include <filesystem>
#include <chrono>
#include <algorithm>

using namespace std::chrono;

// Parameters for the simulation
const int L = 21;              // Global lattice size (L x L x L grid)
const int num_steps = 100;     // Number of Monte Carlo steps
const double J = 1;          // Interaction strength
const double T = 2.5 ;          // Temperature (in units of k_B)
const double h = 0;            // External magnetic field
const int save_interval = 10;   // Interval to save the lattice

// Timer utility
class Timer {
public:
    void start() { start_time = high_resolution_clock::now(); }
    void stop() { end_time = high_resolution_clock::now(); elapsed += duration_cast<duration<double>>(end_time - start_time).count(); }
    double get_elapsed() { return elapsed; }
    void reset() { elapsed = 0.0; }

private:
    high_resolution_clock::time_point start_time, end_time;
    double elapsed = 0.0;
};

// Random number generation
std::random_device rd;
std::uniform_real_distribution<> dis(0.0, 1.0);

// Initialize lattice with random spins (+1 or -1)
void initialize_lattice(std::vector<std::vector<std::vector<int>>>& spins, int local_x, int local_y, int local_z, int rank) {
    // Create a unique generator for this process
    std::mt19937 gen(rd() + rank);  // Add rank to the seed
    std::uniform_int_distribution<> spin_dis(0, 1);
    for (int i = 0; i < local_x; i++) {
        for (int j = 0; j < local_y; j++) {
            for (int k = 0; k < local_z; k++) {
                spins[i][j][k] = spin_dis(gen) == 0 ? 1 : -1;
            }
        }
    }
}


// Halo exchange
void halo_exchange(std::vector<std::vector<std::vector<int>>>& spins, MPI_Comm cart_comm, int local_x, int local_y, int local_z) {
    // Ensure all processes are synchronized before exchange
    MPI_Barrier(cart_comm);

    static int exchange_count = 0;
    exchange_count++;
    
    // Verify boundary conditions before exchange
    if (exchange_count % 100 == 0) { // Check every 100 exchanges to avoid too much output
        int rank;
        MPI_Comm_rank(cart_comm, &rank);
        if (rank == 0) {
            std::cout << "\nBoundary Check at exchange " << exchange_count << std::endl;
        }
        
        // Check left boundary
        double left_avg = 0;
        for(int j = 0; j < local_y; j++) {
            for(int k = 0; k < local_z; k++) {
                left_avg += spins[0][j][k];
            }
        }
        left_avg /= (local_y * local_z);
        
        // Check right boundary
        double right_avg = 0;
        for(int j = 0; j < local_y; j++) {
            for(int k = 0; k < local_z; k++) {
                right_avg += spins[local_x-1][j][k];
            }
        }
        right_avg /= (local_y * local_z);
        
        std::cout << "Rank " << rank << " boundary averages - Left: " << left_avg 
                 << " Right: " << right_avg << std::endl;
    }
    
    MPI_Request requests[12];
    MPI_Status statuses[12];  // Add status tracking

    int left, right, up, down, front, back;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 2, 1, &front, &back);

    // Buffers for communication
    std::vector<int> send_left(local_y * local_z), recv_left(local_y * local_z);
    std::vector<int> send_right(local_y * local_z), recv_right(local_y * local_z);
    std::vector<int> send_up(local_x * local_z), recv_up(local_x * local_z);
    std::vector<int> send_down(local_x * local_z), recv_down(local_x * local_z);
    std::vector<int> send_front(local_x * local_y), recv_front(local_x * local_y);
    std::vector<int> send_back(local_x * local_y), recv_back(local_x * local_y);

    // Fill send buffers
    for (int j = 0; j < local_y; j++) {
        for (int k = 0; k < local_z; k++) {
            // Send the rightmost actual data (local_x-2) to the right neighbor's left boundary
            // Send the leftmost actual data (1) to the left neighbor's right boundary
            send_right[j * local_z + k] = spins[local_x-2][j][k];
            send_left[j * local_z + k] = spins[1][j][k];
        }
    }
    for (int i = 0; i < local_x; i++) {
        for (int k = 0; k < local_z; k++) {
            // Send the bottommost actual data (local_y-2) to the bottom neighbor's top boundary
            // Send the topmost actual data (1) to the top neighbor's bottom boundary
            send_down[i * local_z + k] = spins[i][local_y-2][k];
            send_up[i * local_z + k] = spins[i][1][k];
        }
    }
    for (int i = 0; i < local_x; i++) {
        for (int j = 0; j < local_y; j++) {
            // Send the backmost actual data (local_z-2) to the back neighbor's front boundary
            // Send the frontmost actual data (1) to the front neighbor's back boundary
            send_back[i * local_y + j] = spins[i][j][local_z-2];
            send_front[i * local_y + j] = spins[i][j][1];
        }
    }

    // Use MPI_Sendrecv instead of Isend/Irecv for more reliable exchange
    // X direction exchanges
    MPI_Sendrecv(send_left.data(), local_y * local_z, MPI_INT, left, 0,
                 recv_right.data(), local_y * local_z, MPI_INT, right, 0,
                 cart_comm, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv(send_right.data(), local_y * local_z, MPI_INT, right, 1,
                 recv_left.data(), local_y * local_z, MPI_INT, left, 1,
                 cart_comm, MPI_STATUS_IGNORE);

    // Y direction exchanges
    MPI_Sendrecv(send_up.data(), local_x * local_z, MPI_INT, up, 2,
                 recv_down.data(), local_x * local_z, MPI_INT, down, 2,
                 cart_comm, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv(send_down.data(), local_x * local_z, MPI_INT, down, 3,
                 recv_up.data(), local_x * local_z, MPI_INT, up, 3,
                 cart_comm, MPI_STATUS_IGNORE);

    // Z direction exchanges
    MPI_Sendrecv(send_front.data(), local_x * local_y, MPI_INT, front, 4,
                 recv_back.data(), local_x * local_y, MPI_INT, back, 4,
                 cart_comm, MPI_STATUS_IGNORE);
    
    MPI_Sendrecv(send_back.data(), local_x * local_y, MPI_INT, back, 5,
                 recv_front.data(), local_x * local_y, MPI_INT, front, 5,
                 cart_comm, MPI_STATUS_IGNORE);

    // Verify exchange success
    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    
    // Verify data was actually exchanged
    if (exchange_count % 100 == 0) {
        bool exchange_verified = true;
        // Check X direction
        for(int j = 0; j < local_y; j++) {
            for(int k = 0; k < local_z; k++) {
                if (spins[0][j][k] != recv_left[j * local_z + k] ||
                    spins[local_x-1][j][k] != recv_right[j * local_z + k]) {
                    exchange_verified = false;
                    break;
                }
            }
        }
        // Check Y direction
        for(int i = 0; i < local_x; i++) {
            for(int k = 0; k < local_z; k++) {
                if (spins[i][0][k] != recv_up[i * local_z + k] ||
                    spins[i][local_y-1][k] != recv_down[i * local_z + k]) {
                    exchange_verified = false;
                    break;
                }
            }
        }
        // Check Z direction
        for(int i = 0; i < local_x; i++) {
            for(int j = 0; j < local_y; j++) {
                if (spins[i][j][0] != recv_front[i * local_y + j] ||
                    spins[i][j][local_z-1] != recv_back[i * local_y + j]) {
                    exchange_verified = false;
                    break;
                }
            }
        }
        std::cout << "Rank " << rank << " Exchange verification: " 
                 << (exchange_verified ? "SUCCESS" : "FAILED") << std::endl;
    }

    // Update halo regions in the lattice
    // X direction
    for (int j = 0; j < local_y; j++) {
        for (int k = 0; k < local_z; k++) {
            spins[0][j][k] = recv_left[j * local_z + k];
            spins[local_x-1][j][k] = recv_right[j * local_z + k];
        }
    }
    // Y direction
    for (int i = 0; i < local_x; i++) {
        for (int k = 0; k < local_z; k++) {
            spins[i][0][k] = recv_up[i * local_z + k];
            spins[i][local_y-1][k] = recv_down[i * local_z + k];
        }
    }
    // Z direction
    for (int i = 0; i < local_x; i++) {
        for (int j = 0; j < local_y; j++) {
            spins[i][j][0] = recv_front[i * local_y + j];
            spins[i][j][local_z-1] = recv_back[i * local_y + j];
        }
    }

    // Ensure all processes complete exchange before continuing
    MPI_Barrier(cart_comm);
}

void monte_carlo_step(std::vector<std::vector<std::vector<int>>>& spins,
                      int local_x, int local_y, int local_z,
                      double& energy, double& magnetization,
                      MPI_Comm cart_comm, int rank) 
{
    // Ensure boundary data is up-to-date
    halo_exchange(spins, cart_comm, local_x, local_y, local_z);

    // Make a copy of the old configuration
    auto old_spins = spins;

    // Reset energy increment since we'll do a full recalculation after the sweep
    double delta_energy_accum = 0.0;

    // Sweep through all interior spins
    // (We exclude the ghost layers: 0 and local_x-1 in x, similarly for y and z)
    for (int i = 1; i < local_x - 1; i++) {
        for (int j = 1; j < local_y - 1; j++) {
            for (int k = 1; k < local_z - 1; k++) {
                double current_spin = old_spins[i][j][k];

                // Compute sum of neighbors from the old configuration
                double sum_neighbors = old_spins[i-1][j][k] + old_spins[i+1][j][k] +
                                       old_spins[i][j-1][k] + old_spins[i][j+1][k] +
                                       old_spins[i][j][k-1] + old_spins[i][j][k+1];

                double delta_energy = 2.0 * J * current_spin * sum_neighbors + 2.0 * h * current_spin;

                // Metropolis acceptance test
                double r = dis(rd);
                if (delta_energy < 0.0 || r < exp(-delta_energy / T)) {
                    // Flip the spin
                    spins[i][j][k] = -current_spin;
                    // Accumulate energy change
                    delta_energy_accum += delta_energy;
                } else {
                    // No flip
                    spins[i][j][k] = current_spin;
                }
            }
        }
    }

    // After the full sweep, recalculate total local energy and magnetization
    // to keep track correctly
    double local_energy = 0.0;
    double local_magnetization = 0.0;
    for (int i = 1; i < local_x - 1; i++) {
        for (int j = 1; j < local_y - 1; j++) {
            for (int k = 1; k < local_z - 1; k++) {
                double curr = spins[i][j][k];
                local_energy += -J * curr * (spins[i-1][j][k] + spins[i+1][j][k] +
                                             spins[i][j-1][k] + spins[i][j+1][k] +
                                             spins[i][j][k-1] + spins[i][j][k+1]) - h * curr;
                local_magnetization += curr;
            }
        }
    }

    // Update the references
    energy = local_energy;
    magnetization = local_magnetization / ((local_x - 2) * (local_y - 2) * (local_z - 2));
}


// Add these function declarations at the top of the file (after includes)
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

void write_to_file(const std::string& filename, const std::vector<double>& data) {
    std::ofstream file(filename);
    for (size_t i = 0; i < data.size(); i++) {
        file << i << " " << data[i] << "\n";
    }
}

// Add this function after the write_to_file function and before main
void gather_lattice_data(const std::vector<std::vector<std::vector<int>>>& local_spins,
                        std::vector<std::vector<std::vector<int>>>& full_lattice,
                        MPI_Comm cart_comm, int* dims, int* coords) {
    int rank, num_procs;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &num_procs);

    std::cout << "Rank " << rank << " entering gather_lattice_data" << std::endl;
    MPI_Barrier(cart_comm);  // Ensure all processes enter the function

    // Calculate local actual sizes (excluding ghost cells)
    int local_actual_x = L / dims[0];
    int local_actual_y = L / dims[1];
    int local_actual_z = L / dims[2];
    
    // Calculate send buffer size for this process
    int send_count = local_actual_x * local_actual_y * local_actual_z;
    
    std::cout << "Rank " << rank << " local sizes: " << local_actual_x << "x" 
              << local_actual_y << "x" << local_actual_z << " (send_count=" 
              << send_count << ")" << std::endl;
    MPI_Barrier(cart_comm);  // Synchronize after printing sizes
    
    // Verify all processes have same local sizes
    int all_counts[num_procs];
    MPI_Allgather(&send_count, 1, MPI_INT, all_counts, 1, MPI_INT, cart_comm);
    
    if (rank == 0) {
        for (int i = 0; i < num_procs; i++) {
            if (all_counts[i] != send_count) {
                std::cerr << "Error: Process " << i << " has different send_count: " 
                         << all_counts[i] << " vs " << send_count << std::endl;
                MPI_Abort(cart_comm, 1);
            }
        }
    }
    
    // Create send buffer and pack local data (excluding ghost cells)
    std::vector<int> send_buffer(send_count);
    int idx = 0;
    for(int i = 1; i < local_actual_x + 1; i++) {
        for(int j = 1; j < local_actual_y + 1; j++) {
            for(int k = 1; k < local_actual_z + 1; k++) {
                send_buffer[idx++] = local_spins[i][j][k];
            }
        }
    }

    std::cout << "Rank " << rank << " packed " << idx << " elements" << std::endl;
    MPI_Barrier(cart_comm);  // Synchronize after packing

    // All processes need these vectors
    std::vector<int> recvcounts(num_procs, send_count);  // All processes send same amount
    std::vector<int> displs(num_procs, 0);

    // Calculate displacements correctly for 3D decomposition
    for(int p = 0; p < num_procs; p++) {
        int proc_coords[3];
        MPI_Cart_coords(cart_comm, p, 3, proc_coords);
        
        // Calculate displacement considering all three dimensions
        displs[p] = (proc_coords[0] * local_actual_x * L * L +
                     proc_coords[1] * local_actual_y * L +
                     proc_coords[2] * local_actual_z);
    }

    // Only root needs the receive buffer
    std::vector<int> recv_buffer;
    if (rank == 0) {
        recv_buffer.resize(L * L * L);
        std::cout << "Rank 0 allocated receive buffer of size " << L * L * L << std::endl;
    }

    std::cout << "Rank " << rank << " before MPI_Gatherv" << std::endl;
    MPI_Barrier(cart_comm);  // Synchronize before gather

    // All processes must participate in Gatherv
    MPI_Gatherv(send_buffer.data(), send_count, MPI_INT,
                rank == 0 ? recv_buffer.data() : nullptr,
                recvcounts.data(), displs.data(),
                MPI_INT, 0, cart_comm);

    std::cout << "Rank " << rank << " after MPI_Gatherv" << std::endl;

    // Root process reconstructs the full 3D lattice
    if(rank == 0) {
        std::cout << "Rank 0 starting reconstruction" << std::endl;
        
        // Clear the full lattice first
        for(int i = 0; i < L; i++) {
            for(int j = 0; j < L; j++) {
                for(int k = 0; k < L; k++) {
                    full_lattice[i][j][k] = 0;
                }
            }
        }

        // Reconstruct the lattice
        for(int p = 0; p < num_procs; p++) {
            int proc_coords[3];
            MPI_Cart_coords(cart_comm, p, 3, proc_coords);
            
            int start_x = proc_coords[0] * local_actual_x;
            int start_y = proc_coords[1] * local_actual_y;
            int start_z = proc_coords[2] * local_actual_z;
            
            // Copy data from receive buffer to full lattice
            int buf_idx = displs[p];
            for(int i = 0; i < local_actual_x; i++) {
                for(int j = 0; j < local_actual_y; j++) {
                    for(int k = 0; k < local_actual_z; k++) {
                        full_lattice[start_x + i][start_y + j][start_z + k] = 
                            recv_buffer[buf_idx++];
                    }
                }
            }
        }
        std::cout << "Rank 0 finished reconstruction" << std::endl;
    }
    
    std::cout << "Rank " << rank << " exiting gather_lattice_data" << std::endl;
}

// Main function
int main(int argc, char** argv) {
    Timer init_timer, mc_timer, comm_timer, io_timer, total_timer;
    total_timer.start();

    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create output directory if rank 0
    if (rank == 0) {
        std::filesystem::create_directories("out/mpi/3d/lattice");
    }

    std::cout << "Rank " << rank << std::endl;
    

    // Create 3D Cartesian communicator
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(num_procs, 3, dims);
    int periods[3] = {1, 1, 1};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    // Initialize local lattice
    init_timer.start();
    int local_x = L / dims[0] + 2;
    int local_y = L / dims[1] + 2;
    int local_z = L / dims[2] + 2;
    std::vector<std::vector<std::vector<int>>> spins(local_x, 
        std::vector<std::vector<int>>(local_y, std::vector<int>(local_z)));
    initialize_lattice(spins, local_x, local_y, local_z, rank);
    init_timer.stop();

    double energy = 0.0, magnetization = 0.0;
    std::vector<double> energy_history;
    std::vector<double> magnetization_history;

    mc_timer.start();
    for (int step = 0; step < num_steps; step++) {
        comm_timer.start();
        halo_exchange(spins, cart_comm, local_x, local_y, local_z);
        comm_timer.stop();

        monte_carlo_step(spins, local_x, local_y, local_z, energy, magnetization, cart_comm, rank);

        double global_energy, global_magnetization;
        comm_timer.start();
        MPI_Reduce(&energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
        MPI_Reduce(&magnetization, &global_magnetization, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
        comm_timer.stop();

        // Update histories only on rank 0
        if (rank == 0) {
            energy_history.push_back(global_energy);
            magnetization_history.push_back(global_magnetization);
        }

        // All ranks participate in saving the lattice
        if (step % save_interval == 0) {
            io_timer.start();
            
            // Create full_lattice only on rank 0
            std::vector<std::vector<std::vector<int>>> full_lattice;
            if (rank == 0) {
                full_lattice.resize(L, 
                    std::vector<std::vector<int>>(L, std::vector<int>(L)));
            }
            
            // All ranks must participate in gather
            gather_lattice_data(spins, full_lattice, cart_comm, dims, coords);
            
            // Only rank 0 writes to file
            if (rank == 0) {
                write_lattice_to_file("out/mpi/3d/lattice/step_" + std::to_string(step) + ".dat", 
                                    full_lattice);
            }
            io_timer.stop();
        }

        // Print progress only on rank 0
        if (rank == 0 && step % 10 == 0) {
            std::cout << "Step: " << step
                      << " Energy: " << global_energy
                      << " Magnetization: " << global_magnetization << std::endl;
        }
    }
    mc_timer.stop();

    if (rank == 0) {    
        io_timer.start();
        write_to_file("out/mpi/3d/energy_3d.dat", energy_history);
        write_to_file("out/mpi/3d/magnetization_3d.dat", magnetization_history);
        io_timer.stop();
    }

    total_timer.stop();

    // Gather timing statistics from all processes
    std::vector<double> init_times(num_procs), mc_times(num_procs), 
                        comm_times(num_procs), io_times(num_procs), total_times(num_procs);
    
    double init_time = init_timer.get_elapsed();
    double mc_time = mc_timer.get_elapsed();
    double comm_time = comm_timer.get_elapsed();
    double io_time = io_timer.get_elapsed();
    double total_time = total_timer.get_elapsed();

    MPI_Gather(&init_time, 1, MPI_DOUBLE, init_times.data(), 1, MPI_DOUBLE, 0, cart_comm);
    MPI_Gather(&mc_time, 1, MPI_DOUBLE, mc_times.data(), 1, MPI_DOUBLE, 0, cart_comm);
    MPI_Gather(&comm_time, 1, MPI_DOUBLE, comm_times.data(), 1, MPI_DOUBLE, 0, cart_comm);
    MPI_Gather(&io_time, 1, MPI_DOUBLE, io_times.data(), 1, MPI_DOUBLE, 0, cart_comm);
    MPI_Gather(&total_time, 1, MPI_DOUBLE, total_times.data(), 1, MPI_DOUBLE, 0, cart_comm);

    if (rank == 0) {
        // Calculate statistics
        auto [min_init, max_init] = std::minmax_element(init_times.begin(), init_times.end());
        auto [min_mc, max_mc] = std::minmax_element(mc_times.begin(), mc_times.end());
        auto [min_comm, max_comm] = std::minmax_element(comm_times.begin(), comm_times.end());
        auto [min_io, max_io] = std::minmax_element(io_times.begin(), io_times.end());
        auto [min_total, max_total] = std::minmax_element(total_times.begin(), total_times.end());

        std::cout << "\nTiming Summary (min/max across all processes):\n";
        std::cout << "Initialization Time: " << *min_init << "/" << *max_init << " seconds\n";
        std::cout << "Monte Carlo Steps Time: " << *min_mc << "/" << *max_mc << " seconds\n";
        std::cout << "Communication Time: " << *min_comm << "/" << *max_comm << " seconds\n";
        std::cout << "File I/O Time: " << *min_io << "/" << *max_io << " seconds\n";
        std::cout << "Total Execution Time: " << *min_total << "/" << *max_total << " seconds\n";
        std::cout << "Number of processes: " << num_procs << " (" << dims[0] << "x" << dims[1] << "x" << dims[2] << ")\n";
    }

    MPI_Finalize();
    return 0;
}
