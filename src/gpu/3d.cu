#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <mpi.h>
#include <filesystem>
#include <chrono>
#include <algorithm>

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#define cudaGetDeviceCount     hipGetDeviceCount
#define cudaSetDevice          hipSetDevice
#define cudaDeviceSynchronize  hipDeviceSynchronize
#define cudaMalloc             hipMalloc
#define cudaFree               hipFree
#define cudaHostMalloc         hipHostMalloc
#define cudaMemcpy             hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaError_t            hipError_t
#define cudaMemset             hipMemset
#define cudaGetErrorString     hipGetErrorString
#define cudaSuccess            hipSuccess
#define cudaGetLastError       hipGetLastError
#define cudaEvent_t            hipEvent_t
#define cudaEventCreate        hipEventCreate
#define cudaEventRecord        hipEventRecord
#define cudaEventSynchronize   hipEventSynchronize
#define cudaEventElapsedTime   hipEventElapsedTime
#define cudaDeviceProp         hipDeviceProp_t
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaStream_t           hipStream_t
#define cudaStreamCreate       hipStreamCreate
#define cudaStreamSynchronize  hipStreamSynchronize
#define cudaHostAllocDefault   hipHostMallocDefault
#define cudaMemcpyAsync        hipMemcpyAsync
#define cudaMemsetAsync        hipMemsetAsync
#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>
#include <rocrand/rocrand_kernel.h>
#define curandState rocrand_state_xorwow
#define curand_init rocrand_init_xorwow
#define curand_uniform rocrand_uniform
#else
#include <curand.h>
#include <curand_kernel.h>
#endif

// Add CUDA error checking helper
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Add MPI error checking helper
#define MPI_CHECK(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            char error_string[MPI_MAX_ERROR_STRING]; \
            int length; \
            MPI_Error_string(err, error_string, &length); \
            fprintf(stderr, "MPI error at %s:%d: %s\n", \
                    __FILE__, __LINE__, error_string); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

using namespace std::chrono;

// Parameters for the simulation
const int L = 64;              // Global lattice size (L x L x L grid)
const int num_steps = 1000;     // Number of Monte Carlo steps
const double J = 1.0;          // Interaction strength
const double T = 1.0;          // Temperature (in units of J/k_B) - lowered from 2.5
const double h = 0.0;            // External magnetic field
const int save_interval = 100;  // Interval to save the lattice

// RNG and exp table
const int MAX_ENERGY = 24;  // Max possible energy change for 6 neighbors
std::vector<double> exp_table;

// Add initialization before main loop
void init_exp_table() {
    exp_table.resize(MAX_ENERGY + 1);
    for (int dE = 0; dE <= MAX_ENERGY; dE += 4) {
        exp_table[dE] = exp(-static_cast<double>(dE) / 2.5);
    }
}

// Thread-local RNG
thread_local std::mt19937 gen(std::random_device{}());
thread_local std::uniform_real_distribution<> dis(0.0, 1.0);


void init_rng(int rank) {
    gen.seed(std::random_device{}() + rank);
}

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

// Add this after Timer class and before any function declarations
struct GPUPerformanceCounters {
    // Operation counters for GPU only
    long long compute_flops = 0;      // GPU floating point operations
    long long memory_bytes = 0;        // GPU global memory traffic
    
    // GPU specifications (MI250X)
    const double peak_flops = 47.9 * 1e12;     // 47.9 TFLOPS
    const double peak_bandwidth = 1638.4 * 1e9; // 1.6 TB/s HBM2e bandwidth
    
    void add_spin_update(int num_sites) {
        // Per spin site on GPU:
        // Compute: ~15 FLOPs
        // - 6 additions for neighbor sum
        // - 2 multiplications (J*sum, h*spin)
        // - 1 addition for energy
        // - 2 multiplications for energy comparison
        // - 4 misc operations (exp, comparisons, etc)
        compute_flops += num_sites * 15;

        // Memory traffic: Count each memory operation
        // - 6 neighbor reads (6 loads)
        // - 1 center spin read (1 load)
        // - 1 center spin write (1 store)
        // Each int is 4 bytes, and we need to count both loads and stores
        memory_bytes += num_sites * (7 * 4 + 1 * 4); // 7 loads + 1 store, each 4 bytes
    }
    
    void add_energy_calc(int num_sites) {
        // Per spin site on GPU:
        // Compute: ~10 FLOPs
        // - 6 additions for neighbor sum
        // - 2 multiplications (J*sum, h*spin)
        // - 2 additions for energy/magnetization
        compute_flops += num_sites * 10;

        // Memory traffic:
        // - 6 neighbor reads (6 loads)
        // - 1 center spin read (1 load)
        // - 2 atomic operations for energy/magnetization (2 read-modify-write = 4 operations)
        // Each int is 4 bytes, doubles are 8 bytes
        memory_bytes += num_sites * (7 * 4) + // 7 int loads (4 bytes each)
                       num_sites * (4 * 8);   // 4 double operations for atomics (8 bytes each)
    }
    
    void print_stats(double elapsed_time, MPI_Comm comm) {
        int rank, num_procs;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &num_procs);
        
        // Gather statistics from all processes
        long long global_flops = 0, global_mem_bytes = 0;
        MPI_Allreduce(&compute_flops, &global_flops, 1, MPI_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(&memory_bytes, &global_mem_bytes, 1, MPI_LONG_LONG, MPI_SUM, comm);
        
        if (rank == 0) {
            std::cout << "\n=== GPU Roofline Analysis ===\n";
            
            // Calculate achieved performance metrics
            double achieved_flops = static_cast<double>(global_flops) / elapsed_time;
            double achieved_bandwidth = static_cast<double>(global_mem_bytes) / elapsed_time;
            
            // Calculate arithmetic intensity (FLOPs per byte of memory traffic)
            double arithmetic_intensity = static_cast<double>(global_flops) / global_mem_bytes;
            
            // Theoretical peak performance at this arithmetic intensity
            double peak_performance = std::min(peak_flops, peak_bandwidth * arithmetic_intensity);
            double achieved_fraction = (achieved_flops / peak_performance) * 100.0;
            
            // Ridge point (FLOPs/byte where compute and memory bounds intersect)
            double ridge_point = peak_flops / peak_bandwidth;
            
            std::cout << "\nGPU Operation Counts:\n";
            std::cout << "Total GPU FLOPs: " << global_flops << "\n";
            std::cout << "Total GPU Memory Traffic: " << global_mem_bytes << " bytes\n";
            
            std::cout << "\nGPU Performance Metrics:\n";
            std::cout << "Achieved FLOPS: " << achieved_flops / 1e12 << " TFLOPS\n";
            std::cout << "Memory Bandwidth: " << achieved_bandwidth / 1e9 << " GB/s\n";
            
            std::cout << "\nGPU Roofline Analysis:\n";
            std::cout << "Arithmetic Intensity: " << arithmetic_intensity << " FLOPS/byte\n";
            std::cout << "Ridge Point: " << ridge_point << " FLOPS/byte\n";
            std::cout << "Achieved: " << achieved_fraction << "% of theoretical peak\n";
            
            if (arithmetic_intensity < ridge_point) {
                std::cout << "\nKernel is MEMORY BANDWIDTH bound\n";
            } else {
                std::cout << "\nKernel is COMPUTE bound\n";
            }
        }
    }
};

// Forward declare monte_carlo_step_gpu with the GPUPerformanceCounters parameter
void monte_carlo_step_gpu(int* d_spins,  
                         double* d_rand_nums,
                         double* d_local_energy, double* d_local_magnet,
                         int local_x, int local_y, int local_z,
                         double& energy, double& magnetization,
                         MPI_Comm cart_comm, int rank, double current_T,
                         GPUPerformanceCounters& perf);

// GPU kernels
// Flattening index:
// index = i + local_x * (j + local_y * k)
// local_x, local_y, local_z are dimensions including ghost layers
// The "interior" spins are at 1 <= i < local_x-1, etc.

// Kernel to perform parity update
__global__
void gpu_update_parity(int* spins,
                       curandState* states,
                       int local_x, int local_y, int local_z, int parity,
                       double J, double h, double T, const double* exp_table) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i < local_x-1 &&
        j >= 1 && j < local_y-1 &&
        k >= 1 && k < local_z-1) {
        if (((i + j + k) % 2) == parity) {
            int idx = i + local_x*(j + local_y*k);
            int current_spin = spins[idx];
            
            int sum_neighbors = spins[(i-1) + local_x*(j + local_y*k)]
                              + spins[(i+1) + local_x*(j + local_y*k)]
                              + spins[i + local_x*((j-1) + local_y*k)]
                              + spins[i + local_x*((j+1) + local_y*k)]
                              + spins[i + local_x*(j + local_y*(k-1))]
                              + spins[i + local_x*(j + local_y*(k+1))];

            // Calculate energy change for proposed flip
            double dE = 2.0 * current_spin * (J * sum_neighbors + h);
            
            // Accept if dE < 0 or with probability exp(-dE/T)
            if (dE <= 0.0) {
                spins[idx] = -current_spin;
            } else {
                float rand_num = curand_uniform(&states[idx]);
                if (rand_num < exp(-dE/T)) {
                    spins[idx] = -current_spin;
                }
            }
        }
    }
}

// Kernel to compute local energy and magnetization
__global__
void gpu_compute_energy_magnetization(const int* spins,
                                      int local_x, int local_y, int local_z,
                                      double J, double h,
                                      double* local_energy, double* local_magnet) {
    extern __shared__ double sdata[]; 
    double* senergy = sdata; 
    double* smagnet = sdata + blockDim.x*blockDim.y*blockDim.z;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    int idx = i + local_x*(j + local_y*k);

    double e = 0.0;
    double m = 0.0;

    if (i >= 1 && i < local_x-1 &&
        j >= 1 && j < local_y-1 &&
        k >= 1 && k < local_z-1) {
        double curr = (double)spins[idx];
        double sum_neighbors = spins[(i-1) + local_x*(j + local_y*k)]
                             + spins[(i+1) + local_x*(j + local_y*k)]
                             + spins[i + local_x*((j-1) + local_y*k)]
                             + spins[i + local_x*((j+1) + local_y*k)]
                             + spins[i + local_x*(j + local_y*(k-1))]
                             + spins[i + local_x*(j + local_y*(k+1))];

        e = -J * curr * sum_neighbors - h * curr;
        m = curr;
    }

    // Compute 3D thread ID for shared mem index
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int tid = tx + blockDim.x*(ty + blockDim.y*tz);

    senergy[tid] = e;
    smagnet[tid] = m;

    __syncthreads();

    // Reduce within block
    int total_threads = blockDim.x*blockDim.y*blockDim.z;
    for (int stride = total_threads/2; stride > 0; stride /= 2) {
        if (tid < stride) {
            senergy[tid] += senergy[tid+stride];
            smagnet[tid] += smagnet[tid+stride];
        }
        __syncthreads();
    }

    // Write block results to global memory
    if (tid == 0) {
        atomicAdd(local_energy, senergy[0]);
        atomicAdd(local_magnet, smagnet[0]);
    }
}

// Host helper functions

void initialize_lattice(std::vector<std::vector<std::vector<int>>>& spins, int local_x, int local_y, int local_z, int rank) {
    // Initialize all spins to +1 (or randomly with high probability of +1)
    thread_local std::mt19937 init_gen(std::random_device{}() + rank);
    thread_local std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < local_x; i++) {
        for (int j = 0; j < local_y; j++) {
            for (int k = 0; k < local_z; k++) {
                // 90% chance of +1, 10% chance of -1
                spins[i][j][k] = (dis(init_gen) < 0.9) ? 1 : -1;
            }
        }
    }
}

// Halo exchange function from previous code remains the same
void halo_exchange(std::vector<std::vector<std::vector<int>>>& spins, MPI_Comm cart_comm, int local_x, int local_y, int local_z) {
    MPI_Barrier(cart_comm);
    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    int left, right, up, down, front, back;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 2, 1, &front, &back);

    std::vector<int> send_left(local_y * local_z), recv_left(local_y * local_z);
    std::vector<int> send_right(local_y * local_z), recv_right(local_y * local_z);
    std::vector<int> send_up(local_x * local_z), recv_up(local_x * local_z);
    std::vector<int> send_down(local_x * local_z), recv_down(local_x * local_z);
    std::vector<int> send_front(local_x * local_y), recv_front(local_x * local_y);
    std::vector<int> send_back(local_x * local_y), recv_back(local_x * local_y);

    for (int j = 0; j < local_y; j++) {
        for (int k = 0; k < local_z; k++) {
            send_right[j * local_z + k] = spins[local_x-2][j][k];
            send_left[j * local_z + k]  = spins[1][j][k];
        }
    }
    for (int i = 0; i < local_x; i++) {
        for (int k = 0; k < local_z; k++) {
            send_down[i * local_z + k] = spins[i][local_y-2][k];
            send_up[i * local_z + k]   = spins[i][1][k];
        }
    }
    for (int i = 0; i < local_x; i++) {
        for (int j = 0; j < local_y; j++) {
            send_back[i * local_y + j]  = spins[i][j][local_z-2];
            send_front[i * local_y + j] = spins[i][j][1];
        }
    }

    // X direction
    MPI_Sendrecv(send_left.data(), local_y*local_z, MPI_INT, left, 0,
                 recv_right.data(), local_y*local_z, MPI_INT, right, 0, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_right.data(), local_y*local_z, MPI_INT, right, 1,
                 recv_left.data(), local_y*local_z, MPI_INT, left, 1, cart_comm, MPI_STATUS_IGNORE);

    // Y direction
    MPI_Sendrecv(send_up.data(), local_x*local_z, MPI_INT, up, 2,
                 recv_down.data(), local_x*local_z, MPI_INT, down, 2, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_down.data(), local_x*local_z, MPI_INT, down, 3,
                 recv_up.data(), local_x*local_z, MPI_INT, up, 3, cart_comm, MPI_STATUS_IGNORE);

    // Z direction
    MPI_Sendrecv(send_front.data(), local_x*local_y, MPI_INT, front, 4,
                 recv_back.data(), local_x*local_y, MPI_INT, back, 4, cart_comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_back.data(), local_x*local_y, MPI_INT, back, 5,
                 recv_front.data(), local_x*local_y, MPI_INT, front, 5, cart_comm, MPI_STATUS_IGNORE);

    for (int j = 0; j < local_y; j++) {
        for (int k = 0; k < local_z; k++) {
            spins[0][j][k] = recv_left[j * local_z + k];
            spins[local_x-1][j][k] = recv_right[j * local_z + k];
        }
    }
    for (int i = 0; i < local_x; i++) {
        for (int k = 0; k < local_z; k++) {
            spins[i][0][k] = recv_up[i * local_z + k];
            spins[i][local_y-1][k] = recv_down[i * local_z + k];
        }
    }
    for (int i = 0; i < local_x; i++) {
        for (int j = 0; j < local_y; j++) {
            spins[i][j][0] = recv_front[i * local_y + j];
            spins[i][j][local_z-1] = recv_back[i * local_y + j];
        }
    }

    MPI_Barrier(cart_comm);
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

void write_to_file(const std::string& filename, const std::vector<double>& data) {
    std::ofstream file(filename);
    for (size_t i = 0; i < data.size(); i++) {
        file << i << " " << data[i] << "\n";
    }
}

void gather_lattice_data(const std::vector<std::vector<std::vector<int>>>& local_spins,
                        std::vector<std::vector<std::vector<int>>>& full_lattice,
                        MPI_Comm cart_comm, int* dims, int* coords) {
    int rank, num_procs;
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Comm_size(cart_comm, &num_procs);

    // local_actual sizes
    int local_actual_x = L / dims[0];
    int local_actual_y = L / dims[1];
    int local_actual_z = L / dims[2];

    int send_count = local_actual_x * local_actual_y * local_actual_z;

    // Pack local data (excluding ghost layers)
    std::vector<int> send_buffer(send_count);
    int idx = 0;
    for(int i = 1; i < local_actual_x+1; i++) {
        for(int j = 1; j < local_actual_y+1; j++) {
            for(int k = 1; k < local_actual_z+1; k++) {
                send_buffer[idx++] = local_spins[i][j][k];
            }
        }
    }

    std::vector<int> recvcounts(num_procs, send_count);
    std::vector<int> displs(num_procs, 0);
    for(int p = 0; p < num_procs; p++) {
        int proc_coords[3];
        MPI_Cart_coords(cart_comm, p, 3, proc_coords);
        displs[p] = (proc_coords[0]*local_actual_x)*L*L
                   + (proc_coords[1]*local_actual_y)*L
                   + (proc_coords[2]*local_actual_z);
    }

    std::vector<int> recv_buffer;
    if (rank == 0) {
        recv_buffer.resize(L*L*L);
    }

    MPI_Gatherv(send_buffer.data(), send_count, MPI_INT,
                rank == 0 ? recv_buffer.data() : nullptr,
                recvcounts.data(), displs.data(),
                MPI_INT, 0, cart_comm);

    if(rank == 0) {
        for(int i = 0; i < L; i++) {
            for(int j = 0; j < L; j++) {
                for(int k = 0; k < L; k++) {
                    full_lattice[i][j][k] = 0;
                }
            }
        }

        for(int p = 0; p < num_procs; p++) {
            int proc_coords[3];
            MPI_Cart_coords(cart_comm, p, 3, proc_coords);
            int start_x = proc_coords[0]*local_actual_x;
            int start_y = proc_coords[1]*local_actual_y;
            int start_z = proc_coords[2]*local_actual_z;

            int buf_idx = displs[p];
            for(int i = 0; i < local_actual_x; i++) {
                for(int j = 0; j < local_actual_y; j++) {
                    for(int k = 0; k < local_actual_z; k++) {
                        full_lattice[start_x + i][start_y + j][start_z + k] = recv_buffer[buf_idx++];
                    }
                }
            }
        }
    }
}

// Modify the DetailedTimers class to include RNG initialization time
class DetailedTimers {
public:
    Timer random_gen;    // Time spent generating random numbers
    Timer rng_init;      // Time spent initializing RNG (one-time cost)
    Timer kernel_parity; // Time spent in parity update kernels
    Timer halo;         // Time spent in halo exchange
    Timer energy;       // Time spent computing energy/magnetization
    Timer memcpy;       // Time spent in GPU memcpy operations
    
    void print_stats(MPI_Comm comm) {
        int rank;
        MPI_Comm_rank(comm, &rank);
        
        // if (rank == 0) {
        //     std::cout << "\nDetailed Monte Carlo Timing Breakdown:\n";
        //     std::cout << "RNG initialization (one-time): " << rng_init.get_elapsed() << " seconds\n";
        //     std::cout << "Random number generation: " << random_gen.get_elapsed() << " seconds\n";
        //     std::cout << "Parity update kernels: " << kernel_parity.get_elapsed() << " seconds\n";
        //     std::cout << "Halo exchange: " << halo.get_elapsed() << " seconds\n";
        //     std::cout << "Energy computation: " << energy.get_elapsed() << " seconds\n";
        //     std::cout << "Memory transfers: " << memcpy.get_elapsed() << " seconds\n";
        // }
    }
};

// Add this as a global variable
DetailedTimers detailed_timers;

// Add near the top of the file, after the DetailedTimers declaration
curandState* d_states = nullptr;  // Global declaration

// Add new kernel for random number generation initialization
__global__ 
void init_rng_kernel(curandState* states, unsigned long long seed, int volume) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < volume) {
        #ifdef USE_HIP
        rocrand_init(seed, idx, 0, &states[idx]);
        #else
        curand_init(seed, idx, 0, &states[idx]);
        #endif
    }
}

// Then modify monte_carlo_step_gpu to include the timers:
void monte_carlo_step_gpu(int* d_spins,  
                         double* d_rand_nums,
                         double* d_local_energy, double* d_local_magnet,
                         int local_x, int local_y, int local_z,
                         double& energy, double& magnetization,
                         MPI_Comm cart_comm, int rank, double current_T,
                         GPUPerformanceCounters& perf)
{
    int volume = local_x * local_y * local_z;
    Timer compute_timer, transfer_timer, halo_timer, reduce_timer;
    
    static bool first_call = true;  // Keep this static
    
    if (first_call) {
        Timer rng_init_timer;
        rng_init_timer.start();
        CUDA_CHECK(cudaMalloc(&d_states, volume * sizeof(curandState)));
        
        // Initialize RNG states
        int block_size = 256;
        int grid_size = (volume + block_size - 1) / block_size;
        unsigned long long seed = static_cast<unsigned long long>(time(NULL)) + rank;
        init_rng_kernel<<<grid_size, block_size>>>(d_states, seed, volume);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        rng_init_timer.stop();
        if (rank == 0) {
            std::cout << "RNG initialization time: " << rng_init_timer.get_elapsed() << "s\n";
        }
        first_call = false;
    }

    // Declare d_exp_table
    double* d_exp_table;
    
    // Copy exp_table to device
    CUDA_CHECK(cudaMalloc(&d_exp_table, (MAX_ENERGY + 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_exp_table, exp_table.data(), (MAX_ENERGY + 1) * sizeof(double), 
               cudaMemcpyHostToDevice));

    // Modify the gpu_update_parity kernel to generate random numbers inline
    dim3 block(8,8,8);
    dim3 grid((local_x+block.x-1)/block.x,(local_y+block.y-1)/block.y,(local_z+block.z-1)/block.z);

    // Parity = 0 update
    gpu_update_parity<<<grid, block>>>(d_spins, d_states, local_x, local_y, local_z, 0, J, h, current_T, d_exp_table);
    CUDA_CHECK(cudaDeviceSynchronize());
    perf.add_spin_update((local_x-2)*(local_y-2)*(local_z-2)); // Interior points only

    // Convert 1D h_spins to 3D h_spins_3d before halo_exchange
    std::vector<int> h_spins(volume);
    CUDA_CHECK(cudaMemcpy(h_spins.data(), d_spins, volume*sizeof(int), cudaMemcpyDeviceToHost));
    
    std::vector<std::vector<std::vector<int>>> h_spins_3d(local_x,
        std::vector<std::vector<int>>(local_y, std::vector<int>(local_z)));
    for (int i = 0; i < local_x; i++) {
        for (int j = 0; j < local_y; j++) {
            for (int k = 0; k < local_z; k++) {
                int idx = i + local_x*(j + local_y*k);
                h_spins_3d[i][j][k] = h_spins[idx];
            }
        }
    }

    // Perform halo exchange using the 3D array
    halo_exchange(h_spins_3d, cart_comm, local_x, local_y, local_z);

    // Convert h_spins_3d back to 1D
    for (int i = 0; i < local_x; i++) {
        for (int j = 0; j < local_y; j++) {
            for (int k = 0; k < local_z; k++) {
                int idx = i + local_x*(j + local_y*k);
                h_spins[idx] = h_spins_3d[i][j][k];
            }
        }
    }

    // Copy updated spins back to device
    CUDA_CHECK(cudaMemcpy(d_spins, h_spins.data(), volume*sizeof(int), cudaMemcpyHostToDevice));

    // Parity = 1 update
    gpu_update_parity<<<grid, block>>>(d_spins, d_states, local_x, local_y, local_z, 1, J, h, current_T, d_exp_table);
    CUDA_CHECK(cudaDeviceSynchronize());
    perf.add_spin_update((local_x-2)*(local_y-2)*(local_z-2)); // Interior points only

    // Halo exchange after parity=1
    CUDA_CHECK(cudaMemcpy(h_spins.data(), d_spins, volume*sizeof(int), cudaMemcpyDeviceToHost));

    // 1D -> 3D conversion and halo exchange
    for (int i = 0; i < local_x; i++) {
        for (int j = 0; j < local_y; j++) {
            for (int k = 0; k < local_z; k++) {
                int idx = i + local_x*(j + local_y*k);
                h_spins_3d[i][j][k] = h_spins[idx];
            }
        }
    }
    halo_exchange(h_spins_3d, cart_comm, local_x, local_y, local_z);
    // 3D -> 1D conversion
    for (int i = 0; i < local_x; i++) {
        for (int j = 0; j < local_y; j++) {
            for (int k = 0; k < local_z; k++) {
                int idx = i + local_x*(j + local_y*k);
                h_spins[idx] = h_spins_3d[i][j][k];
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(d_spins, h_spins.data(), volume*sizeof(int), cudaMemcpyHostToDevice));

    // Energy computation
    CUDA_CHECK(cudaMemset(d_local_energy, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_magnet, 0, sizeof(double)));

    size_t shared_size = 2*block.x*block.y*block.z*sizeof(double);
    gpu_compute_energy_magnetization<<<grid, block, shared_size>>>(
        d_spins, local_x, local_y, local_z, J, h, d_local_energy, d_local_magnet);
    CUDA_CHECK(cudaDeviceSynchronize());
    perf.add_energy_calc((local_x-2)*(local_y-2)*(local_z-2));

    double local_energy, local_m;
    CUDA_CHECK(cudaMemcpy(&local_energy, d_local_energy, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&local_m, d_local_magnet, sizeof(double), cudaMemcpyDeviceToHost));

    double global_energy, global_m;
    MPI_Reduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_m, &global_m, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);

    if (rank == 0) {
        magnetization = global_m / (L*L*L);
        energy = global_energy;
    }

    // Free d_exp_table
    CUDA_CHECK(cudaFree(d_exp_table));
}

// Add this function before main()
double get_temperature(int step, int total_steps) {
    double T_start = 2.5;
    double T_end = 0.5;
    return T_start * pow(T_end/T_start, static_cast<double>(step)/total_steps);
}

// Main
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    init_exp_table();
    init_rng(rank);

    // Create output directory if rank 0
    if (rank == 0) {
        std::filesystem::create_directories("out/gpu/3d/lattice");
    }

    // Create 3D Cartesian communicator
    int dims[3] = {0, 0, 0};
    MPI_Dims_create(num_procs, 3, dims);
    int periods[3] = {1, 1, 1};
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, rank, 3, coords);

    // GPU selection logic
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    int local_rank = rank % num_gpus; // Simple assignment if no SLURM or else use local rank env
    CUDA_CHECK(cudaSetDevice(local_rank));

    // Local sizes with ghost layers
    int local_x = L/dims[0] + 2;
    int local_y = L/dims[1] + 2;
    int local_z = L/dims[2] + 2;

    // Initialize lattice on CPU
    std::vector<std::vector<std::vector<int>>> spins(local_x, std::vector<std::vector<int>>(local_y, std::vector<int>(local_z)));
    initialize_lattice(spins, local_x, local_y, local_z, rank);

    double energy = 0.0, magnetization = 0.0;
    std::vector<double> energy_history;
    std::vector<double> magnetization_history;

    // Allocate GPU memory
    int volume = local_x*local_y*local_z;
    int* d_spins;
    double* d_rand_nums;
    double* d_local_energy;
    double* d_local_magnet;

    CUDA_CHECK(cudaMalloc(&d_spins, volume*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rand_nums, volume*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_energy, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_magnet, sizeof(double)));

    // Copy initial spins to GPU
    {
        std::vector<int> h_spins_flat(volume);
        for (int i = 0; i < local_x; i++) {
            for (int j = 0; j < local_y; j++) {
                for (int k = 0; k < local_z; k++) {
                    int idx = i + local_x*(j + local_y*k);
                    h_spins_flat[idx] = spins[i][j][k];
                }
            }
        }
        CUDA_CHECK(cudaMemcpy(d_spins, h_spins_flat.data(), volume*sizeof(int), cudaMemcpyHostToDevice));
    }

    Timer init_timer, mc_timer, comm_timer, io_timer, total_timer;
    total_timer.start();
    init_timer.start();
    init_timer.stop();

    GPUPerformanceCounters perf;
    mc_timer.start();
    for (int step = 0; step < num_steps; step++) {
        double current_T = get_temperature(step, num_steps);
        monte_carlo_step_gpu(d_spins, d_rand_nums, d_local_energy, d_local_magnet,
                             local_x, local_y, local_z, energy, magnetization,
                             cart_comm, rank, current_T, perf);

        double global_energy = energy;
        double global_magnetization = magnetization;

        if (rank == 0) {
            energy_history.push_back(global_energy);
            magnetization_history.push_back(global_magnetization);
        }

        // Periodically save lattice
        if (step % save_interval == 0) {
            io_timer.start();
            // Gather full lattice to rank 0
            std::vector<int> h_spins_flat(volume);
            CUDA_CHECK(cudaMemcpy(h_spins_flat.data(), d_spins, volume*sizeof(int), cudaMemcpyDeviceToHost));
            // Rebuild spins array for gather
            for (int i = 0; i < local_x; i++) {
                for (int j = 0; j < local_y; j++) {
                    for (int k = 0; k < local_z; k++) {
                        spins[i][j][k] = h_spins_flat[i + local_x*(j + local_y*k)];
                    }
                }
            }

            std::vector<std::vector<std::vector<int>>> full_lattice;
            if (rank == 0) {
                full_lattice.resize(L,
                    std::vector<std::vector<int>>(L, std::vector<int>(L)));
            }
            gather_lattice_data(spins, full_lattice, cart_comm, dims, coords);

            if (rank == 0) {
                write_lattice_to_file("out/gpu/3d/lattice/step_" + std::to_string(step) + ".dat", 
                                    full_lattice);
            }
            io_timer.stop();
        }

        if (rank == 0 && step % 10 == 0) {
            std::cout << "Step: " << step
                      << " Energy: " << energy
                      << " Magnetization: " << magnetization << std::endl;
        }
    }
    mc_timer.stop();

    if (rank == 0) {    
        io_timer.start();
        write_to_file("out/gpu/3d/energy_3d.dat", energy_history);
        write_to_file("out/gpu/3d/magnetization_3d.dat", magnetization_history);
        io_timer.stop();
    }

    total_timer.stop();

    // Gather timing info
    double init_time = init_timer.get_elapsed();
    double mc_time = mc_timer.get_elapsed();
    double comm_time = 0.0; // Not separately measured now
    double io_time_val = io_timer.get_elapsed();
    double total_time = total_timer.get_elapsed();

    std::vector<double> init_times(num_procs), mc_times(num_procs), 
                        comm_times(num_procs), io_times(num_procs), total_times(num_procs);

    MPI_Gather(&init_time, 1, MPI_DOUBLE, init_times.data(), 1, MPI_DOUBLE, 0, cart_comm);
    MPI_Gather(&mc_time, 1, MPI_DOUBLE, mc_times.data(), 1, MPI_DOUBLE, 0, cart_comm);
    MPI_Gather(&comm_time, 1, MPI_DOUBLE, comm_times.data(), 1, MPI_DOUBLE, 0, cart_comm);
    MPI_Gather(&io_time_val, 1, MPI_DOUBLE, io_times.data(), 1, MPI_DOUBLE, 0, cart_comm);
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

        std::cout << "\n=== Final Performance Analysis ===\n";
        std::cout << "Total execution time: " << total_timer.get_elapsed() << " seconds\n";
        std::cout << "Number of Monte Carlo steps: " << num_steps << "\n";
    }

    // Ensure synchronization before printing performance stats
    MPI_Barrier(cart_comm);

    // Print performance statistics with explicit elapsed time
    perf.print_stats(mc_timer.get_elapsed(), cart_comm);

    // Add another barrier after printing
    MPI_Barrier(cart_comm);

    // Cleanup
    CUDA_CHECK(cudaFree(d_spins));
    CUDA_CHECK(cudaFree(d_rand_nums));
    CUDA_CHECK(cudaFree(d_local_energy));
    CUDA_CHECK(cudaFree(d_local_magnet));

    if (d_states != nullptr) {
        CUDA_CHECK(cudaFree(d_states));
    }

    MPI_Finalize();
    return 0;
}
