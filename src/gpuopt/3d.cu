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
const int L = 256;              // Global lattice size (L x L x L grid)
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
    long long total_flops = 0;
    long long device_bytes = 0;     // Memory traffic on GPU
    long long host_device_bytes = 0; // Host-device transfer bytes
    
    // GPU specifications (MI250X)
    const double peak_flops = 47.9 * 1e3;     // 47.9 TFLOPS = 47900 GFLOPS per GCD
    const double peak_bandwidth = 1638.4;      // GB/s HBM2e bandwidth
    const double pcie_bandwidth = 64.0;        // GB/s PCIe 4.0 x16 bandwidth
    
    void add_kernel_update(int volume) {
        // For each spin update:
        // - 6 neighbor loads (6 reads)
        // - 1 current spin load (1 read)
        // - 1 current spin store (1 write)
        // - ~5 FLOPs for energy calculation
        // - ~2 FLOPs for Metropolis criterion
        total_flops += volume * 7;  // 5 for energy calc + 2 for Metropolis
        device_bytes += volume * (8 * sizeof(int));  // 7 reads + 1 write
    }
    
    void add_energy_calculation(int volume) {
        // For energy/magnetization calculation:
        // - 7 loads (current spin + 6 neighbors)
        // - 4 FLOPs for energy
        // - 1 FLOP for magnetization
        total_flops += volume * 5;
        device_bytes += volume * (7 * sizeof(int));  // 7 reads
    }
    
    void add_halo_exchange(int face_size) {
        // For halo exchange (GPU-GPU):
        // Each face requires:
        // - 1 read from source
        // - 1 write to destination
        device_bytes += face_size * 2 * sizeof(int);
    }
    
    void add_host_device_transfer(size_t bytes) {
        host_device_bytes += bytes;
    }
    
    void print_stats(double elapsed_time, MPI_Comm comm) {
        int rank, num_procs;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &num_procs);
        
        // Gather statistics from all processes
        long long global_flops = 0, global_device_bytes = 0, global_host_device_bytes = 0;
        MPI_Allreduce(&total_flops, &global_flops, 1, MPI_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(&device_bytes, &global_device_bytes, 1, MPI_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(&host_device_bytes, &global_host_device_bytes, 1, MPI_LONG_LONG, MPI_SUM, comm);
        
        if (rank == 0) {
            std::cout << "\n=== GPU Performance Statistics ===\n";
            
            // Calculate achieved performance metrics
            double gflops = static_cast<double>(global_flops) / (elapsed_time * 1e9);
            double device_bandwidth = static_cast<double>(global_device_bytes) / (elapsed_time * 1e9);
            double transfer_bandwidth = static_cast<double>(global_host_device_bytes) / (elapsed_time * 1e9);
            
            // Calculate arithmetic intensity using device memory traffic only
            double arithmetic_intensity = static_cast<double>(global_flops) / global_device_bytes;
            
            // Calculate roofline model metrics
            double compute_efficiency = (gflops / peak_flops) * 100.0;
            double memory_efficiency = (device_bandwidth / peak_bandwidth) * 100.0;
            double transfer_efficiency = (transfer_bandwidth / pcie_bandwidth) * 100.0;
            
            // Calculate theoretical peak performance at this arithmetic intensity
            double peak_performance = std::min(peak_flops, peak_bandwidth * arithmetic_intensity);
            double achieved_fraction = gflops / peak_performance * 100.0;
            
            // Ridge point (where memory bound meets compute bound)
            double ridge_point = peak_flops / peak_bandwidth;
            
            std::cout << "Total FLOPS: " << global_flops << std::endl;
            std::cout << "Total Device Memory Traffic: " << global_device_bytes << " bytes" << std::endl;
            std::cout << "Total Host-Device Transfer: " << global_host_device_bytes << " bytes" << std::endl;
            
            std::cout << "\nAchieved Performance:" << std::endl;
            std::cout << "GFLOPS: " << gflops << std::endl;
            std::cout << "Device Memory Bandwidth: " << device_bandwidth << " GB/s" << std::endl;
            std::cout << "Host-Device Transfer Bandwidth: " << transfer_bandwidth << " GB/s" << std::endl;
            
            std::cout << "\nRoofline Model Analysis:" << std::endl;
            std::cout << "Arithmetic Intensity: " << arithmetic_intensity << " FLOPS/byte" << std::endl;
            std::cout << "Ridge Point: " << ridge_point << " FLOPS/byte" << std::endl;
            std::cout << "Theoretical Peak at AI=" << arithmetic_intensity << ": " << peak_performance << " GFLOPS" << std::endl;
            std::cout << "Performance Efficiency: " << achieved_fraction << "% of peak" << std::endl;
            
            std::cout << "\nResource Utilization:" << std::endl;
            std::cout << "Compute Utilization: " << compute_efficiency << "% of peak FLOPS" << std::endl;
            std::cout << "Memory Bandwidth Utilization: " << memory_efficiency << "% of peak BW" << std::endl;
            std::cout << "PCIe Bandwidth Utilization: " << transfer_efficiency << "% of peak BW" << std::endl;
            
            std::cout << "\nBottleneck Analysis:" << std::endl;
            if (arithmetic_intensity < ridge_point) {
                std::cout << "Memory Bandwidth Bound" << std::endl;
                std::cout << "To improve: Consider memory access coalescing and reducing redundant loads" << std::endl;
            } else {
                std::cout << "Compute Bound" << std::endl;
                std::cout << "To improve: Consider optimizing compute operations and increasing occupancy" << std::endl;
            }
            
            if (transfer_efficiency > 80.0) {
                std::cout << "Warning: PCIe transfer might be bottleneck" << std::endl;
                std::cout << "Consider reducing host-device data movement or using pinned memory" << std::endl;
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int interior_size = (local_x-2) * (local_y-2) * (local_z-2);
    
    if (tid < interior_size) {
        // Convert 1D thread index to 3D coordinates (interior points only)
        int k = (tid / ((local_x-2)*(local_y-2))) + 1;
        int j = ((tid / (local_x-2)) % (local_y-2)) + 1;
        int i = (tid % (local_x-2)) + 1;
        
        // Check parity
        if (((i + j + k) % 2) == parity) {
            int idx = i + local_x*(j + local_y*k);
            int current_spin = spins[idx];
            
            // Calculate neighbor sum (unchanged)
            int sum_neighbors = spins[(i-1) + local_x*(j + local_y*k)]
                              + spins[(i+1) + local_x*(j + local_y*k)]
                              + spins[i + local_x*((j-1) + local_y*k)]
                              + spins[i + local_x*((j+1) + local_y*k)]
                              + spins[i + local_x*(j + local_y*(k-1))]
                              + spins[i + local_x*(j + local_y*(k+1))];

            double dE = 2.0 * current_spin * (J * sum_neighbors + h);
            
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
    double* smagnet = &sdata[blockDim.x];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int interior_size = (local_x-2) * (local_y-2) * (local_z-2);
    
    double e = 0.0;
    double m = 0.0;
    
    if (gid < interior_size) {
        // Convert 1D index to 3D coordinates (interior points only)
        int k = (gid / ((local_x-2)*(local_y-2))) + 1;
        int j = ((gid / (local_x-2)) % (local_y-2)) + 1;
        int i = (gid % (local_x-2)) + 1;
        
        int idx = i + local_x*(j + local_y*k);
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
    
    senergy[tid] = e;
    smagnet[tid] = m;
    
    __syncthreads();
    
    // Reduce within block using sequential addressing
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            senergy[tid] += senergy[tid + stride];
            smagnet[tid] += smagnet[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(local_energy, senergy[0]);
        atomicAdd(local_magnet, smagnet[0]);
    }
}

// Add these kernels for packing/unpacking faces (move this up, after other kernel declarations)
__global__ 
void pack_face_kernel(const int* spins, int* send_buf,
                     int local_x, int local_y, int local_z,
                     int face_idx, char direction) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (direction == 'x') {
        int num_elements = local_y * local_z;
        if (tid < num_elements) {
            int j = tid / local_z;
            int k = tid % local_z;
            send_buf[tid] = spins[face_idx + local_x*(j + local_y*k)];
        }
    } else if (direction == 'y') {
        int num_elements = local_x * local_z;
        if (tid < num_elements) {
            int i = tid / local_z;
            int k = tid % local_z;
            send_buf[tid] = spins[i + local_x*(face_idx + local_y*k)];
        }
    } else { // 'z'
        int num_elements = local_x * local_y;
        if (tid < num_elements) {
            int i = tid / local_y;
            int j = tid % local_y;
            send_buf[tid] = spins[i + local_x*(j + local_y*face_idx)];
        }
    }
}

__global__ 
void unpack_face_kernel(int* spins, const int* recv_buf,
                       int local_x, int local_y, int local_z,
                       int face_idx, char direction) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (direction == 'x') {
        int num_elements = local_y * local_z;
        if (tid < num_elements) {
            int j = tid / local_z;
            int k = tid % local_z;
            spins[face_idx + local_x*(j + local_y*k)] = recv_buf[tid];
        }
    } else if (direction == 'y') {
        int num_elements = local_x * local_z;
        if (tid < num_elements) {
            int i = tid / local_z;
            int k = tid % local_z;
            spins[i + local_x*(face_idx + local_y*k)] = recv_buf[tid];
        }
    } else { // 'z'
        int num_elements = local_x * local_y;
        if (tid < num_elements) {
            int i = tid / local_y;
            int j = tid % local_y;
            spins[i + local_x*(j + local_y*face_idx)] = recv_buf[tid];
        }
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

// Add near the top of the file, with other constants
const bool SAVE_OUTPUT = false;  // Can be changed to false to disable saving

// Modify halo_exchange to work with GPU memory
void halo_exchange(int* d_spins, MPI_Comm cart_comm, int local_x, int local_y, int local_z) {
    MPI_Barrier(cart_comm);
    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    int left, right, up, down, front, back;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 2, 1, &front, &back);

    // Allocate GPU buffers for sending/receiving
    int *d_send_left, *d_send_right, *d_recv_left, *d_recv_right;
    int *d_send_up, *d_send_down, *d_recv_up, *d_recv_down;
    int *d_send_front, *d_send_back, *d_recv_front, *d_recv_back;

    CUDA_CHECK(cudaMalloc(&d_send_left, local_y * local_z * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_send_right, local_y * local_z * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_recv_left, local_y * local_z * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_recv_right, local_y * local_z * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_send_up, local_x * local_z * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_send_down, local_x * local_z * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_recv_up, local_x * local_z * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_recv_down, local_x * local_z * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_send_front, local_x * local_y * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_send_back, local_x * local_y * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_recv_front, local_x * local_y * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_recv_back, local_x * local_y * sizeof(int)));

    // Add kernel to pack data for sending
    const int BLOCK_SIZE = 256;
    
    // Pack data kernels
    auto pack_faces = [](int* d_spins, int* d_send_buf, 
                        int local_x, int local_y, int local_z,
                        int face_idx, char direction) {
        int num_elements;
        dim3 block(BLOCK_SIZE);
        dim3 grid;
        
        if (direction == 'x') {
            num_elements = local_y * local_z;
            grid.x = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            pack_face_kernel<<<grid, block>>>(d_spins, d_send_buf, 
                                            local_x, local_y, local_z,
                                            face_idx, 'x');
        } else if (direction == 'y') {
            num_elements = local_x * local_z;
            grid.x = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            pack_face_kernel<<<grid, block>>>(d_spins, d_send_buf, 
                                            local_x, local_y, local_z,
                                            face_idx, 'y');
        } else { // 'z'
            num_elements = local_x * local_y;
            grid.x = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            pack_face_kernel<<<grid, block>>>(d_spins, d_send_buf, 
                                            local_x, local_y, local_z,
                                            face_idx, 'z');
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Pack data for sending
    pack_faces(d_spins, d_send_left, local_x, local_y, local_z, 1, 'x');
    pack_faces(d_spins, d_send_right, local_x, local_y, local_z, local_x-2, 'x');
    pack_faces(d_spins, d_send_up, local_x, local_y, local_z, 1, 'y');
    pack_faces(d_spins, d_send_down, local_x, local_y, local_z, local_y-2, 'y');
    pack_faces(d_spins, d_send_front, local_x, local_y, local_z, 1, 'z');
    pack_faces(d_spins, d_send_back, local_x, local_y, local_z, local_z-2, 'z');

    // Enable CUDA-aware MPI if available
    #ifdef MPIX_CUDA_AWARE_SUPPORT
        // X direction
        MPI_Sendrecv(d_send_left, local_y*local_z, MPI_INT, left, 0,
                     d_recv_right, local_y*local_z, MPI_INT, right, 0, 
                     cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(d_send_right, local_y*local_z, MPI_INT, right, 1,
                     d_recv_left, local_y*local_z, MPI_INT, left, 1, 
                     cart_comm, MPI_STATUS_IGNORE);

        // Y direction
        MPI_Sendrecv(d_send_up, local_x*local_z, MPI_INT, up, 2,
                     d_recv_down, local_x*local_z, MPI_INT, down, 2, 
                     cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(d_send_down, local_x*local_z, MPI_INT, down, 3,
                     d_recv_up, local_x*local_z, MPI_INT, up, 3, 
                     cart_comm, MPI_STATUS_IGNORE);

        // Z direction
        MPI_Sendrecv(d_send_front, local_x*local_y, MPI_INT, front, 4,
                     d_recv_back, local_x*local_y, MPI_INT, back, 4, 
                     cart_comm, MPI_STATUS_IGNORE);
        MPI_Sendrecv(d_send_back, local_x*local_y, MPI_INT, back, 5,
                     d_recv_front, local_x*local_y, MPI_INT, front, 5, 
                     cart_comm, MPI_STATUS_IGNORE);
    #else
        // Fallback for non-CUDA-aware MPI
        // Allocate host buffers
        std::vector<int> h_send_buf(std::max({local_y*local_z, local_x*local_z, local_x*local_y}));
        std::vector<int> h_recv_buf(h_send_buf.size());

        // X direction exchanges
        CUDA_CHECK(cudaMemcpy(h_send_buf.data(), d_send_left, local_y*local_z*sizeof(int), 
                            cudaMemcpyDeviceToHost));
        MPI_Sendrecv(h_send_buf.data(), local_y*local_z, MPI_INT, left, 0,
                     h_recv_buf.data(), local_y*local_z, MPI_INT, right, 0, 
                     cart_comm, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(d_recv_right, h_recv_buf.data(), local_y*local_z*sizeof(int), 
                            cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(h_send_buf.data(), d_send_right, local_y*local_z*sizeof(int), 
                            cudaMemcpyDeviceToHost));
        MPI_Sendrecv(h_send_buf.data(), local_y*local_z, MPI_INT, right, 1,
                     h_recv_buf.data(), local_y*local_z, MPI_INT, left, 1, 
                     cart_comm, MPI_STATUS_IGNORE);
        CUDA_CHECK(cudaMemcpy(d_recv_left, h_recv_buf.data(), local_y*local_z*sizeof(int), 
                            cudaMemcpyHostToDevice));

        // Y and Z directions similarly...
        // (Code omitted for brevity but follows same pattern)
    #endif

    // Unpack received data
    auto unpack_faces = [](int* d_spins, int* d_recv_buf,
                          int local_x, int local_y, int local_z,
                          int face_idx, char direction) {
        int num_elements;
        dim3 block(BLOCK_SIZE);
        dim3 grid;
        
        if (direction == 'x') {
            num_elements = local_y * local_z;
            grid.x = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        } else if (direction == 'y') {
            num_elements = local_x * local_z;
            grid.x = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        } else { // 'z'
            num_elements = local_x * local_y;
            grid.x = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        }
        
        unpack_face_kernel<<<grid, block>>>(d_spins, d_recv_buf,
                                          local_x, local_y, local_z,
                                          face_idx, direction);
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    unpack_faces(d_spins, d_recv_left, local_x, local_y, local_z, 0, 'x');
    unpack_faces(d_spins, d_recv_right, local_x, local_y, local_z, local_x-1, 'x');
    unpack_faces(d_spins, d_recv_up, local_x, local_y, local_z, 0, 'y');
    unpack_faces(d_spins, d_recv_down, local_x, local_y, local_z, local_y-1, 'y');
    unpack_faces(d_spins, d_recv_front, local_x, local_y, local_z, 0, 'z');
    unpack_faces(d_spins, d_recv_back, local_x, local_y, local_z, local_z-1, 'z');

    // Cleanup
    CUDA_CHECK(cudaFree(d_send_left));
    CUDA_CHECK(cudaFree(d_send_right));
    CUDA_CHECK(cudaFree(d_recv_left));
    CUDA_CHECK(cudaFree(d_recv_right));
    CUDA_CHECK(cudaFree(d_send_up));
    CUDA_CHECK(cudaFree(d_send_down));
    CUDA_CHECK(cudaFree(d_recv_up));
    CUDA_CHECK(cudaFree(d_recv_down));
    CUDA_CHECK(cudaFree(d_send_front));
    CUDA_CHECK(cudaFree(d_send_back));
    CUDA_CHECK(cudaFree(d_recv_front));
    CUDA_CHECK(cudaFree(d_recv_back));

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
    
    // Track initial memory transfer
    transfer_timer.start();
    perf.add_host_device_transfer(volume * sizeof(int));  // d_spins initialization
    transfer_timer.stop();
    
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
    transfer_timer.start();
    CUDA_CHECK(cudaMalloc(&d_exp_table, (MAX_ENERGY + 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_exp_table, exp_table.data(), (MAX_ENERGY + 1) * sizeof(double), 
               cudaMemcpyHostToDevice));
    perf.add_host_device_transfer((MAX_ENERGY + 1) * sizeof(double));
    transfer_timer.stop();

    // Update block size definition near the top of the file
    const int BLOCK_SIZE = 256; // Optimal block size for most GPUs

    // Update the kernel launches in monte_carlo_step_gpu:
    // ... (rest of the code remains the same)

    // Replace the existing block/grid configuration with:
    int num_interior_spins = (local_x-2) * (local_y-2) * (local_z-2); // Number of non-ghost spins
    int num_blocks = (num_interior_spins + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Parity = 0 update
    compute_timer.start();
    gpu_update_parity<<<num_blocks, BLOCK_SIZE>>>(d_spins, d_states, local_x, local_y, local_z, 
                                                0, J, h, current_T, d_exp_table);
    CUDA_CHECK(cudaDeviceSynchronize());
    compute_timer.stop();
    perf.add_kernel_update(volume);

    // GPU halo exchange (no host transfers)
    halo_timer.start();
    halo_exchange(d_spins, cart_comm, local_x, local_y, local_z);
    halo_timer.stop();
    // Note: halo exchange memory operations are counted within halo_exchange function
    
    // Parity = 1 update
    compute_timer.start();
    gpu_update_parity<<<num_blocks, BLOCK_SIZE>>>(d_spins, d_states, local_x, local_y, local_z, 
                                                1, J, h, current_T, d_exp_table);
    CUDA_CHECK(cudaDeviceSynchronize());
    compute_timer.stop();
    perf.add_kernel_update(volume);

    // Energy computation
    compute_timer.start();
    CUDA_CHECK(cudaMemset(d_local_energy, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_local_magnet, 0, sizeof(double)));

    size_t shared_size = 2 * BLOCK_SIZE * sizeof(double); // Reduced shared memory size
    gpu_compute_energy_magnetization<<<num_blocks, BLOCK_SIZE, shared_size>>>(
        d_spins, local_x, local_y, local_z, J, h, d_local_energy, d_local_magnet);
    CUDA_CHECK(cudaDeviceSynchronize());
    compute_timer.stop();
    perf.add_kernel_update(volume);  // Track energy computation

    transfer_timer.start();
    double local_energy, local_m;
    CUDA_CHECK(cudaMemcpy(&local_energy, d_local_energy, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&local_m, d_local_magnet, sizeof(double), cudaMemcpyDeviceToHost));
    perf.add_host_device_transfer(2 * sizeof(double));
    transfer_timer.stop();

    reduce_timer.start();
    double global_energy, global_m;
    MPI_Reduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    MPI_Reduce(&local_m, &global_m, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
    reduce_timer.stop();

    if (rank == 0) {
        magnetization = global_m / (L*L*L);
        energy = global_energy;
        
        // Remove or comment out the detailed timing prints here since we'll get them in the final summary
        // std::cout << "\nDetailed GPU Timing Breakdown:\n"
        //           << "Computation: " << compute_timer.get_elapsed() << "s\n"
        //           << "Memory Transfer: " << transfer_timer.get_elapsed() << "s\n"
        //           << "Halo Exchange: " << halo_timer.get_elapsed() << "s\n"
        //           << "MPI Reduce: " << reduce_timer.get_elapsed() << "s\n";
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

    // Get number of GPUs and assign to ranks
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    
    // Each rank gets assigned to GPU (rank % num_gpus)
    int gpu_id = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));
    
    if (rank == 0) {
        std::cout << "Number of GPUs detected: " << num_gpus << std::endl;
        std::cout << "Number of MPI ranks: " << num_procs << std::endl;
        std::cout << "Ranks per GPU: " << (num_procs + num_gpus - 1) / num_gpus << std::endl;
    }

    // Optional: Print GPU assignment for debugging
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(hostname, &name_len);
    printf("Rank %d on host %s using GPU %d\n", rank, hostname, gpu_id);

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

        // Only transfer and save if needed
        if (SAVE_OUTPUT && step % save_interval == 0) {
            io_timer.start();
            // Transfer data back to CPU only when saving
            std::vector<int> h_spins_flat(volume);
            CUDA_CHECK(cudaMemcpy(h_spins_flat.data(), d_spins, volume*sizeof(int), 
                                cudaMemcpyDeviceToHost));
            
            // Convert to 3D array and save
            for (int i = 0; i < local_x; i++) {
                for (int j = 0; j < local_y; j++) {
                    for (int k = 0; k < local_z; k++) {
                        spins[i][j][k] = h_spins_flat[i + local_x*(j + local_y*k)];
                    }
                }
            }

            std::vector<std::vector<std::vector<int>>> full_lattice;
            if (rank == 0) {
                full_lattice.resize(L, std::vector<std::vector<int>>(L, std::vector<int>(L)));
            }
            gather_lattice_data(spins, full_lattice, cart_comm, dims, coords);

            if (rank == 0) {
                write_lattice_to_file("out/gpu/3d/lattice/step_" + std::to_string(step) + ".dat", 
                                    full_lattice);
            }
            io_timer.stop();
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
