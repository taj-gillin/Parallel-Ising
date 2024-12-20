import matplotlib.pyplot as plt
import numpy as np

# System specifications
PEAK_MEMORY_BW = 1638.4  # GB/s (MI250X HBM2e bandwidth)
PEAK_FLOPS = 47900  # GFLOPS (MI250X peak performance)
# PCIE_BW = 64.0  # GB/s (PCIe 4.0 x16)

# CPU specifications
CPU_MEMORY_BW = 204.8  # GB/s (AMD EPYC 7V13 theoretical memory bandwidth)
CPU_PEAK_FLOPS = 2450.0  # GFLOPS (based on CPU max frequency)

# Measured performance from outputs
# GPU Implementation
GPU_GFLOPS = 19.9292
GPU_MEM_BW = 42.5155
GPU_TRANSFER_BW = 8.85984
GPU_AI = 0.46875

# Add optimized GPU implementation results
GPU_OPT_GFLOPS = 139.731
GPU_OPT_MEM_BW = 298.093
GPU_OPT_TRANSFER_BW = 12.4208
GPU_OPT_AI = 0.46875

# RedBlack Implementation (from job/108116.out)
RB_FLOPS = 3932160000  # Total FLOPS from output
RB_BYTES = 8388608000  # Total Bytes from output
RB_TIME = 36.2112  # Monte Carlo Steps Time
RB_GFLOPS = RB_FLOPS / (RB_TIME * 1e9)
RB_BW = RB_BYTES / (RB_TIME * 1e9)
RB_AI = RB_FLOPS / RB_BYTES

# MPI+RedBlack Implementation (from job/108266.out)
MPIRB_FLOPS = 6029312000  # Total FLOPS from output
MPIRB_BYTES = 15728640000  # Total Memory Bytes
MPIRB_COMM_BYTES = 443904000  # Total Communication Bytes
MPIRB_GFLOPS = 0.858179  # From output
MPIRB_BW = 2.23873  # GB/s from output
MPIRB_COMM_BW = 0.0631829  # GB/s from output
MPIRB_AI = 0.383333  # From output

def plot_roofline():
    # Create arithmetic intensity range
    ai_range = np.logspace(-3, 4, 1000)
    
    # Calculate roofline components
    gpu_memory_bound = PEAK_MEMORY_BW * ai_range
    gpu_compute_bound = np.full_like(ai_range, PEAK_FLOPS)
    # pcie_bound = PCIE_BW * ai_range
    cpu_memory_bound = CPU_MEMORY_BW * ai_range
    cpu_compute_bound = np.full_like(ai_range, CPU_PEAK_FLOPS)
    
    # Calculate combined rooflines
    gpu_roofline = np.minimum(gpu_compute_bound, gpu_memory_bound)
    cpu_roofline = np.minimum(cpu_compute_bound, cpu_memory_bound)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot GPU bounds
    plt.loglog(ai_range, gpu_memory_bound, 'b--', alpha=0.5, label='GPU Memory Bandwidth Bound')
    plt.loglog(ai_range, gpu_compute_bound, 'b-', alpha=0.5, label='GPU Compute Bound')
    # plt.loglog(ai_range, pcie_bound, 'g--', alpha=0.5, label='PCIe Bandwidth Bound')
    
    # Plot CPU bounds
    plt.loglog(ai_range, cpu_memory_bound, 'r--', alpha=0.5, label='CPU Memory Bandwidth Bound')
    plt.loglog(ai_range, cpu_compute_bound, 'r-', alpha=0.5, label='CPU Compute Bound')
    
    # Plot measured points (without labels)
    plt.plot(GPU_AI, GPU_GFLOPS, 'bo', markersize=10)
    plt.plot(GPU_OPT_AI, GPU_OPT_GFLOPS, 'mo', markersize=10)
    plt.plot(RB_AI, RB_GFLOPS, 'ro', markersize=10)
    plt.plot(MPIRB_AI, MPIRB_GFLOPS, 'go', markersize=10)
    
    # Add annotations
    plt.annotate(f'GPU\n{GPU_GFLOPS:.2f} GFLOPS\nAI: {GPU_AI:.3f}',
                xy=(GPU_AI, GPU_GFLOPS),
                xytext=(GPU_AI*2, GPU_GFLOPS*2),
                arrowprops=dict(facecolor='blue', shrink=0.05))
    
    plt.annotate(f'GPU (Optimized)\n{GPU_OPT_GFLOPS:.2f} GFLOPS\nAI: {GPU_OPT_AI:.3f}',
                xy=(GPU_OPT_AI, GPU_OPT_GFLOPS),
                xytext=(GPU_OPT_AI*2, GPU_OPT_GFLOPS*2),
                arrowprops=dict(facecolor='magenta', shrink=0.05))
    
    plt.annotate(f'RedBlack\n{RB_GFLOPS:.2f} GFLOPS\nAI: {RB_AI:.3f}',
                xy=(RB_AI, RB_GFLOPS),
                xytext=(RB_AI*2, RB_GFLOPS*2),
                arrowprops=dict(facecolor='red', shrink=0.05))
    
    plt.annotate(f'MPI+RB\n{MPIRB_GFLOPS:.2f} GFLOPS\nAI: {MPIRB_AI:.3f}',
                xy=(MPIRB_AI, MPIRB_GFLOPS),
                xytext=(MPIRB_AI*2, MPIRB_GFLOPS*2),
                arrowprops=dict(facecolor='green', shrink=0.05))
    
    # Calculate ridge points (for analysis only, not plotting)
    gpu_ridge_point = PEAK_FLOPS / PEAK_MEMORY_BW
    cpu_ridge_point = CPU_PEAK_FLOPS / CPU_MEMORY_BW
    
    # Customize plot
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Arithmetic Intensity (FLOPS/byte)', fontsize=12)
    plt.ylabel('Performance (GFLOPS)', fontsize=12)
    plt.title('Roofline Model Analysis - Implementation Comparison', fontsize=14)
    plt.legend(fontsize=10, loc='upper left')
    
    # Set axis limits
    plt.xlim(0.1, 100)
    plt.ylim(0.01, PEAK_FLOPS * 1.2)
    
    # Save plot
    plt.savefig('roofline_comparison.png', dpi=300, bbox_inches='tight')
    
    # Print analysis
    print("\nPerformance Analysis:")
    print("\nGPU Implementation:")
    print(f"Peak Performance: {PEAK_FLOPS:.1f} GFLOPS")
    print(f"Peak Memory Bandwidth: {PEAK_MEMORY_BW:.1f} GB/s")
    print(f"Ridge Point: {gpu_ridge_point:.2f} FLOPS/byte")
    print(f"Measured GFLOPS: {GPU_GFLOPS:.2f}")
    print(f"Memory Bandwidth: {GPU_MEM_BW:.2f} GB/s")
    print(f"PCIe Bandwidth: {GPU_TRANSFER_BW:.2f} GB/s")
    print(f"Arithmetic Intensity: {GPU_AI:.3f} FLOPS/byte")
    
    print("\nRedBlack Implementation:")
    print(f"Peak Performance: {CPU_PEAK_FLOPS:.1f} GFLOPS")
    print(f"Peak Memory Bandwidth: {CPU_MEMORY_BW:.1f} GB/s")
    print(f"Ridge Point: {cpu_ridge_point:.2f} FLOPS/byte")
    print(f"Measured GFLOPS: {RB_GFLOPS:.2f}")
    print(f"Memory Bandwidth: {RB_BW:.2f} GB/s")
    print(f"Arithmetic Intensity: {RB_AI:.3f} FLOPS/byte")
    
    print("\nMPI+RedBlack Implementation:")
    print(f"Measured GFLOPS: {MPIRB_GFLOPS:.2f}")
    print(f"Memory Bandwidth: {MPIRB_BW:.2f} GB/s")
    print(f"Arithmetic Intensity: {MPIRB_AI:.3f} FLOPS/byte")
    
    # Calculate efficiencies
    gpu_peak_at_ai = min(PEAK_FLOPS, PEAK_MEMORY_BW * GPU_AI)
    gpu_efficiency = (GPU_GFLOPS / gpu_peak_at_ai) * 100
    
    rb_peak_at_ai = min(CPU_PEAK_FLOPS, CPU_MEMORY_BW * RB_AI)
    rb_efficiency = (RB_GFLOPS / rb_peak_at_ai) * 100
    
    mpirb_peak_at_ai = min(CPU_PEAK_FLOPS, CPU_MEMORY_BW * MPIRB_AI)
    mpirb_efficiency = (MPIRB_GFLOPS / mpirb_peak_at_ai) * 100
    
    print("\nEfficiencies:")
    print(f"GPU Efficiency: {gpu_efficiency:.2f}% of peak at AI={GPU_AI:.3f}")
    print(f"RedBlack Efficiency: {rb_efficiency:.2f}% of peak at AI={RB_AI:.3f}")
    print(f"MPI+RedBlack Efficiency: {mpirb_efficiency:.2f}% of peak at AI={MPIRB_AI:.3f}")
    
    # Print bottleneck analysis
    print("\nBottleneck Analysis:")
    print("GPU Implementation:", "Memory Bound" if GPU_AI < gpu_ridge_point else "Compute Bound")
    print("RedBlack Implementation:", "Memory Bound" if RB_AI < cpu_ridge_point else "Compute Bound")
    print("MPI+RedBlack Implementation:", "Memory Bound" if MPIRB_AI < cpu_ridge_point else "Compute Bound")
    
    # Print optimized GPU implementation results
    print("\nOptimized GPU Implementation:")
    print(f"Measured GFLOPS: {GPU_OPT_GFLOPS:.2f}")
    print(f"Memory Bandwidth: {GPU_OPT_MEM_BW:.2f} GB/s")
    print(f"PCIe Bandwidth: {GPU_OPT_TRANSFER_BW:.2f} GB/s")
    print(f"Arithmetic Intensity: {GPU_OPT_AI:.3f} FLOPS/byte")
    
    # Calculate optimized GPU efficiency
    gpu_opt_peak_at_ai = min(PEAK_FLOPS, PEAK_MEMORY_BW * GPU_OPT_AI)
    gpu_opt_efficiency = (GPU_OPT_GFLOPS / gpu_opt_peak_at_ai) * 100
    
    print(f"Optimized GPU Efficiency: {gpu_opt_efficiency:.2f}% of peak at AI={GPU_OPT_AI:.3f}")
    
    # Print optimized GPU bottleneck analysis
    print("Optimized GPU Implementation:", "Memory Bound" if GPU_OPT_AI < gpu_ridge_point else "Compute Bound")

if __name__ == "__main__":
    plot_roofline()