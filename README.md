# 3D Ising Model with Parallel Implementations

This project implements a 3D Ising Model using various parallel computing approaches, including MPI for distributed memory and GPU acceleration. The implementation focuses on performance optimization and scalability for large-scale simulations.

## Overview

The Ising Model provides a simplified yet effective framework to study ferromagnetism and phase transitions in statistical mechanics. First proposed by Wilhelm Lenz in 1920 and solved by Ernst Ising in 1D, it has become one of the most important models in statistical physics. This implementation focuses on a 3D Ising Model (3 spatial dimensions) and its application to high-dimensional problems.

### Key Features

- 3D Ising Model implementation with periodic boundary conditions
- Multiple parallel implementations:
  - Serial baseline
  - MPI with domain decomposition
  - GPU-accelerated version
  - Hybrid MPI+GPU implementation
- Red-black update scheme to prevent pattern formation
- Advanced optimizations including:
  - GPU-aware MPI
  - Shared memory optimizations
  - Precomputed random numbers using cuRAND
- Interactive web visualization
- Comprehensive performance analysis tools

## Theory

The Ising Model represents spins on a lattice, each taking a value of +1 or -1. The system's Hamiltonian is defined as:

```
H = -J Σ(si * sj) - h Σ(si)
```

where:
- J is the interaction strength
- h is the external magnetic field
- si denotes the spin at site i
- The first sum runs over nearest neighbors only

In the 3D implementation, each lattice site has six nearest neighbors (±x, ±y, ±z directions).

## Performance

Performance comparison across different implementations (for 1000 steps):

| Implementation | Runtime (seconds) |
|----------------|------------------|
| Serial | 36.2 |
| MPI (1 rank) | 35.6 |
| MPI (8 ranks) | 5.8 |
| GPU (basic) | 0.95 |
| GPU (optimized) | 0.28 |
| GPU optimized (256³) | 5.5 |

Key performance highlights:
- Near-linear scaling with MPI implementation
- 129x speedup with optimized GPU implementation over serial
- Excellent scaling behavior: 64x problem size increase (64³ to 256³) results in only 19.6x runtime increase

## Requirements

- CUDA-capable GPU (tested on AMD MI250X)
- MPI implementation (OpenMPI recommended)
- C++ compiler with C++11 support
- CMake 3.10 or higher
- cuRAND library
- Slurm

## Building

In the build directory of the implementation you want to build, run:
```bash
make 3d
```

## Usage

### Basic Run

From the root directory, run:
```bash
sbatch launch/<implementation>/3d.launcher
```

## Visualization

An interactive web visualization is available at: https://tajgillin.neocities.org/redblack/rb

The visualization allows real-time observation of:
- Spin configurations
- Energy evolution
- Magnetization changes
- Phase transitions

## Implementation Details

### Domain Decomposition

The 3D lattice is divided into smaller sub-domains, each handled by a separate MPI process. The global L×L×L lattice is partitioned into Px×Py×Pz sub-domains, where Px·Py·Pz = P (total number of processes).

### Optimization Strategies

1. **Precomputed Random Numbers**
   - Uses cuRAND for GPU-based generation
   - Reduced random number generation overhead from 60% to <5%

2. **Shared Memory**
   - Block-based approach for lattice updates
   - Two-level reduction scheme for global calculations
   - Significant reduction in global memory traffic

3. **GPU-Aware MPI**
   - Direct GPU-to-GPU communication
   - Eliminates CPU staging buffers
   - Automatic fallback to staged transfers when needed

## Future Enhancements

Planned improvements include:
- Pinned memory implementation for improved GPU transfers
- Asynchronous CUDA streams for computation-communication overlap
- Temperature scaling studies for phase transition analysis
- Additional optimization strategies for memory access patterns

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to Professor Grinberg for guidance and support throughout this project.

## Citation

If you use this code in your research, please cite:
```
@misc{gillin2024ising,
  author = {Gillin, Taj},
  title = {3D Ising Model with Parallel Implementations},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/username/repo}
}
```
