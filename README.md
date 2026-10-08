# Parallel Matrix Multiplication: Cannon, Fox and SUMMA on MPI and CUDA

A comparative study of three classic distributed matrix multiplication algorithms (**Cannon's algorithm**, **Fox's algorithm** and **SUMMA**), each implemented as MPI-only, CUDA-only and hybrid MPI+CUDA variants, and benchmarked under strong and weak scaling on RPI's **AiMOS** supercomputer (IBM POWER9 + NVIDIA V100).

Final project for *Parallel Programming and Computing*, Rensselaer Polytechnic Institute, Spring 2026.
Full write-up: [`PCP_Group_Project_Report.pdf`](PCP_Group_Project_Report.pdf)

## Highlights

- **1,478 GFLOPS** from the single-GPU CUDA SUMMA variant on a 4096 × 4096 matrix (0.093 s).
- **Near-ideal GPU strong scaling** for Fox MPI+CUDA on 2¹⁵ × 2¹⁵ matrices: 47.26 s → 12.11 s → 3.73 s at 1 → 4 → 16 ranks (about 4× per 4× ranks; **12.7×** overall).
- **Near-linear CPU strong scaling** for all three MPI-only implementations, up to 64 ranks (Cannon: about 15 s → 1.5 s from 4 to 64 ranks on 2¹² × 2¹²).
- A clear **CPU/GPU crossover**: distributing work across GPUs only pays off once each dimension reaches about 2¹³–2¹⁴. Below that, a single GPU wins because MPI communication and host↔device copies dominate.

## Algorithms

All three algorithms arrange `p` MPI ranks in a √p × √p 2D Cartesian grid (`MPI_Cart_create`, periodic) and give each rank one block of A, B and C. Blocks are distributed with `MPI_Scatterv` using a custom `MPI_Type_vector` block type and collected with `MPI_Gatherv`.

| Algorithm | Communication pattern | Notes |
|---|---|---|
| **Cannon** (1969) | Initial skew (A left by row index, B up by column index), then √p rounds of multiply → shift A left, B up | Point-to-point neighbor shifts; constant memory per rank regardless of `p` |
| **Fox** (1988) | √p rounds of: broadcast a block of A along each processor row → multiply → shift B up | Row-communicator broadcasts map well to the AiMOS interconnect |
| **SUMMA** (van de Geijn & Watts, 1997) | For each k: broadcast A[i,k] along row i and B[k,j] along column j, then accumulate C[i,j] += A[i,k]·B[k,j] | No skew phase; simplest and most general, but the most collective communication (O(p log p) broadcasts) |

In the hybrid variants each MPI rank is bound to its own GPU (`cudaSetDevice`). The local block multiply runs as a CUDA kernel while MPI handles all inter-rank data movement.

## Results

All timings use the POWER9 time-base register (512 MHz) and cover the full computation, including communication.

### SUMMA: 4096 × 4096

| Implementation | Ranks | Time (s) | GFLOPS | vs. CUDA-only |
|---|---:|---:|---:|---:|
| CUDA-only | 1 GPU | 0.093 | 1,478 | 1.0× |
| MPI+CUDA | 4 | 0.597 | 230 | 0.16× |
| MPI+CUDA | 16 | 1.653 | 83 | 0.06× |
| MPI-only | 16 | 11.478 | 12 | 0.008× |

From 4 to 16 hybrid ranks, communication latency grows about 30× while per-rank computation shrinks only 4×, which produces *negative* scaling. On CPU, where computation is slow relative to communication, MPI-only SUMMA scales almost perfectly linearly.

### Fox

| Variant | Matrix | Ranks | Time (s) |
|---|---|---:|---:|
| MPI-only, strong | 2¹² × 2¹² | 4 / 16 | 10.13 / 3.57 |
| MPI-only, weak | grows with ranks | 4 → 64 | 1.23 → 5.78 |
| MPI+CUDA | 2¹² × 2¹² | 1 GPU | 0.116 *(vs. 1.48 s for 64 MPI-only ranks)* |
| MPI+CUDA, strong | 2¹⁵ × 2¹⁵ | 1 / 4 / 16 | 47.26 / 12.11 / 3.73 |
| MPI+CUDA, weak | grows with ranks | 1 → 16 | 0.79 → 3.73 |

### Cannon

| Variant | Matrix | Ranks | Time (s, approx.) |
|---|---|---:|---:|
| MPI-only, strong | 2¹² × 2¹² | 4 / 16 / 64 | 15 / 5 / 1.5 |
| MPI-only, weak | 2¹² → 2¹⁴ | 4 → 64 | within about 1 s across configs |
| MPI+CUDA, strong | fixed | 4 / 16 / 64 | 18 / 7 / 3 |
| MPI+CUDA, weak | grows with ranks | 4 → 64 | 5 – 6.5 |

Plots for every configuration are in the [report](PCP_Group_Project_Report.pdf) (Figures 1–9).

### Takeaways

1. **If it fits on one GPU, use one GPU.** CUDA-only beats every distributed configuration at small and medium sizes.
2. **Hybrid MPI+CUDA only pays off when blocks are large enough** to keep each GPU busy relative to transfer and communication costs, about 2¹³–2¹⁴ per dimension in these experiments.
3. **MPI-only scaling is predictable and near-linear**, because CPU compute dwarfs communication cost.
4. **With GPUs, the algorithm matters less than block size.** Among the algorithms, SUMMA's frequent broadcasts hurt it most once computation is fast.

## Repository layout

```
src/
├── Cannon/
│   ├── Cannon_MPI.c          # MPI-only
│   ├── Cannon_CUDA.cu        # CUDA-only (tiled shared-memory kernel)
│   ├── MPI_cannon.c          # MPI+CUDA host side  ┐ hybrid, linked together
│   ├── Cuda_Cannon.cu        # MPI+CUDA block kernel ┘
│   ├── Cannon_Hybrid.cu      # alternate single-file hybrid
│   └── Cannon.sh             # run script
├── Fox/
│   ├── MPI_only/fox_mpi.c
│   └── MPI_and_CUDA/         # fox_mpi_cuda.c + fox_mpi_cuda.cu + Makefile
└── SUMMA/
    ├── summa_mpi.c           # MPI-only
    ├── summa_cuda.cu         # CUDA-only (single-GPU baseline)
    ├── summa_mpi_cuda.c      # MPI+CUDA host side
    ├── summa_mpi_cuda_kernel.cu
    ├── Makefile_summa_cuda
    └── run_summa_all.sh      # Slurm job: full strong/weak study → summa_results.csv
PCP_Group_Project_Report.pdf
```

## Building and running

The code targets AiMOS (POWER9, V100 `sm_70`, Spectrum MPI, CUDA 11.2). Load modules first:

```bash
module load xl_r spectrum-mpi cuda/11.2
```

The rank count must be a **perfect square** (1, 4, 16, 64), and hybrid runs need **one GPU per rank**. Matrices are filled with constants, so each run checks its own result (every element of C should equal the expected constant).

### Cannon (argument: matrix dimension `N`)

```bash
# MPI-only
mpicc -O3 Cannon_MPI.c -o Cannon_MPI -lm
mpirun -np 16 ./Cannon_MPI 4096

# CUDA-only
nvcc -O3 -arch=sm_70 Cannon_CUDA.cu -o Cannon_CUDA
./Cannon_CUDA 4096

# MPI+CUDA
mpicc -c MPI_cannon.c && nvcc -c Cuda_Cannon.cu
mpixlc -O3 Cuda_Cannon.o MPI_cannon.o -o cannon \
  -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++
mpirun -np 4 ./cannon 4096
```

### Fox (argument: **exponent** `k`, multiplies 2ᵏ × 2ᵏ matrices)

```bash
# MPI-only (ranks: 1, 4, 16, 64)
mpicc -O3 fox_mpi.c -o fox_mpi
mpirun -np 16 ./fox_mpi 12

# MPI+CUDA (ranks: 1, 4, 16)
cd src/Fox/MPI_and_CUDA && make        # builds fox_mpi_cuda-exe
mpirun -np 4 ./fox_mpi_cuda-exe 15
```

### SUMMA (argument: matrix dimension `N`, must be divisible by √ranks)

```bash
mpixlc -O3 summa_mpi.c -o summa_mpi -lm
nvcc -O3 -arch=sm_70 summa_cuda.cu -o summa_cuda
make -f Makefile_summa_cuda             # builds summa_mpi_cuda

mpirun -np 16 ./summa_mpi 4096
./summa_cuda 4096
mpirun -np 4 ./summa_mpi_cuda 4096

# or run the whole study as one Slurm job (1 node, 4 GPUs):
sbatch run_summa_all.sh                 # writes summa_results.csv
```

SUMMA prints time, GFLOPS and a `C[0,0]` correctness check.

## Known limitations and future work

- **Naive kernels.** Local block multiplies use hand-written CUDA kernels, not cuBLAS. Swapping in `cublasDgemm` is the obvious next step.
- **No compute/communication overlap.** Each step copies host↔device and communicates in sequence. CUDA streams with non-blocking MPI (or CUDA-aware MPI / GPUDirect RDMA / NCCL) would hide much of the latency that limits hybrid scaling.
- **Cannon CUDA timing.** `Cannon_CUDA.cu` stops its timer right after the kernel launch, without a `cudaDeviceSynchronize()`, so its reported microsecond times measure launch overhead rather than the full multiply.
- **Cannon hybrid kernel launch.** `Cuda_Cannon.cu` launches a single 16 × 16 thread block per local multiply, so it only covers local blocks up to 16 × 16.
- **Scale.** Next steps are matrices of 2¹⁶ and larger, plus weak scaling to 256+ ranks, to pin down the distributed-GPU crossover more precisely.

## Team

| Member | Primary implementation |
|---|---|
| Alexandre Gosselin Neves ([@apgneves](https://github.com/apgneves)) | SUMMA (MPI, CUDA, MPI+CUDA) and benchmark automation |
| Logan Acanfrio | Fox's algorithm (MPI, MPI+CUDA) |
| Julian Blanco | Cannon's algorithm (MPI, CUDA, MPI+CUDA) |

## References

1. L. E. Cannon, *A cellular computer to implement the Kalman filter algorithm*, Ph.D. thesis, Montana State University, 1969.
2. G. C. Fox et al., *Solving Problems on Concurrent Processors*, Prentice Hall, 1988.
3. R. A. van de Geijn and J. Watts, "SUMMA: Scalable Universal Matrix Multiplication Algorithm," *Concurrency: Practice and Experience*, 9(4):255–274, 1997.
4. A. Grama, A. Gupta, G. Karypis, V. Kumar, *Introduction to Parallel Computing*, 2nd ed., Addison-Wesley, 2003.
5. V. Volkov and J. W. Demmel, "Benchmarking GPUs to Tune Dense Linear Algebra," SC '08, 2008.

## License

[GNU GPL v3.0](LICENSE)
