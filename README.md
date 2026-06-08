# hpc_lecture

Student ID: 25M38090

## Running the Final Report

To run the final report, execute:
```bash
nvcc -O3 -std=c++17 -arch=sm_90 -Xptxas=-v 13_tensorcore_autotune_wmma_dyn.cu -lcublas -o tc_dyn
```

This code automatically tests different configurations and selects the optimal one.
The `Best custom variant` reported at the end represents the best-performing custom configuration.

### Example Test Run Result (from one of the test run)

```text
cuBLAS reference             359295.41 Gflops  time 0.001913 s
Best custom variant: 128x128x64 p8 w2x4, 200030.34 Gflops, time 0.003435 s
mean_abs_error_vs_cuBLAS: 3.98082213e-03
relative_L1_error_vs_cuBLAS: 3.88796496e-06
```


|          | Topic                                | Sample code               |
| -------- | ------------------------------------ | ------------------------- |
| Class 1  | Introduction to parallel programming | 01_introduction           |
| Class 2  | Shared memory parallelization        | 02_openmp                 |
| Class 3  | Distributed memory parallelization   | 03_mpi                    |
| Class 4  | SIMD parallelization                 | 04_simd                   |
| Class 5  | GPU programming 1                    | 05_openacc                |
| Class 6  | GPU programming 2                    | 06_cuda                   |
| Class 7  | Cache blocking                       | 07_cache                  |
| Class 8  | High Performance Python              | 08_python                 |
| Class 9  | I/O libraries                        | 09_io                     |
| Class 10 | Parallel debugger                    | 10_debugger               |
| Class 11 | Parallel profiler                    | 11_profiler               |
| Class 12 | Containers                           | 12_container              |
| Class 13 | Scientific computing                 | 13_scientific             |
| Class 14 | Deep Learning                        | 14_pytorch                |
