#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call) do {                                             \
  cudaError_t e = (call);                                                  \
  if (e != cudaSuccess) {                                                  \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
            cudaGetErrorString(e));                                        \
    std::exit(1);                                                          \
  }                                                                        \
} while (0)

#define CHECK_CUBLAS(call) do {                                           \
  cublasStatus_t s = (call);                                               \
  if (s != CUBLAS_STATUS_SUCCESS) {                                        \
    fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s);   \
    std::exit(1);                                                          \
  }                                                                        \
} while (0)

static inline cudaError_t prefetch_to_device(const void *ptr, size_t bytes, int dev) {
#if CUDART_VERSION >= 13000
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeDevice;
  loc.id = dev;
  return cudaMemPrefetchAsync(ptr, bytes, loc, 0, 0);
#else
  return cudaMemPrefetchAsync(ptr, bytes, dev, 0);
#endif
}

static inline cudaError_t prefetch_to_host(const void *ptr, size_t bytes) {
#if CUDART_VERSION >= 13000
  cudaMemLocation loc{};
  loc.type = cudaMemLocationTypeHost;
  loc.id = 0;
  return cudaMemPrefetchAsync(ptr, bytes, loc, 0, 0);
#else
  return cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, 0);
#endif
}

__global__ void float_to_half_raw_kernel(const float *__restrict__ X,
                                         half *__restrict__ Xh,
                                         int64_t n) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) Xh[i] = __float2half_rn(X[i]);
}

__global__ void pack_B_to_half_rowmajor_kernel(const float *__restrict__ B,
                                               half *__restrict__ Bp,
                                               int K, int N) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t total = int64_t(K) * N;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  for (int64_t t = tid; t < total; t += stride) {
    int k = int(t / N);
    int n = int(t - int64_t(k) * N);
    Bp[t] = __float2half_rn(B[int64_t(n) * K + k]);
  }
}

template<int BM, int BN, int BK, int PAD, int WM, int WN>
__global__ void wmma_gemm_vec_kernel(int M, int N, int K,
                                     const half *__restrict__ A,
                                     const half *__restrict__ Bp,
                                     float *__restrict__ C) {
  static_assert(BM % (16 * WM) == 0, "BM must match warp tile height");
  static_assert(BN % (16 * WN) == 0, "BN must match warp tile width");
  static_assert(BK % 16 == 0, "BK must be a multiple of 16");
  static_assert(PAD % 8 == 0, "shared row stride must stay 16-byte aligned");
  static_assert(BM % 8 == 0 && BN % 8 == 0, "vector copy uses 8 half elements");

  constexpr int WARPS_M = BM / (16 * WM);
  constexpr int WARPS_N = BN / (16 * WN);
  constexpr int WARPS   = WARPS_M * WARPS_N;
  constexpr int THREADS = WARPS * 32;
  static_assert(WARPS >= 1 && WARPS <= 32, "use 1..32 warps per CTA");

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int warp_m = warp_id / WARPS_N;
  const int warp_n = warp_id - warp_m * WARPS_N;

  const int m0 = blockIdx.x * BM;
  const int n0 = blockIdx.y * BN;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  constexpr int STRIDE_A = BM + PAD;
  constexpr int STRIDE_B = BN + PAD;
  half *As = reinterpret_cast<half *>(smem_raw);
  half *Bs = As + BK * STRIDE_A;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM][WN];
#pragma unroll
  for (int r = 0; r < WM; ++r) {
#pragma unroll
    for (int c = 0; c < WN; ++c) {
      wmma::fill_fragment(acc[r][c], 0.0f);
    }
  }

  for (int k0 = 0; k0 < K; k0 += BK) {
    // Vectorized A tile copy: each transaction copies 8 half values = 16 bytes.
    constexpr int AVEC_COLS = BM / 8;
    for (int t = tid; t < BK * AVEC_COLS; t += THREADS) {
      int kk = t / AVEC_COLS;
      int vi = t - kk * AVEC_COLS;
      int mi = vi * 8;
      uint4 v = *reinterpret_cast<const uint4 *>(A + int64_t(k0 + kk) * M + (m0 + mi));
      *reinterpret_cast<uint4 *>(&As[kk * STRIDE_A + mi]) = v;
    }

    // Vectorized B tile copy.
    constexpr int BVEC_COLS = BN / 8;
    for (int t = tid; t < BK * BVEC_COLS; t += THREADS) {
      int kk = t / BVEC_COLS;
      int vi = t - kk * BVEC_COLS;
      int ni = vi * 8;
      uint4 v = *reinterpret_cast<const uint4 *>(Bp + int64_t(k0 + kk) * N + (n0 + ni));
      *reinterpret_cast<uint4 *>(&Bs[kk * STRIDE_B + ni]) = v;
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[WM];
#pragma unroll
      for (int r = 0; r < WM; ++r) {
        int mi = warp_m * (16 * WM) + r * 16;
        wmma::load_matrix_sync(a_frag[r], &As[kk * STRIDE_A + mi], BM + PAD);
      }

      wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[WN];
#pragma unroll
      for (int c = 0; c < WN; ++c) {
        int ni = warp_n * (16 * WN) + c * 16;
        wmma::load_matrix_sync(b_frag[c], &Bs[kk * STRIDE_B + ni], BN + PAD);
      }

#pragma unroll
      for (int r = 0; r < WM; ++r) {
#pragma unroll
        for (int c = 0; c < WN; ++c) {
          wmma::mma_sync(acc[r][c], a_frag[r], b_frag[c], acc[r][c]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < WM; ++r) {
#pragma unroll
    for (int c = 0; c < WN; ++c) {
      int mi = warp_m * (16 * WM) + r * 16;
      int ni = warp_n * (16 * WN) + c * 16;
      wmma::store_matrix_sync(&C[int64_t(n0 + ni) * M + (m0 + mi)],
                              acc[r][c], M, wmma::mem_col_major);
    }
  }
}

template<int BM, int BN, int BK, int PAD, int WM, int WN>
static float run_variant(const char *name, int M, int N, int K,
                         const half *Ah, const half *Bh, float *C,
                         int Nt, double gflop_count, bool print=true) {
  constexpr int WARPS = (BM / (16 * WM)) * (BN / (16 * WN));
  constexpr int THREADS = WARPS * 32;
  static_assert(WARPS >= 1 && WARPS <= 32, "too many warps per CTA");
  dim3 block(THREADS);
  dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

  if ((M % BM) || (N % BN) || (K % BK)) {
    if (print) printf("%-30s skipped: dimensions not multiples of tile\n", name);
    return 1e30f;
  }

  auto kernel = wmma_gemm_vec_kernel<BM,BN,BK,PAD,WM,WN>;
  constexpr int smem_bytes = int(BK * (BM + PAD + BN + PAD) * sizeof(half));
  CHECK_CUDA(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                  cudaSharedmemCarveoutMaxShared));
  CHECK_CUDA(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem_bytes));

  for (int i = 0; i < 2; ++i) kernel<<<grid, block, smem_bytes>>>(M, N, K, Ah, Bh, C);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < Nt; ++i) kernel<<<grid, block, smem_bytes>>>(M, N, K, Ah, Bh, C);
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  CHECK_CUDA(cudaGetLastError());

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  float sec = ms * 1.0e-3f / Nt;
  if (print) {
    printf("%-30s %9.2f Gflops  time %.6f s  grid(%d,%d) block %d\n",
           name, gflop_count / sec / 1.0e9, sec, grid.x, grid.y, block.x);
  }
  return sec;
}

static float benchmark_cublas(int M, int N, int K, const float *A, const float *B,
                              float *C, int Nt) {
  cublasHandle_t h;
  CHECK_CUBLAS(cublasCreate(&h));
  const float alpha = 1.0f, beta = 0.0f;

  for (int i = 0; i < 2; ++i) {
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K, &alpha,
                              A, CUDA_R_32F, M,
                              B, CUDA_R_32F, K,
                              &beta,
                              C, CUDA_R_32F, M,
                              CUBLAS_COMPUTE_32F_FAST_16F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < Nt; ++i) {
    CHECK_CUBLAS(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K, &alpha,
                              A, CUDA_R_32F, M,
                              B, CUDA_R_32F, K,
                              &beta,
                              C, CUDA_R_32F, M,
                              CUBLAS_COMPUTE_32F_FAST_16F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUBLAS(cublasDestroy(h));
  return ms * 1.0e-3f / Nt;
}

struct BestResult {
  const char *name;
  float sec;
};

#define TRY(BM,BN,BK,PAD,WM,WN) do {                                      \
  float sec = run_variant<BM,BN,BK,PAD,WM,WN>(                             \
      #BM "x" #BN "x" #BK " p" #PAD " w" #WM "x" #WN,                  \
      M, N, K, Ah, Bh, C_my, Nt, gflop_count);                             \
  if (sec < best.sec) {                                                     \
    best.sec = sec;                                                         \
    best.name = #BM "x" #BN "x" #BK " p" #PAD " w" #WM "x" #WN;        \
  }                                                                         \
} while (0)

int main(int argc, char **argv) {
  const int M  = (argc > 1) ? std::atoi(argv[1]) : 10240;
  const int K  = (argc > 2) ? std::atoi(argv[2]) : 4096;
  const int N  = (argc > 3) ? std::atoi(argv[3]) : 8192;
  const int Nt = (argc > 4) ? std::atoi(argv[4]) : 10;

  int dev = 0;
  CHECK_CUDA(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
  printf("GPU: %s\n", prop.name);
  printf("M=%d N=%d K=%d Nt=%d\n", M, N, K, Nt);

  float *A, *B, *C_ref, *C_my;
  half *Ah, *Bh;
  CHECK_CUDA(cudaMallocManaged(&A,     int64_t(M) * K * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&B,     int64_t(K) * N * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&C_ref, int64_t(M) * N * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&C_my,  int64_t(M) * N * sizeof(float)));
  CHECK_CUDA(cudaMallocManaged(&Ah,    int64_t(M) * K * sizeof(half)));
  CHECK_CUDA(cudaMallocManaged(&Bh,    int64_t(K) * N * sizeof(half)));

  srand48(1);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < K; ++j)
      A[int64_t(i) * K + j] = float(drand48());
  for (int i = 0; i < K; ++i)
    for (int j = 0; j < N; ++j)
      B[int64_t(i) * N + j] = float(drand48());
  for (int64_t i = 0; i < int64_t(M) * N; ++i) C_ref[i] = C_my[i] = 0.0f;

  CHECK_CUDA(prefetch_to_device(A,     int64_t(M) * K * sizeof(float), dev));
  CHECK_CUDA(prefetch_to_device(B,     int64_t(K) * N * sizeof(float), dev));
  CHECK_CUDA(prefetch_to_device(C_ref, int64_t(M) * N * sizeof(float), dev));
  CHECK_CUDA(prefetch_to_device(C_my,  int64_t(M) * N * sizeof(float), dev));
  CHECK_CUDA(prefetch_to_device(Ah,    int64_t(M) * K * sizeof(half), dev));
  CHECK_CUDA(prefetch_to_device(Bh,    int64_t(K) * N * sizeof(half), dev));

  const int conv_threads = 256;
  const int conv_blocks = 1024;
  float_to_half_raw_kernel<<<conv_blocks, conv_threads>>>(A, Ah, int64_t(M) * K);
  pack_B_to_half_rowmajor_kernel<<<conv_blocks, conv_threads>>>(B, Bh, K, N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  const double gflop_count = 2.0 * double(M) * double(N) * double(K);

  float tcublas = benchmark_cublas(M, N, K, A, B, C_ref, Nt);
  printf("cuBLAS reference             %9.2f Gflops  time %.6f s\n\n",
         gflop_count / tcublas / 1.0e9, tcublas);

  printf("Custom WMMA variants, vectorized gmem->smem copies:\n");
  BestResult best{"none", 1e30f};

  TRY(128,128,64,8,2,4);
  TRY(128,128,32,8,2,4);
  TRY(128,128,64,8,1,4);
  TRY(128,128,64,8,2,2);
  TRY(128,128,64,8,4,1);
  TRY(128,128,32,8,1,4);
  TRY(128,128,32,8,2,2);
  TRY(128,128,32,8,4,1);
  TRY(64,128,64,8,1,2);
  TRY(128,64,64,8,1,2);

  TRY(64,128,64,8,1,4);
  TRY(64,128,64,8,2,2);
  TRY(128,64,64,8,2,2);
  TRY(128,64,64,8,4,1);
  TRY(64,256,64,8,1,4);
  TRY(128,256,64,8,2,4);
  TRY(256,128,64,8,4,2);

  TRY(128,128,64,0,2,4);
  TRY(128,128,64,0,1,4);
  TRY(128,128,64,0,2,2);

  printf("\nBest custom variant: %s, %.2f Gflops, time %.6f s\n",
         best.name, gflop_count / best.sec / 1.0e9, best.sec);

  CHECK_CUDA(prefetch_to_host(C_ref, int64_t(M) * N * sizeof(float)));
  CHECK_CUDA(prefetch_to_host(C_my,  int64_t(M) * N * sizeof(float)));
  CHECK_CUDA(cudaDeviceSynchronize());

  double err = 0.0, refsum = 0.0;
  for (int64_t i = 0; i < int64_t(M) * N; ++i) {
    err += std::fabs(double(C_ref[i]) - double(C_my[i]));
    refsum += std::fabs(double(C_ref[i]));
  }
  printf("mean_abs_error_vs_cuBLAS: %.8e\n", err / double(int64_t(M) * N));
  printf("relative_L1_error_vs_cuBLAS: %.8e\n", err / refsum);

  CHECK_CUDA(cudaFree(A));
  CHECK_CUDA(cudaFree(B));
  CHECK_CUDA(cudaFree(C_ref));
  CHECK_CUDA(cudaFree(C_my));
  CHECK_CUDA(cudaFree(Ah));
  CHECK_CUDA(cudaFree(Bh));
  return 0;
}

