#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void bucket_count(int *bucket, int *key, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  atomicAdd(&bucket[key[i]], 1);
}

int main() {
  int n = 50;
  int range = 5;

  int *key;
  int *bucket;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));

  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  for (int i = 0; i < range; i++) {
    bucket[i] = 0;
  }

  const int blockSize = 128;
  const int gridSize  = (n + blockSize - 1) / blockSize;
  bucket_count<<<gridSize, blockSize>>>(bucket, key, n);
  cudaDeviceSynchronize();

  for (int i = 0, j = 0; i < range; i++) {
    for (; bucket[i] > 0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i = 0; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}
