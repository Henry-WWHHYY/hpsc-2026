#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 xv = _mm512_load_ps(x);
  __m512 yv = _mm512_load_ps(y);
  __m512 mv = _mm512_load_ps(m);

  for(int i=0; i<N; i++) {
    __m512 xi  = _mm512_set1_ps(x[i]);
    __m512 yi  = _mm512_set1_ps(y[i]);
    __m512 rx  = _mm512_sub_ps(xi, xv);
    __m512 ry  = _mm512_sub_ps(yi, yv);
    __m512 r2  = _mm512_add_ps(_mm512_mul_ps(rx, rx),
                               _mm512_mul_ps(ry, ry));
    __mmask16 mask = _mm512_cmpneq_epi32_mask(
        _mm512_set1_epi32(i),
        _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15));
    __m512 r       = _mm512_sqrt_ps(r2);
    __m512 r3      = _mm512_mul_ps(r2, r);
    __m512 invr3   = _mm512_maskz_div_ps(mask, mv, r3);
    __m512 fxv     = _mm512_mul_ps(rx, invr3);
    __m512 fyv     = _mm512_mul_ps(ry, invr3);
    fx[i] = -_mm512_reduce_add_ps(fxv);
    fy[i] = -_mm512_reduce_add_ps(fyv);
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
