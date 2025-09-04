#include <cuda_runtime.h>

// input, gamma, beta, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, 
                     float* output, int N, float eps) {

}
